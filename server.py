"""
Wan 2.2 TI2V-5B — FastAPI server for Vast.ai RTX 4090
Polls Neon PostgreSQL for pending jobs (DB-centric architecture)
"""
import os, io, uuid, time, threading, logging, requests, tempfile, re
import torch, boto3
from PIL import Image, ImageFilter
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_KEY   = os.environ["API_KEY"]
MODEL_ID  = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
S3_BUCKET = os.environ["AWS_S3_BUCKET"]
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
_DB_URL   = os.environ.get("NEON_DATABASE_URL", "")

# Neon DB URL: strip channel_binding (unsupported by psycopg2) and downgrade to sslmode=require
def _clean_db_url(url: str) -> str:
    url = re.sub(r"channel_binding=[^&]*&?", "", url)
    url = url.replace("sslmode=verify-full", "sslmode=require")
    url = url.rstrip("?&")
    return url

DB_URL = _clean_db_url(_DB_URL) if _DB_URL else ""

# ── Model loading ──────────────────────────────────────────────────────────────
log.info(f"Loading {MODEL_ID} ...")
t0 = time.time()
pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

# FP8 layerwise casting to reduce peak VRAM on 24GB (RTX 4090)
try:
    pipe.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    log.info("FP8 layerwise casting enabled")
except Exception:
    log.info("FP8 not available, using bfloat16")

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# TeaCache: 50% speedup if supported by this diffusers build
try:
    pipe.transformer.enable_teacache(threshold=0.25)
    log.info("TeaCache enabled (threshold=0.25)")
except Exception:
    log.info("TeaCache not available")

# flow_shift=3.0 for 480p (reduces noise density mismatch at low resolution)
try:
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, flow_shift=3.0)
    log.info("flow_shift=3.0 set for 480p")
except Exception as e:
    log.info(f"flow_shift not applied: {e}")

log.info(f"Model loaded in {time.time()-t0:.1f}s")

# Default negative prompt (Chinese — matches model training distribution)
_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# ── S3 ────────────────────────────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=S3_REGION)

def upload_video(local_path: str) -> str:
    key = f"generated/{uuid.uuid4()}.mp4"
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=86400
    )

# ── Inference ─────────────────────────────────────────────────────────────────
def process_payload(payload: dict) -> str:
    resp = requests.get(payload["image_url"], timeout=30)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    width  = int(payload.get("width",  832))
    height = int(payload.get("height", 480))

    # Wan 2.2 requires dimensions divisible by 32
    width  = max(round(width  / 32) * 32, 64)
    height = max(round(height / 32) * 32, 64)

    image = image.resize((width, height), Image.LANCZOS)

    # ALG workaround: slight blur reduces high-frequency image dominance,
    # allowing the text prompt to steer generation more effectively
    image = image.filter(ImageFilter.GaussianBlur(radius=1.2))

    with torch.inference_mode():
        output = pipe(
            image=image,
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt", _NEGATIVE_PROMPT),
            num_frames=int(payload.get("num_frames", 121)),
            num_inference_steps=int(payload.get("steps", 40)),
            guidance_scale=float(payload.get("guidance_scale", 5.0)),
            width=width,
            height=height,
            max_sequence_length=int(payload.get("max_sequence_length", 512)),
        )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    export_to_video(output.frames[0], tmp_path, fps=24)
    url = upload_video(tmp_path)
    os.unlink(tmp_path)
    return url

# ── DB worker ─────────────────────────────────────────────────────────────────
_last_job_time = time.time()
_idle_shutdown_minutes = int(os.environ.get("IDLE_SHUTDOWN_MINUTES", "0"))

def db_worker():
    global _last_job_time
    while True:
        if not DB_URL:
            time.sleep(10)
            continue
        conn = None
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE video_jobs
                       SET status = 'in_progress', started_at = NOW()
                     WHERE id = (
                           SELECT id FROM video_jobs
                            WHERE status = 'pending'
                            ORDER BY created_at ASC
                            LIMIT 1
                            FOR UPDATE SKIP LOCKED
                     )
                     RETURNING *
                """)
                job = cur.fetchone()
                conn.commit()

            if job:
                _last_job_time = time.time()
                log.info(f"Processing job {job['id']}")
                try:
                    result_url = process_payload(job["payload"])
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE video_jobs SET status='completed', result_url=%s, completed_at=NOW() WHERE id=%s",
                            (result_url, job["id"])
                        )
                        conn.commit()
                    log.info(f"Job {job['id']} completed")
                except Exception as e:
                    log.error(f"Job {job['id']} failed: {e}")
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE video_jobs SET status='failed', error_message=%s WHERE id=%s",
                            (str(e), job["id"])
                        )
                        conn.commit()
            else:
                # Check idle shutdown
                if _idle_shutdown_minutes > 0:
                    idle = (time.time() - _last_job_time) / 60
                    if idle >= _idle_shutdown_minutes:
                        log.info(f"Idle for {idle:.1f}min, shutting down")
                        os.system("vastai destroy instance $(cat /etc/vast_instance_id 2>/dev/null || echo '') &")
                        time.sleep(5)
                        os._exit(0)
                time.sleep(5)
        except Exception as e:
            log.error(f"DB worker error: {e}")
            time.sleep(15)
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

threading.Thread(target=db_worker, daemon=True).start()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
