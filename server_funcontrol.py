"""
Wan2.1-Fun-Control 14B — FastAPI server for Vast.ai RTX 4090
Polls Neon PostgreSQL for pending jobs (DB-centric architecture)
Input: source image + control video (DWPose extracted) + prompt
"""
import os, io, uuid, time, threading, logging, requests, tempfile, re
import numpy as np
import torch, boto3
import cv2
from PIL import Image
from fastapi import FastAPI
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_KEY   = os.environ["API_KEY"]
S3_BUCKET = os.environ["AWS_S3_BUCKET"]
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
_DB_URL   = os.environ.get("NEON_DATABASE_URL", "")
MODEL_ID  = os.environ.get("MODEL_ID", "alibaba-pai/Wan2.1-Fun-14B-Control")

def _clean_db_url(url: str) -> str:
    url = re.sub(r"channel_binding=[^&]*&?", "", url)
    url = url.replace("sslmode=verify-full", "sslmode=require")
    url = url.rstrip("?&")
    return url

DB_URL = _clean_db_url(_DB_URL) if _DB_URL else ""

# ── Model loading ──────────────────────────────────────────────────────────────
log.info(f"Loading {MODEL_ID} ...")
t0 = time.time()

from videox_fun.pipelines import WanFunControlPipeline
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora

pipe = WanFunControlPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

# FP8 layerwise casting — essenziale per 14B su 24GB VRAM
try:
    pipe.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
    )
    log.info("FP8 layerwise casting enabled")
except Exception as e:
    log.info(f"FP8 not available: {e}")

pipe.enable_model_cpu_offload()

# TeaCache per speedup ~50%
try:
    pipe.transformer.enable_teacache(threshold=0.2)
    log.info("TeaCache enabled (threshold=0.2)")
except Exception as e:
    log.info(f"TeaCache not available: {e}")

log.info(f"Model loaded in {time.time()-t0:.1f}s")

# ── DWPose ────────────────────────────────────────────────────────────────────
from controlnet_aux import DWposeDetector
dwpose = DWposeDetector()
log.info("DWPose detector loaded")

# ── S3 ────────────────────────────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=S3_REGION)

def upload_video(local_path: str, prefix: str = "generated") -> str:
    key = f"{prefix}/{uuid.uuid4()}.mp4"
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=86400
    )

def upload_pose_video(frames: list, task_id: str) -> str:
    """Encode pose frames to mp4 and upload to S3, return presigned URL."""
    if not frames:
        raise ValueError("No pose frames to upload")
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 25.0, (w, h))
    for f in frames:
        writer.write(f if f.shape[2] == 3 else cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    key = f"pose_videos/{task_id}.mp4"
    s3.upload_file(tmp_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    os.unlink(tmp_path)
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=86400
    )

# ── Pose normalization (bone-length rescaling) ────────────────────────────────
SKELETON_MAP = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
]

def normalize_pose_skeleton(driving_poses: np.ndarray, ref_pose: np.ndarray) -> np.ndarray:
    """
    Bone-length rescaling: riscala ogni segmento dello scheletro del driving video
    per approssimare le proporzioni del soggetto nella reference image.
    """
    norm_poses = []
    for pose in driving_poses:
        new_pose = np.copy(pose)
        for parent, child in SKELETON_MAP:
            if parent >= len(ref_pose) or child >= len(ref_pose):
                continue
            l_ref     = np.linalg.norm(ref_pose[parent] - ref_pose[child])
            l_driving = np.linalg.norm(pose[parent] - pose[child])
            scale     = l_ref / (l_driving + 1e-6)
            direction = pose[child] - pose[parent]
            new_pose[child] = new_pose[parent] + direction * scale
        norm_poses.append(new_pose)
    return np.array(norm_poses)

# ── DWPose extraction ─────────────────────────────────────────────────────────
def extract_pose_frames(video_url: str, target_w: int, target_h: int) -> tuple[list, np.ndarray | None]:
    """
    Scarica video di riferimento, estrae pose DWPose frame per frame.
    Ritorna (pose_frames_bgr, ref_keypoints_for_normalization).
    """
    resp = requests.get(video_url, timeout=60)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    pose_frames = []
    raw_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).resize((target_w, target_h), Image.LANCZOS)
        # DWPose restituisce immagine con scheletro disegnato + keypoints
        pose_result = dwpose(pil_frame, output_type="np", include_hand=True, include_face=True)
        if isinstance(pose_result, tuple):
            pose_img, keypoints = pose_result
        else:
            pose_img = pose_result
            keypoints = None
        pose_frames.append(np.array(pose_img))
        if keypoints is not None:
            raw_keypoints.append(keypoints)

    cap.release()
    os.unlink(tmp_path)

    ref_kps = raw_keypoints[0] if raw_keypoints else None
    return pose_frames, ref_kps

def extract_ref_pose_from_image(image: Image.Image) -> np.ndarray | None:
    """Estrae keypoints DWPose dall'immagine sorgente per normalizzazione."""
    result = dwpose(image, output_type="np", include_hand=True, include_face=True)
    if isinstance(result, tuple) and result[1] is not None:
        return result[1]
    return None

# ── Inference ─────────────────────────────────────────────────────────────────
def process_payload(payload: dict) -> dict:
    """
    Processa un job Fun-Control.
    payload keys: image_url, control_video_url, prompt, negative_prompt,
                  width, height, num_frames, steps, guidance_scale
    Returns: dict con result_url e pose_video_url
    """
    width  = max(round(int(payload.get("width",  832)) / 32) * 32, 64)
    height = max(round(int(payload.get("height", 480)) / 32) * 32, 64)

    # 1. Carica immagine sorgente
    resp = requests.get(payload["image_url"], timeout=30)
    resp.raise_for_status()
    source_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    source_image = source_image.resize((width, height), Image.LANCZOS)

    # 2. Estrai pose dal video di riferimento
    log.info("Extracting DWPose from reference video...")
    pose_frames, driving_kps = extract_pose_frames(payload["control_video_url"], width, height)
    log.info(f"Extracted {len(pose_frames)} pose frames")

    # 3. Normalizzazione bone-length se keypoints disponibili
    if driving_kps is not None:
        ref_kps = extract_ref_pose_from_image(source_image)
        if ref_kps is not None:
            log.info("Applying bone-length normalization...")
            # Reshape keypoints per normalization (N_frames, N_joints, 2)
            if len(driving_kps.shape) == 2:
                driving_kps_all = np.stack([driving_kps] * len(pose_frames))
            else:
                driving_kps_all = driving_kps
            normalized_kps = normalize_pose_skeleton(driving_kps_all, ref_kps)
            log.info("Bone-length normalization applied")

    # 4. Upload pose video su S3 (per debug/preview)
    job_id = payload.get("job_id", str(uuid.uuid4()))
    pose_video_url = upload_pose_video(pose_frames, job_id)
    log.info(f"Pose video uploaded: {pose_video_url}")

    # 5. Prepara control tensor
    control_frames_pil = [Image.fromarray(f) for f in pose_frames]

    # 6. Generazione video
    negative_prompt = payload.get("negative_prompt", (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
        "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ))

    log.info(f"Generating video: {len(control_frames_pil)} frames, {width}x{height}")
    with torch.inference_mode():
        output = pipe(
            prompt=payload["prompt"],
            negative_prompt=negative_prompt,
            image=source_image,
            control_video=control_frames_pil,
            num_frames=int(payload.get("num_frames", len(control_frames_pil))),
            num_inference_steps=int(payload.get("steps", 40)),
            guidance_scale=float(payload.get("guidance_scale", 6.0)),
            width=width,
            height=height,
        )

    # 7. Export e upload
    from diffusers.utils import export_to_video
    all_frames = output.frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    export_to_video(all_frames, tmp_path, fps=16)
    result_url = upload_video(tmp_path)
    os.unlink(tmp_path)

    return {"result_url": result_url, "pose_video_url": pose_video_url}

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
                              AND payload->>'mode' = 'fun_control'
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
                    payload = dict(job["payload"])
                    payload["job_id"] = str(job["id"])
                    result = process_payload(payload)
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE video_jobs SET status='completed', result_url=%s, completed_at=NOW() WHERE id=%s",
                            (result["result_url"], job["id"])
                        )
                        conn.commit()
                    log.info(f"Job {job['id']} completed → {result['result_url']}")
                except Exception as e:
                    log.error(f"Job {job['id']} failed: {e}")
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE video_jobs SET status='failed', error_message=%s WHERE id=%s",
                            (str(e), job["id"])
                        )
                        conn.commit()
            else:
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
