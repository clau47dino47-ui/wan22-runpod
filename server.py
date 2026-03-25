"""
Wan 2.2 I2V — FastAPI server for Vast.ai persistent GPU instance
"""
import os, io, uuid, time, threading, queue, logging, requests, tempfile
import torch, boto3
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_KEY   = os.environ["API_KEY"]
MODEL_ID  = os.environ.get("MODEL_ID", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
S3_BUCKET = os.environ["AWS_S3_BUCKET"]
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")

RESOLUTION_MAP = {"480p": (832, 480), "720p": (1280, 720)}

def fit_to_resolution(orig_w: int, orig_h: int, max_pixels: int) -> tuple[int, int]:
    """Scale to fit within max_pixels while preserving aspect ratio. Round to multiple of 32."""
    ratio = orig_w / orig_h
    h = int((max_pixels / ratio) ** 0.5)
    w = int(h * ratio)
    w = max(round(w / 32) * 32, 64)
    h = max(round(h / 32) * 32, 64)
    return w, h

log.info(f"Loading {MODEL_ID} ...")
t0 = time.time()
pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
log.info(f"Model loaded in {time.time()-t0:.1f}s")

s3 = boto3.client("s3", region_name=S3_REGION)

def upload_video(local_path: str) -> str:
    key = f"generated/{uuid.uuid4()}.mp4"
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=86400
    )

# ── Job queue ─────────────────────────────────────────────────────────────────
job_queue: queue.Queue = queue.Queue()
job_store: dict[str, dict] = {}

def worker():
    while True:
        job_id, payload = job_queue.get()
        job_store[job_id]["status"] = "IN_PROGRESS"
        try:
            max_pixels = {"480p": 832 * 480, "720p": 1280 * 720}.get(payload.get("resolution", "480p"), 832 * 480)
            resp = requests.get(payload["image_url"], timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            width, height = fit_to_resolution(image.width, image.height, max_pixels)
            image = image.resize((width, height), Image.LANCZOS)
            with torch.inference_mode():
                output = pipe(
                    image=image,
                    prompt=payload["prompt"],
                    num_frames=int(payload.get("num_frames", 49)),
                    num_inference_steps=int(payload.get("steps", 25)),
                    guidance_scale=float(payload.get("guidance_scale", 7.0)),
                    width=width, height=height,
                )
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            export_to_video(output.frames[0], tmp_path, fps=16)
            video_url = upload_video(tmp_path)
            os.unlink(tmp_path)
            job_store[job_id] = {"status": "COMPLETED", "video_url": video_url}
            log.info(f"Job {job_id} completed")
        except Exception as e:
            log.error(f"Job {job_id} failed: {e}")
            job_store[job_id] = {"status": "FAILED", "error": str(e)}

threading.Thread(target=worker, daemon=True).start()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()
security = HTTPBearer()

def verify(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != API_KEY:
        raise HTTPException(status_code=401)

class JobRequest(BaseModel):
    prompt: str
    image_url: str | None = None
    resolution: str = "480p"
    num_frames: int = 49
    steps: int = 25
    guidance_scale: float = 7.0

@app.post("/jobs")
def create_job(req: JobRequest, _=Depends(verify)):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "IN_QUEUE"}
    job_queue.put((job_id, req.model_dump()))
    return {"job_id": job_id, "status": "IN_QUEUE"}

@app.get("/jobs/{job_id}")
def get_job(job_id: str, _=Depends(verify)):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404)
    return job

@app.get("/health")
def health():
    return {"status": "ok", "queue": job_queue.qsize()}
