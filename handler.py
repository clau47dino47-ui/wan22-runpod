"""
RunPod Serverless Handler — Wan 2.2 I2V (Image-to-Video)
GPU: RTX 4090 (24 GB VRAM)

Input schema:
{
  "input": {
    "prompt":        "string (required)",
    "image_url":     "https://... (required for I2V)",
    "resolution":    "480p" | "720p"  (default: "480p"),
    "num_frames":    24-81            (default: 49),
    "steps":         10-50            (default: 25),
    "guidance_scale": 5.0-9.0        (default: 7.0),
    "callback_url":  "https://..."   (optional webhook)
  }
}

Output:
{
  "video_url": "https://r2.../video.mp4",
  "duration_s": 5.1
}
"""

import os
import io
import time
import uuid
import tempfile
import logging
import requests
import runpod
import torch
import boto3
from botocore.config import Config
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-480P")
R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"]
R2_BUCKET       = os.environ["R2_BUCKET"]
R2_ACCESS_KEY   = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY   = os.environ["R2_SECRET_KEY"]

RESOLUTION_MAP = {
    "480p": (832, 480),
    "720p": (1280, 720),
}

# ── Load model (once at cold start) ──────────────────────────────────────────
log.info(f"Loading {MODEL_ID} ...")
t0 = time.time()

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,         # FP16 — risparmia ~7 GB vs FP32
)

# CPU offload: sposta automaticamente i layer non usati sulla RAM
pipe.enable_model_cpu_offload()

# VAE tiling: evita picchi OOM durante la decodifica dei frame ad alta res
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

log.info(f"Model loaded in {time.time() - t0:.1f}s")

# ── R2 / S3 client ────────────────────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="auto",
)

def upload_video(local_path: str) -> str:
    key = f"generated/{uuid.uuid4()}.mp4"
    with open(local_path, "rb") as f:
        s3.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=f,
            ContentType="video/mp4",
        )
    # URL pre-firmato valido 24 ore
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": key},
        ExpiresIn=86400,
    )
    log.info(f"Uploaded {key}")
    return url

# ── Handler ───────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    prompt       = job_input.get("prompt", "")
    image_url    = job_input.get("image_url")
    resolution   = job_input.get("resolution", "480p")
    num_frames   = int(job_input.get("num_frames", 49))
    steps        = int(job_input.get("steps", 25))
    guidance     = float(job_input.get("guidance_scale", 7.0))
    callback_url = job_input.get("callback_url")

    if not prompt:
        return {"error": "prompt is required"}
    if not image_url:
        return {"error": "image_url is required for I2V"}

    width, height = RESOLUTION_MAP.get(resolution, (832, 480))

    # Scarica e prepara l'immagine sorgente
    log.info(f"Downloading source image from {image_url[:80]}...")
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((width, height))

    log.info(f"Generating {num_frames} frames @ {width}x{height}, steps={steps}")
    t0 = time.time()

    with torch.inference_mode():
        output = pipe(
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        )

    elapsed = time.time() - t0
    log.info(f"Generation done in {elapsed:.1f}s")

    # Esporta in mp4 e carica su R2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    export_to_video(output.frames[0], tmp_path, fps=16)
    video_url = upload_video(tmp_path)
    os.unlink(tmp_path)

    result = {
        "video_url":  video_url,
        "duration_s": round(elapsed, 1),
        "frames":     num_frames,
        "resolution": resolution,
    }

    # Webhook opzionale
    if callback_url:
        try:
            requests.post(callback_url, json=result, timeout=10)
            log.info(f"Webhook sent to {callback_url}")
        except Exception as e:
            log.warning(f"Webhook failed: {e}")

    return result


runpod.serverless.start({"handler": handler})
