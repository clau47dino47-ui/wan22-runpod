"""
RunPod Serverless Handler — Wan 2.2 I2V (Image-to-Video)
GPU: RTX 4090 (24 GB VRAM) — uploads to AWS S3
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
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ID   = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-480P")
S3_BUCKET  = os.environ["AWS_S3_BUCKET"]
S3_REGION  = os.environ.get("AWS_REGION", "us-east-1")

RESOLUTION_MAP = {
    "480p": (832, 480),
    "720p": (1280, 720),
}

# ── Load model once at cold start ─────────────────────────────────────────────
log.info(f"Loading {MODEL_ID} ...")
t0 = time.time()

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

log.info(f"Model loaded in {time.time() - t0:.1f}s")

# ── S3 client ─────────────────────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=S3_REGION)

def upload_video(local_path: str) -> str:
    key = f"generated/{uuid.uuid4()}.mp4"
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=86400,
    )
    log.info(f"Uploaded s3://{S3_BUCKET}/{key}")
    return url

# ── Handler ───────────────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    inp = job.get("input", {})

    prompt      = inp.get("prompt", "")
    image_url   = inp.get("image_url")
    resolution  = inp.get("resolution", "480p")
    num_frames  = int(inp.get("num_frames", 49))
    steps       = int(inp.get("steps", 25))
    guidance    = float(inp.get("guidance_scale", 7.0))

    if not prompt:
        return {"error": "prompt is required"}
    if not image_url:
        return {"error": "image_url is required"}

    width, height = RESOLUTION_MAP.get(resolution, (832, 480))

    log.info(f"Downloading image from {image_url[:80]}...")
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((width, height))

    log.info(f"Generating {num_frames} frames @ {width}x{height}")
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
    log.info(f"Done in {elapsed:.1f}s")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    export_to_video(output.frames[0], tmp_path, fps=16)
    video_url = upload_video(tmp_path)
    os.unlink(tmp_path)

    return {
        "video_url":  video_url,
        "duration_s": round(elapsed, 1),
        "frames":     num_frames,
        "resolution": resolution,
    }


runpod.serverless.start({"handler": handler})
