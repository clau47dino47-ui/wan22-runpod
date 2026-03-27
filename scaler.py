"""
Wan 2.2 — Vast.ai RTX 4090 auto-scaler
Monitors Neon PostgreSQL for pending video_jobs and creates/destroys
interruptible RTX 4090 instances accordingly.

Usage:
  python scaler.py

Required env vars:
  NEON_DATABASE_URL
  VASTAI_ACCOUNT_API_KEY
  AWS_S3_BUCKET, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  API_KEY  (passed to worker server as VASTAI_API_KEY)

Optional:
  VASTAI_IMAGE          (default: ghcr.io/clau47dino47-ui/wan22-runpod:vastai)
  MAX_PRICE_PER_HOUR    (default: 1.50)
  MIN_DOWNLOAD_MBPS     (default: 10000)
  IDLE_SHUTDOWN_MINUTES (default: 10, passed to worker)
  SCALE_CHECK_INTERVAL  (default: 15 seconds)
"""
import os, re, json, time, subprocess, logging, psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_DB_URL      = os.environ["NEON_DATABASE_URL"]
VASTAI_KEY   = os.environ["VASTAI_ACCOUNT_API_KEY"]
API_KEY      = os.environ["API_KEY"]
IMAGE        = os.environ.get("VASTAI_IMAGE", "ghcr.io/clau47dino47-ui/wan22-runpod:vastai")
MAX_PRICE    = float(os.environ.get("MAX_PRICE_PER_HOUR", "1.50"))
MIN_DL       = int(os.environ.get("MIN_DOWNLOAD_MBPS", "10000"))
IDLE_MIN     = int(os.environ.get("IDLE_SHUTDOWN_MINUTES", "10"))
CHECK_SECS   = int(os.environ.get("SCALE_CHECK_INTERVAL", "15"))

# Blacklisted machine IDs (never create on these)
BLACKLISTED_MACHINES = {
    39959, 34072, 11611, 44039, 25723, 43995,
    58565, 30574, 57421, 56764,
}
# Blacklisted host IDs
BLACKLISTED_HOSTS = {94202}
# Blacklisted datacenter IDs
BLACKLISTED_DATACENTERS = {125728}

VASTAI_CLI = os.path.expanduser("~/.local/bin/vastai")

# ── Neon URL cleanup (psycopg2 compat) ───────────────────────────────────────
def _clean_db_url(url: str) -> str:
    url = re.sub(r"channel_binding=[^&]*&?", "", url)
    url = url.replace("sslmode=verify-full", "sslmode=require")
    url = url.rstrip("?&")
    return url

DB_URL = _clean_db_url(_DB_URL)

# ── State ─────────────────────────────────────────────────────────────────────
_active_instance_id: int | None = None

# ── DB helpers ────────────────────────────────────────────────────────────────
def count_pending() -> int:
    conn = psycopg2.connect(DB_URL, connect_timeout=10)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM video_jobs WHERE status IN ('pending', 'in_progress')")
            return cur.fetchone()[0]
    finally:
        conn.close()

# ── Vast.ai helpers ───────────────────────────────────────────────────────────
def _vastai(*args) -> str:
    cmd = [VASTAI_CLI, "--api-key", VASTAI_KEY] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args[:2])}: {result.stderr.strip()}")
    return result.stdout.strip()

def find_rtx4090_offer() -> int | None:
    """Return the cheapest valid interruptible RTX 4090 offer ID."""
    raw = _vastai(
        "search", "offers",
        "--type", "interruptible",
        "--order", "dph_total asc",
        "--raw",
        f"gpu_name=RTX_4090 num_gpus=1 dph_total<={MAX_PRICE} inet_down>={MIN_DL}",
    )
    try:
        offers = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Could not parse offers JSON")
        return None

    for offer in offers:
        mid = int(offer.get("machine_id", 0))
        hid = int(offer.get("host_id", 0))
        did = int(offer.get("datacenter_id") or 0)
        if mid in BLACKLISTED_MACHINES:
            continue
        if hid in BLACKLISTED_HOSTS:
            continue
        if did in BLACKLISTED_DATACENTERS:
            continue
        log.info(f"Found offer {offer['id']} machine={mid} price=${offer['dph_total']:.3f}/hr dl={offer.get('inet_down','?')} MB/s")
        return int(offer["id"])

    log.warning("No suitable RTX 4090 offer found")
    return None

def create_instance(offer_id: int) -> int | None:
    """Create a Vast.ai instance and return its instance ID."""
    s3_env = (
        f"-e API_KEY={API_KEY} "
        f"-e NEON_DATABASE_URL={_DB_URL} "
        f"-e AWS_S3_BUCKET={os.environ['AWS_S3_BUCKET']} "
        f"-e AWS_REGION={os.environ.get('AWS_REGION', 'us-east-1')} "
        f"-e AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']} "
        f"-e AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']} "
        f"-e IDLE_SHUTDOWN_MINUTES={IDLE_MIN}"
    )
    onstart = (
        "env >> /etc/environment; "
        "python3.10 -m uvicorn server:app --host 0.0.0.0 --port 8000"
    )
    out = _vastai(
        "create", "instance", str(offer_id),
        "--image", IMAGE,
        "--disk", "55",
        "--env", s3_env,
        "--ssh", "--direct",
        "--onstart-cmd", onstart,
    )
    log.info(f"create instance output: {out}")
    try:
        data = json.loads(out)
        iid = data.get("new_contract") or data.get("id")
        if iid:
            return int(iid)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: parse "new_contract: <id>"
    m = re.search(r"new_contract[:\s]+(\d+)", out)
    if m:
        return int(m.group(1))
    log.error(f"Could not parse instance ID from: {out}")
    return None

def instance_still_alive(instance_id: int) -> bool:
    """Check if an instance is still listed (not destroyed)."""
    try:
        raw = _vastai("show", "instances", "--raw")
        instances = json.loads(raw)
        return any(int(i.get("id", 0)) == instance_id for i in instances)
    except Exception as e:
        log.warning(f"Could not check instances: {e}")
        return True  # assume alive on error

def destroy_instance(instance_id: int):
    try:
        out = _vastai("destroy", "instance", str(instance_id))
        log.info(f"Destroyed instance {instance_id}: {out}")
    except Exception as e:
        log.error(f"Failed to destroy instance {instance_id}: {e}")

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global _active_instance_id
    log.info(f"Scaler started — checking every {CHECK_SECS}s, max ${MAX_PRICE}/hr, min {MIN_DL} MB/s dl")

    while True:
        try:
            pending = count_pending()
            log.info(f"Pending/active jobs: {pending}, tracked instance: {_active_instance_id}")

            # If we have a tracked instance but it died, clear it
            if _active_instance_id and not instance_still_alive(_active_instance_id):
                log.warning(f"Instance {_active_instance_id} no longer alive, clearing")
                _active_instance_id = None

            # Scale up: jobs waiting but no worker
            if pending > 0 and _active_instance_id is None:
                offer_id = find_rtx4090_offer()
                if offer_id:
                    iid = create_instance(offer_id)
                    if iid:
                        _active_instance_id = iid
                        log.info(f"Created instance {iid} for {pending} pending job(s)")
                    else:
                        log.error("Instance creation returned no ID")
                else:
                    log.warning("No offers available — will retry")

        except Exception as e:
            log.error(f"Scaler loop error: {e}")

        time.sleep(CHECK_SECS)

if __name__ == "__main__":
    main()
