"""RunPod Serverless entrypoint.

Boot-stage logging so failures during import / startup show up in RunPod logs
instead of the worker silently crash-looping without ever picking a job.
"""
import os
import sys
import time
import traceback

_t0 = time.time()
print(f"[BOOT 0] handler.py start  python={sys.version.split()[0]}  "
      f"cuda_vis={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
      f"hf_token_set={'HF_TOKEN' in os.environ}", flush=True)

# probe: can we see GPU at all?
try:
    import torch
    print(f"[BOOT 1] torch={torch.__version__}  "
          f"cuda_available={torch.cuda.is_available()}  "
          f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
          flush=True)
except Exception as e:
    print(f"[BOOT 1] torch import FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc(file=sys.stderr)
    raise

# probe: runpod lib
try:
    import runpod
    print(f"[BOOT 2] runpod loaded  version={getattr(runpod, '__version__', '?')}", flush=True)
except Exception as e:
    print(f"[BOOT 2] runpod import FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc(file=sys.stderr)
    raise

# probe: our pipeline module (heaviest import)
try:
    from pipeline import process_session
    print(f"[BOOT 3] pipeline imported OK  t={time.time()-_t0:.1f}s", flush=True)
except Exception as e:
    print(f"[BOOT 3] pipeline import FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc(file=sys.stderr)
    raise

print(f"[BOOT 4] handler ready  total_boot={time.time()-_t0:.1f}s", flush=True)


def handler(event):
    t0 = time.time()
    try:
        inp = event.get("input") or {}
        session_id = int(inp["session_id"])
        chunks = inp.get("chunks") or []
        enrolled = inp.get("enrolled_speakers") or {}
        print(f"[JOB {session_id}] received  chunks={len(chunks)}  enrolled={list(enrolled)}", flush=True)
        if not chunks:
            return {"error": "no chunks provided", "session_id": session_id}
        result = process_session(session_id, chunks, enrolled)
        print(f"[JOB {session_id}] done  elapsed={time.time()-t0:.1f}s", flush=True)
        return result
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"[JOB] ERROR: {type(e).__name__}: {e}", flush=True)
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    print("[BOOT 5] starting runpod serverless loop", flush=True)
    runpod.serverless.start({"handler": handler})
