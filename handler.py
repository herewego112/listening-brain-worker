"""RunPod Serverless entrypoint.

Expected input shape (what the PC dispatcher sends):
{
  "input": {
    "session_id": 42,
    "chunks": [
      {"chunk_id": 1, "device_id": "huawei-01", "user_id": "C",
       "start_utc": "2026-04-18T21:44:32+00:00",
       "audio_url": "https://.../foo.opus?t=xxx"}
    ],
    "enrolled_speakers": {"C": "https://.../C.npy?t=xxx"}
  }
}

Output shape:
{
  "session_id": 42,
  "segments": [ {...} ],
  "unknown_speakers": ["unknown_1", ...],
  "processing_ms": 12400
}
"""
import sys
import traceback

import runpod

from pipeline import process_session


def handler(event):
    try:
        inp = event.get("input") or {}
        session_id = int(inp["session_id"])
        chunks = inp.get("chunks") or []
        enrolled = inp.get("enrolled_speakers") or {}
        if not chunks:
            return {"error": "no chunks provided", "session_id": session_id}
        return process_session(session_id, chunks, enrolled)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
