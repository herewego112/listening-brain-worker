# Listening Brain — GPU Worker

RunPod Serverless worker for the Listening Brain pipeline. One image, 7 stages:

```
chunks -> VAD trim -> DeepFilterNet -> Whisper large-v3
                                    `-> pyannote 3.1 diarize -> ECAPA speaker-ID
                                        -> segments with speaker labels
```

All model weights are baked into the image at build time so cold starts don't pay the download cost.

## What's in here

| file | purpose |
|---|---|
| `Dockerfile` | builds the image |
| `requirements.txt` | pinned Python deps |
| `download_models.py` | pulls Whisper, pyannote, ECAPA, DFN3, silero-vad during build |
| `pipeline.py` | the 7-stage audio processing |
| `handler.py` | RunPod serverless entry (`runpod.serverless.start(...)`) |

## Pre-flight (one-time)

### 1. HuggingFace token for pyannote
pyannote's diarization model is "gated" — you must accept terms once on Hugging Face:
- go to <https://huggingface.co/pyannote/speaker-diarization-3.1> → Accept
- also <https://huggingface.co/pyannote/segmentation-3.0> → Accept
- create a read token at <https://huggingface.co/settings/tokens>

Save it locally as `HF_TOKEN`.

### 2. Docker registry
RunPod pulls from a public registry. Simplest: **Docker Hub**.
Create an account at <https://hub.docker.com> and a repo called `listening-brain-worker`.

## Build

```bash
cd C:/Users/inthe/listening-brain/gpu-worker

# replace these two
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
export DOCKER_USER=yourdockeruser

# build (this takes ~15-25 min first time — downloads + bakes all weights)
docker build --build-arg HF_TOKEN=$HF_TOKEN \
    -t $DOCKER_USER/listening-brain-worker:latest .

# login + push
docker login
docker push $DOCKER_USER/listening-brain-worker:latest
```

Image size will be **~10-12 GB** (model weights). That's normal.

## Deploy on RunPod Serverless

1. Open <https://www.runpod.io/console/serverless>
2. **New Endpoint**
3. Name: `listening-brain`
4. Select GPU: **RTX 4090** (or RTX 6000 Ada as fallback)
5. Workers: min=0, max=1 (so we only pay while active)
6. Container image: `docker.io/DOCKER_USER/listening-brain-worker:latest`
7. Container disk: 20 GB
8. Idle timeout: 5 seconds (releases GPU immediately after each run)
9. Env vars (all required for the worker):
   - `HF_TOKEN=hf_xxxxxxxx`
10. Save. RunPod gives you an **Endpoint ID** like `abc123xyz`.

## Wire it into the PC dispatcher

Save these env vars (cmd.exe shown; powershell uses `$env:`):

```cmd
setx RUNPOD_API_KEY       rpa_xxxxxxxx
setx RUNPOD_ENDPOINT_ID   abc123xyz
setx BRAIN_PUBLIC_URL     https://your-cloudflared-url
```

Then to dispatch manually:

```bash
python C:/Users/inthe/listening-brain/scripts/dispatcher.py
python C:/Users/inthe/listening-brain/scripts/router.py
```

## Local smoke test (no RunPod)

Before pushing to RunPod, you can verify the handler runs on your PC (if you have a CUDA GPU):

```bash
docker run --rm --gpus all \
    -e HF_TOKEN=$HF_TOKEN \
    -e RUNPOD_REALTIME_STARTED=1 \
    -p 8000:8000 \
    $DOCKER_USER/listening-brain-worker:latest
```

Then POST `{"input": {"session_id": 0, "chunks": [...]}}` to `http://localhost:8000/run`.

## Cost expectations

RunPod 4090 Serverless: **$1.10/hr**. Typical 10-min-of-audio run: ~5-10 GPU-min => **$0.09-0.18/run**.

For 24 hourly runs/day: **~$2.50-4.50/day** (scales with how much of each hour was actually speech).

The watchdog (`scripts/watchdog.py`) kills any pod alive > 30 min and trips a pause flag if daily spend exceeds `BRAIN_COST_CAP_USD` (default $5).
