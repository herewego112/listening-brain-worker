"""Pre-download all model weights into the image so cold starts don't pay for it.

HF_TOKEN must be set at build time for pyannote (gated model, user accepts terms).
"""
import os
import sys
from pathlib import Path

# Patch torch.load to default weights_only=False — pyannote's internal
# Model.from_pretrained loads checkpoints that include TorchVersion objects which
# are not in the default safe-globals allow-list of torch 2.6+.
import torch as _torch
_orig_torch_load = _torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
_torch.load = _patched_torch_load


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main() -> None:
    # 1) silero-vad — via torch.hub, cached under ~/.cache/torch/hub
    import torch
    log("[models] silero-vad")
    torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

    # 2) DeepFilterNet3 — pulls from HF; cached under ~/.cache/huggingface
    log("[models] DeepFilterNet3")
    from df.enhance import init_df
    init_df()  # downloads if not cached

    # 3) faster-whisper large-v3
    log("[models] faster-whisper large-v3")
    from faster_whisper import WhisperModel
    WhisperModel("large-v3", device="cpu", download_root="/models/whisper")

    # 4) pyannote 3.1 diarization
    log("[models] pyannote 3.1 diarization")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log("  !! HF_TOKEN missing — pyannote download will fail")
        log("  set it at build time: docker build --build-arg HF_TOKEN=... .")
        sys.exit(1)
    from pyannote.audio import Pipeline
    Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # 5) ECAPA-TDNN (SpeechBrain)
    log("[models] ECAPA-TDNN speaker embedding")
    from speechbrain.inference.speaker import EncoderClassifier
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/models/ecapa",
        run_opts={"device": "cpu"},
    )

    log("[models] all downloaded")


if __name__ == "__main__":
    main()
