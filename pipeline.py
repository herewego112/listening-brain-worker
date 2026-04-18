"""The 7-stage audio processing pipeline. All GPU where possible.

Input:  list of chunk dicts (audio_url, start_utc, device_id, user_id, chunk_id)
        + enrolled_speakers dict (name -> embedding .npy URL)
Output: list of segments (start_offset_s, end_offset_s, speaker, text, confidence)
        + list of unknown speaker labels that appeared
"""
from __future__ import annotations
import os
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch

TARGET_SR = 16000
UNKNOWN_SIM_THRESHOLD = 0.55       # below this cosine sim → unknown speaker
KNOWN_SIM_THRESHOLD   = 0.70       # above this → confident match


@dataclass
class Segment:
    start_offset_s: float
    end_offset_s:   float
    speaker_id:     str
    speaker_name:   str
    speaker_confidence: float
    text:           str
    whisper_confidence: float
    source_device_id:   str


# ── lazy singletons ─────────────────────────────────────────────────────────

_vad = None
_df_state = None
_whisper = None
_diarizer = None
_spk_encoder = None


def vad():
    global _vad
    if _vad is None:
        m, _ = torch.hub.load("snakers4/silero-vad", "silero_vad",
                              trust_repo=True, verbose=False)
        _vad = m.to("cuda" if torch.cuda.is_available() else "cpu")
    return _vad


def df_state():
    global _df_state
    if _df_state is None:
        from df.enhance import init_df
        model, df_state_obj, _ = init_df()
        _df_state = (model, df_state_obj)
    return _df_state


def whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _whisper = WhisperModel(
            "large-v3", device=device, compute_type=compute,
            download_root="/models/whisper",
        )
    return _whisper


def diarizer():
    global _diarizer
    if _diarizer is None:
        from pyannote.audio import Pipeline
        _diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN"),
        )
        if torch.cuda.is_available():
            _diarizer.to(torch.device("cuda"))
    return _diarizer


def spk_encoder():
    global _spk_encoder
    if _spk_encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/models/ecapa",
            run_opts={"device": device},
        )
    return _spk_encoder


# ── stages ──────────────────────────────────────────────────────────────────

def download(url: str, dest: Path) -> None:
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for ch in r.iter_content(1 << 16):
            f.write(ch)


def load_audio(path: Path) -> np.ndarray:
    """Load any format to 16kHz mono float32 via librosa."""
    import librosa
    y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return y.astype(np.float32)


def vad_trim(audio: np.ndarray) -> np.ndarray:
    """Drop leading/trailing silence. Doesn't split mid-speech."""
    from silero_vad import get_speech_timestamps
    model = vad()
    tensor = torch.from_numpy(audio)
    ts = get_speech_timestamps(tensor, model, sampling_rate=TARGET_SR,
                               min_speech_duration_ms=250,
                               min_silence_duration_ms=500)
    if not ts:
        return audio[:0]
    start = max(0, ts[0]["start"] - int(0.1 * TARGET_SR))
    end   = min(len(audio), ts[-1]["end"] + int(0.1 * TARGET_SR))
    return audio[start:end]


def denoise(audio: np.ndarray, strong: bool = False) -> np.ndarray:
    """DeepFilterNet3 denoise. strong=True applies 2 passes."""
    from df.enhance import enhance
    model, state = df_state()
    tensor = torch.from_numpy(audio).unsqueeze(0)
    out = enhance(model, state, tensor)
    if strong:
        out = enhance(model, state, out)
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def concat_chunks_by_time(chunks: list[dict], audio_by_id: dict[int, np.ndarray]
                          ) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """Concatenate device chunks in chronological order.

    Returns (merged_audio, [(start_offset_s, end_offset_s, chunk_id), ...])
    For now, single-device: straight concat. Multi-device merge is future work.
    """
    # sort by start_utc
    from datetime import datetime
    chunks_sorted = sorted(chunks, key=lambda c: datetime.fromisoformat(c["start_utc"]))
    parts = []
    spans = []
    offset = 0.0
    for c in chunks_sorted:
        a = audio_by_id.get(c["chunk_id"])
        if a is None or len(a) == 0:
            continue
        parts.append(a)
        dur = len(a) / TARGET_SR
        spans.append((offset, offset + dur, c["chunk_id"]))
        offset += dur
    merged = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return merged, spans


def transcribe(audio: np.ndarray) -> list[dict]:
    """faster-whisper large-v3 → list of {start, end, text, avg_logprob}."""
    model = whisper()
    segments, _ = model.transcribe(
        audio, language=None,
        vad_filter=False,        # we already trimmed
        word_timestamps=False,
        condition_on_previous_text=True,
    )
    out = []
    for s in segments:
        out.append({
            "start": float(s.start),
            "end":   float(s.end),
            "text":  s.text.strip(),
            "avg_logprob": float(s.avg_logprob) if s.avg_logprob is not None else -1.0,
        })
    return out


def diarize(audio: np.ndarray) -> list[dict]:
    """pyannote 3.1 → list of {start, end, speaker}."""
    pipe = diarizer()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, TARGET_SR)
        tmp_path = f.name
    try:
        diar = pipe(tmp_path)
    finally:
        os.unlink(tmp_path)
    segs = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segs.append({"start": float(turn.start),
                     "end":   float(turn.end),
                     "speaker": speaker})
    return segs


def embed_speaker(audio: np.ndarray) -> np.ndarray:
    """ECAPA-TDNN → 192-dim embedding (L2-normalized)."""
    enc = spk_encoder()
    t = torch.from_numpy(audio).unsqueeze(0)
    emb = enc.encode_batch(t).squeeze().detach().cpu().numpy()
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def load_enrolled(enrolled: dict[str, str]) -> dict[str, np.ndarray]:
    """Download .npy embedding files, return name -> L2-normalized vector."""
    out = {}
    for name, url in (enrolled or {}).items():
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            download(url, Path(f.name))
            tmp = f.name
        try:
            v = np.load(tmp)
        finally:
            os.unlink(tmp)
        n = np.linalg.norm(v)
        out[name] = v / n if n > 0 else v
    return out


def match_speaker(emb: np.ndarray, library: dict[str, np.ndarray]
                  ) -> tuple[str, float]:
    """Cosine-similarity against library. Returns (best_name, sim)."""
    if not library:
        return ("unknown", 0.0)
    best_name, best_sim = "unknown", -1.0
    for name, v in library.items():
        sim = float(np.dot(emb, v))
        if sim > best_sim:
            best_name, best_sim = name, sim
    return (best_name, best_sim)


def assign_speakers(segments_text: list[dict], segments_diar: list[dict],
                    audio: np.ndarray, library: dict[str, np.ndarray]
                    ) -> list[Segment]:
    """Merge Whisper text segments with diarizer speaker spans.

    For each text segment: find the diarizer speaker whose span overlaps most,
    slice audio for that speaker, get ECAPA embedding, match against library.
    """
    out: list[Segment] = []
    unknown_counter = 1
    anon_map: dict[str, str] = {}   # pyannote speaker label -> stable 'unknown_N'

    for ts in segments_text:
        # find best-overlap diarizer segment
        best = None
        best_ov = 0.0
        for ds in segments_diar:
            ov = max(0.0, min(ts["end"], ds["end"]) - max(ts["start"], ds["start"]))
            if ov > best_ov:
                best_ov = ov
                best = ds
        if best is None:
            continue
        # slice audio for this segment to get speaker embedding
        s_idx = int(max(0, ts["start"]) * TARGET_SR)
        e_idx = int(min(len(audio) / TARGET_SR, ts["end"]) * TARGET_SR)
        slice_ = audio[s_idx:e_idx]
        if len(slice_) < TARGET_SR // 2:  # < 0.5s, too short for embedding
            speaker_name = anon_map.setdefault(best["speaker"], f"unknown_{unknown_counter}")
            if speaker_name.startswith("unknown_") and best["speaker"] not in anon_map:
                unknown_counter += 1
            confidence = 0.0
            speaker_id = speaker_name
        else:
            emb = embed_speaker(slice_)
            name, sim = match_speaker(emb, library)
            if sim >= KNOWN_SIM_THRESHOLD:
                speaker_name = name
                speaker_id = name
                confidence = sim
            elif sim >= UNKNOWN_SIM_THRESHOLD:
                # low-confidence match → use anon label but keep the guess in id
                anon = anon_map.setdefault(best["speaker"], f"unknown_{unknown_counter}")
                if best["speaker"] not in anon_map:
                    unknown_counter += 1
                speaker_name = anon
                speaker_id = f"maybe_{name}"
                confidence = sim
            else:
                anon = anon_map.setdefault(best["speaker"], f"unknown_{unknown_counter}")
                if best["speaker"] not in anon_map:
                    unknown_counter += 1
                speaker_name = anon
                speaker_id = anon
                confidence = sim

        # convert avg_logprob to pseudo-confidence
        whisper_conf = float(np.exp(ts.get("avg_logprob", -1.0)))

        out.append(Segment(
            start_offset_s=ts["start"],
            end_offset_s=ts["end"],
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            speaker_confidence=confidence,
            text=ts["text"],
            whisper_confidence=whisper_conf,
            source_device_id="",  # single-device for now
        ))
    return out


# ── orchestrator ────────────────────────────────────────────────────────────

def process_session(session_id: int, chunks: list[dict],
                    enrolled: dict[str, str]) -> dict:
    t0 = time.time()

    # 1) download all chunks
    audio_by_id: dict[int, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for c in chunks:
            local = tmp_path / f"{c['chunk_id']}.opus"
            download(c["audio_url"], local)
            audio_by_id[c["chunk_id"]] = load_audio(local)

    # 2) per-chunk VAD trim (cheap, removes silence)
    audio_by_id = {k: vad_trim(v) for k, v in audio_by_id.items()}

    # 3) concat in chronological order, produce merged audio + span index
    merged, _spans = concat_chunks_by_time(chunks, audio_by_id)
    if len(merged) == 0:
        return {"session_id": session_id, "segments": [],
                "unknown_speakers": [], "processing_ms": int(1000*(time.time()-t0)),
                "note": "no audio after VAD"}

    # 4) denoise twice — mild for diarization, strong for transcription
    mild_clean   = denoise(merged, strong=False)
    strong_clean = denoise(merged, strong=True)

    # 5) transcribe (Whisper on strong-cleaned)
    text_segs = transcribe(strong_clean)

    # 6) diarize (pyannote on mild-cleaned)
    diar_segs = diarize(mild_clean)

    # 7) speaker identification — match against enrolled library
    library = load_enrolled(enrolled)
    final = assign_speakers(text_segs, diar_segs, mild_clean, library)

    unknown = sorted({s.speaker_name for s in final if s.speaker_name.startswith("unknown_")})

    return {
        "session_id": session_id,
        "segments": [asdict(s) for s in final],
        "unknown_speakers": unknown,
        "processing_ms": int(1000 * (time.time() - t0)),
    }
