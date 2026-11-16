"""Wearer enrolment for the diart path: clean clips recorded on the Vexo band -> the wearer profile.

The SAME procedure as live matching: 16 kHz mono, segmentation-3.0 windows of the model's duration, and diart's
overlap-aware ECAPA embedding (gamma, beta, unit norm). A clip is refused if it is silent, clipped, drowned in
noise, or if the segmentation hears a second voice. The confirmed profile is written only here; live sessions
never change it.
"""
import datetime
import io
import json
import os
import sys
import wave

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from familiar.enroll import check_sample  # noqa: E402  (silence / clipping / noise checks, numpy only)

OVERLAP_SHARE = 0.03      # more than 3% of speech frames with two voices: not a clean clip
SECOND_VOICE = 0.15       # a second local speaker active this share of the speech: someone else talked


def decode(data, content_type=None):
    """WAV bytes (any rate, mono or stereo, 16-bit) or raw 16-bit mono PCM at 16 kHz -> float32 at 16 kHz."""
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data)) as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError("only 16-bit WAV is supported")
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)
    else:
        rate = 16000
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
    if rate != 16000:
        import torch
        import torchaudio.functional as AF
        x = AF.resample(torch.from_numpy(x), rate, 16000).numpy()
    return x.astype(np.float32)


def windows(x, n):
    """Model-duration windows covering x (the last one padded with silence)."""
    out = []
    for i in range(0, max(1, len(x)), n):
        w = x[i:i + n]
        if len(w) < n:
            w = np.concatenate([w, np.zeros(n - len(w), np.float32)])
        out.append(w)
    return out


def analyse_clip(audio, models):
    """{"ok", "problem", "seconds", "embedding"} for one clip, with the live pipeline's own models and settings."""
    import torch
    from diart.blocks import OverlapAwareSpeakerEmbedding, SpeakerSegmentation
    speech, problem = check_sample(audio)
    info = {"ok": False, "problem": problem, "seconds": round(len(speech) / 16000, 2)}
    if problem:
        return info
    cfg = models.config
    n = int(round(models.duration * cfg.sample_rate))
    # exactly the blocks the live pipeline uses (diart SpeakerDiarization builds these from the same config)
    segment = SpeakerSegmentation(cfg.segmentation, cfg.device)
    embed = OverlapAwareSpeakerEmbedding(cfg.embedding, cfg.gamma, cfg.beta, norm=1,
                                         normalize_weights=cfg.normalize_embedding_weights, device=cfg.device)
    vectors, overlap_frames, speech_frames, second = [], 0, 0, 0.0
    from speaker_service.engine import INFERENCE_LOCK
    for w in windows(audio, n):
        batch = torch.from_numpy(w).reshape(1, -1, 1)                           # (batch, samples, channels)
        with INFERENCE_LOCK, torch.no_grad():
            seg = segment(batch)                                                # (batch, frames, local speakers)
            emb = embed(batch, seg)                                             # (batch, local speakers, dim)
        act = seg[0].cpu().numpy() > 0.5                                        # (frames, local speakers)
        talking = act.any(axis=1)
        speech_frames += int(talking.sum())
        overlap_frames += int((act.sum(axis=1) > 1).sum())
        per = act.sum(axis=0)
        if talking.sum():
            ranked = np.sort(per)[::-1]
            second = max(second, float(ranked[1] / talking.sum()) if len(ranked) > 1 else 0.0)
            main = int(np.argmax(per))
            v = emb[0, main].cpu().numpy()
            if np.all(np.isfinite(v)):
                vectors.append((v / np.linalg.norm(v), float(per[main])))
    if speech_frames and overlap_frames / speech_frames > OVERLAP_SHARE:
        info["problem"] = "two voices at once in %.0f%% of the speech: record alone" % (100 * overlap_frames / speech_frames)
        return info
    if second > SECOND_VOICE:
        info["problem"] = "another voice is in the recording (%.0f%% of the speech): record alone" % (100 * second)
        return info
    if not vectors:
        info["problem"] = "no usable speech found by the segmentation model"
        return info
    weights = np.array([w for _, w in vectors])
    centroid = np.sum([v * w for (v, _), w in zip(vectors, weights)], axis=0)
    info.update(ok=True, embedding=(centroid / np.linalg.norm(centroid)).tolist())
    return info


def build_profile(clips, models, settings):
    """clips: analyse_clip results that are ok. The confirmed wearer profile."""
    vectors = np.array([c["embedding"] for c in clips])
    centroid = vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    pairs = [float(vectors[i] @ vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return {
        "kind": "vexo-wearer-diart", "version": 1, "confirmed": True,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "segmentation_model": settings.segmentation, "embedding_model": settings.embedding,
        "preprocessing": {"sample_rate": 16000, "channels": 1, "window_seconds": models.duration,
                          "gamma": settings.gamma, "beta": settings.beta, "norm": 1,
                          "embedding": "diart OverlapAwareSpeakerEmbedding, dominant local speaker per window"},
        "clips": len(clips), "seconds": round(sum(c["seconds"] for c in clips), 1),
        "consistency": round(min(pairs), 3) if pairs else None,
        "embedding": [round(float(x), 6) for x in centroid],
    }


def save(profile, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=1)


def load(path, settings=None):
    try:
        with open(path) as f:
            profile = json.load(f)
    except (OSError, ValueError):
        return None
    if settings is not None and profile.get("embedding_model") != settings.embedding:
        return None                                       # made with another model: not comparable
    return profile
