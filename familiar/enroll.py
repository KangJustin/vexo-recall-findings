"""The wearer's voice baseline: enrol ("Teach it your voice", 3 read-aloud samples) and test it.

Nothing is trained: a pretrained speaker model (SpeechBrain ECAPA, familiar/voices.py) turns speech into a
voiceprint. Enrolment averages the voiceprints of three read-aloud samples into the wearer's baseline and
picks a starting threshold from how consistent those samples are. The voice test then measures, on new speech
from the wearer and from other people, how well "you" and "not you" separate, and which threshold is best.

The baseline is the wearer's own voiceprint (192 numbers, not audio). It is saved only on this laptop
(profiles/me.json, git-ignored) and removed with `python3 run.py --forget-voice`.
"""
import datetime
import json
import os

import numpy as np

SR = 16000
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE = os.path.join(APP_DIR, "profiles", "me.json")

PROMPTS = [
    "I meet a lot of new people, and I'd like to remember every one of them.",
    "Yesterday I walked to the coffee shop, ordered a large black coffee, and read the news.",
    "My favourite projects mix hardware and software, like robots, sensors, and small gadgets.",
]

FRAME = 0.03            # seconds per frame for the speech detector
MIN_SPEECH = 2.0        # seconds of speech a sample needs
MIN_LEVEL_DB = -45.0    # quieter than this is not speech near the band
CLIP_SHARE = 0.01       # more than 1% of samples at full scale: too loud, distorted
WINDOWS = (1.0, 2.0, 3.0)


# ---------------------------------------------------------------- audio helpers

def frames_db(audio, frame=FRAME):
    n = int(frame * SR)
    count = len(audio) // n
    if count == 0:
        return np.zeros(0)
    x = audio[:count * n].reshape(count, n).astype(np.float64)
    return 20 * np.log10(np.maximum(np.sqrt((x ** 2).mean(axis=1)), 1e-6))


def speech_only(audio, frame=FRAME, hang=0.2, margin=12.0):
    """The audio with silence and quiet background removed (simple energy detector with a short hang-over).
    margin: dB above the noise floor that counts as speech (12 for enrolment; 6 keeps quieter, farther voices).
    Returns (speech samples, seconds of speech, noise floor dB, speech level dB)."""
    db = frames_db(audio, frame)
    if not len(db):
        return audio[:0], 0.0, -120.0, -120.0
    floor = float(np.percentile(db, 10))
    gate = max(floor + margin, MIN_LEVEL_DB - (12.0 - margin))
    on = db >= gate
    keep = on.copy()
    hold = int(hang / frame)
    for i in np.flatnonzero(on):                       # keep a little after each loud frame (word endings)
        keep[i:i + hold + 1] = True
    n = int(frame * SR)
    mask = np.repeat(keep, n)
    speech = audio[:len(mask)][mask]
    level = float(np.median(db[on])) if on.any() else -120.0
    return speech, len(speech) / SR, floor, level


def check_sample(audio):
    """(speech, problem or None) for one enrolment sample."""
    speech, seconds, floor, level = speech_only(audio)
    if np.mean(np.abs(audio) >= 0.99) > CLIP_SHARE:
        return speech, "too loud (the sound is distorted): hold the band a little further away"
    if seconds < MIN_SPEECH:
        if floor > MIN_LEVEL_DB + 10:                    # loud all the time: speech cannot stand out
            return speech, "too much background noise (%.0f dB even between words): find a quieter spot" % floor
        return speech, "only %.1f s of speech heard: read the whole sentence, close to the band" % seconds
    if level < MIN_LEVEL_DB + 5:
        return speech, "very quiet (%.0f dB): speak up or hold the band closer" % level
    if level - floor < 15:
        return speech, "too much background noise (speech only %.0f dB above it): find a quieter spot" % (level - floor)
    return speech, None


def windows(speech, seconds, hop=None):
    """Pieces of `seconds` length from speech-only audio (half-overlapping)."""
    n, step = int(seconds * SR), int((hop or seconds / 2) * SR)
    return [speech[i:i + n] for i in range(0, len(speech) - n + 1, step)]


# ---------------------------------------------------------------- enrolment

def build_profile(samples, encode):
    """samples: speech-only audio of each read-aloud sample. Returns the baseline profile dict."""
    vectors = [encode(s) for s in samples]
    centroid = np.mean(vectors, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    pairs = [float(vectors[i] @ vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    # how 2 s pieces of each sample score against the OTHER samples' average: what "you" looks like on short speech
    held_out = []
    for i, s in enumerate(samples):
        rest = np.mean([v for j, v in enumerate(vectors) if j != i], axis=0)
        rest = rest / np.linalg.norm(rest)
        held_out += [float(encode(w) @ rest) for w in windows(s, 2.0)]
    low = float(np.percentile(held_out, 5)) if held_out else min(pairs)
    loud = [float(np.median(frames_db(s))) for s in samples]
    return {
        "version": 1, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": "speechbrain/spkrec-ecapa-voxceleb", "embedding": [round(float(x), 6) for x in centroid],
        "samples": len(samples), "seconds": round(sum(len(s) for s in samples) / SR, 1),
        "consistency": round(min(pairs), 3) if pairs else None,
        "held_out_2s": {"p5": round(low, 3), "median": round(float(np.median(held_out)), 3) if held_out else None},
        # a starting point only: the voice test measures the real one
        "threshold": round(float(np.clip(low - 0.10, 0.30, 0.60)), 3),
        "loudness_db": round(float(np.median(loud)), 1),
    }


def save_profile(profile, path=PROFILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=1)
    return path


def load_profile(path=PROFILE):
    try:
        with open(path) as f:
            profile = json.load(f)
        vector = np.asarray(profile["embedding"], dtype=np.float32)
        profile["vector"] = vector / np.linalg.norm(vector)
        return profile
    except (OSError, ValueError, KeyError):
        return None


def forget_profile(path=PROFILE):
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------------------------------------------------------- the voice test

def score_windows(speech, profile, encode, seconds):
    return [float(encode(w) @ profile["vector"]) for w in windows(speech, seconds)]


def separation(mine, others):
    """How well scores separate: the best threshold, accuracy there, and the equal error rate."""
    if not mine or not others:
        return None
    mine, others = np.asarray(mine), np.asarray(others)
    best = None
    for t in np.unique(np.concatenate([mine, others])):
        miss = float(np.mean(mine < t))                 # you, but called someone else
        false = float(np.mean(others >= t))             # someone else, called you
        acc = 1 - (miss * len(mine) + false * len(others)) / (len(mine) + len(others))
        if best is None or acc > best["accuracy"]:
            best = {"threshold": round(float(t), 3), "accuracy": round(acc, 3), "miss": round(miss, 3),
                    "false_accept": round(false, 3)}
    eer = min(((abs(np.mean(mine < t) - np.mean(others >= t)), (np.mean(mine < t) + np.mean(others >= t)) / 2)
               for t in np.linspace(-1, 1, 401)), key=lambda x: x[0])[1]
    return dict(best, eer=round(float(eer), 3),
                you={"n": len(mine), "median": round(float(np.median(mine)), 3), "min": round(float(mine.min()), 3)},
                others={"n": len(others), "median": round(float(np.median(others)), 3),
                        "max": round(float(others.max()), 3)})


def at_threshold(mine, others, t):
    if not mine or not others:
        return None
    miss, false = float(np.mean(np.asarray(mine) < t)), float(np.mean(np.asarray(others) >= t))
    return {"threshold": t, "miss": round(miss, 3), "false_accept": round(false, 3),
            "accuracy": round(1 - (miss * len(mine) + false * len(others)) / (len(mine) + len(others)), 3)}


def report(you_speech, other_speech, profile, encode):
    """The voice test's numbers for 1, 2 and 3 s windows, plus loudness."""
    out = {"profile_threshold": profile["threshold"], "windows": {}}
    for w in WINDOWS:
        mine = score_windows(you_speech, profile, encode, w)
        others = score_windows(other_speech, profile, encode, w)
        out["windows"]["%gs" % w] = {"separation": separation(mine, others),
                                     "at_profile_threshold": at_threshold(mine, others, profile["threshold"])}
    you_db = float(np.median(frames_db(you_speech))) if len(you_speech) else None
    other_db = float(np.median(frames_db(other_speech))) if len(other_speech) else None
    out["loudness_db"] = {"you": round(you_db, 1) if you_db is not None else None,
                          "others": round(other_db, 1) if other_db is not None else None}
    return out
