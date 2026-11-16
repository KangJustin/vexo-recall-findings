"""Check the wearer baseline (familiar/enroll.py) on macOS text-to-speech voices, with no band.

    python3 eval/voice_baseline.py      -> prints the report, writes eval/results/voice_baseline_synthetic.json

"Samantha" enrols with the three prompts, then new Samantha speech is scored against four other voices that
are quieter (across the table), in a quiet and a noisy room. Synthetic voices are far more distinct than
real people: the real test is `python3 run.py <band> --voice-test`.
"""
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from familiar import enroll  # noqa: E402
from familiar.voices import load_encoder  # noqa: E402

TALK = ["So what are you working on for the demo this weekend?",
        "I think the camera should go above the arm, not on the side.",
        "We tried three different grippers and the first two kept slipping.",
        "Honestly the hardest part was getting the firmware to talk to the app.",
        "Let's grab lunch after this and look at the wiring together.",
        "Do you know if the workshop downstairs has a soldering station?"]


def say(voice, text, gain=1.0):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.wav")
        subprocess.run(["say", "-v", voice, "-o", path, "--data-format=LEI16@16000", text], check=True)
        with wave.open(path) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return x * gain


def noisy(x, db):
    """Add pink-ish room noise at `db` dBFS."""
    rng = np.random.default_rng(1)
    white = rng.standard_normal(len(x) + 64)
    pink = np.convolve(white, np.ones(64) / 64, mode="valid")[:len(x)]
    pink = pink / (np.sqrt(np.mean(pink ** 2)) + 1e-9) * 10 ** (db / 20)
    return np.clip(x + pink.astype(np.float32), -1, 1)


def main():
    encode = load_encoder()
    samples = []
    for prompt in enroll.PROMPTS:
        speech, problem = enroll.check_sample(say("Samantha", prompt))
        assert problem is None, problem
        samples.append(speech)
    profile = enroll.build_profile(samples, encode)
    profile["vector"] = np.asarray(profile["embedding"], dtype=np.float32)
    print("enrolled: %.1f s, consistency %.2f, held-out 2 s p5 %.2f -> starting threshold %.2f" % (
        profile["seconds"], profile["consistency"], profile["held_out_2s"]["p5"], profile["threshold"]))
    results = {}
    for room, noise_db in [("quiet", None), ("noisy (-35 dBFS room noise)", -35)]:
        def prep(x):
            return enroll.speech_only(noisy(x, noise_db) if noise_db else x, margin=6.0)[0]
        you = np.concatenate([prep(say("Samantha", t)) for t in TALK])
        others = np.concatenate([prep(say(v, t, gain=0.3)) for v in ["Daniel", "Karen", "Fred", "Moira"] for t in TALK[:3]])
        results[room] = enroll.report(you, others, profile, encode)
        print("\n== %s: you %.0f s, others %.0f s, loudness you %s dB / others %s dB" % (
            room, len(you) / 16000, len(others) / 16000, results[room]["loudness_db"]["you"], results[room]["loudness_db"]["others"]))
        for w, r in results[room]["windows"].items():
            s, a = r["separation"], r["at_profile_threshold"]
            print("  %s windows: you median %.2f (min %.2f) | others median %.2f (max %.2f) | best threshold %.2f -> "
                  "accuracy %.1f%%, EER %.1f%% | at starting threshold %.2f: accuracy %.1f%%, missed you %.1f%%, "
                  "others taken for you %.1f%%" % (
                      w, s["you"]["median"], s["you"]["min"], s["others"]["median"], s["others"]["max"], s["threshold"],
                      100 * s["accuracy"], 100 * s["eer"], a["threshold"], 100 * a["accuracy"], 100 * a["miss"],
                      100 * a["false_accept"]))
    profile.pop("vector")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "voice_baseline_synthetic.json"), "w") as f:
        json.dump({"profile": {k: v for k, v in profile.items() if k != "embedding"}, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
