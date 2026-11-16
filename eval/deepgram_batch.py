"""Deepgram's RECORDED-audio diarization (diarize_model=latest -> v2 for batch) on each scenario's audio.

    python3 eval/deepgram_batch.py   -> eval/results/deepgram_batch/<scenario>.json  (+ latency)

Uses familiar/second_pass.deepgram_batch, the same call the app's second pass makes.
"""
import io
import json
import os
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import corpus  # noqa: E402
from familiar import second_pass  # noqa: E402

OUT = os.path.join(HERE, "results", "deepgram_batch")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[1:] or corpus.names():
        audio, _ = corpus.load(name)
        started = time.monotonic()
        raw = second_pass.deepgram_batch(second_pass.wav_bytes(audio))
        took = time.monotonic() - started
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump({"seconds_taken": round(took, 3), "response": raw}, f)
        print("%-18s %.2f s" % (name, took))


if __name__ == "__main__":
    main()
