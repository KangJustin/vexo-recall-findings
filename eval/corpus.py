"""Labelled test conversations for speaker attribution, made with macOS text-to-speech voices.

Each scenario is 16 kHz mono audio plus ground truth: who speaks over which interval, and where two
voices genuinely overlap. Synthetic voices are cleaner and more distinct than real people on a wrist
mic, so results here show how the PIPELINE handles each case, not real-world recognition accuracy;
for that, record and label real conversations (see eval/README.md).

    python3 eval/corpus.py            # writes eval/corpus/<scenario>.wav and .json
"""
import json
import os
import subprocess
import tempfile
import wave

import numpy as np

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus")

# speaker -> (macOS voice, gain). A is the wearer: on the wrist, so much louder than the others.
VOICES = {"A": ("Samantha", 1.0), "B": ("Daniel", 0.3), "C": ("Karen", 0.3)}

# Each step: (speaker, text, gap_before_seconds). A negative gap starts the line before the previous one
# ends: genuine overlapping speech. ("insert", ...) puts a line INSIDE the previous line, at a fraction of it.
SCENARIOS = {
    "alternating": [
        ("A", "Hey, welcome to the hardware meetup. Are you here for the robot demos?", 0.0),
        ("B", "Yes, it is my first time here. I work on gripper enclosures.", 0.7),
        ("A", "Oh nice, we have been struggling with gripper housings for months.", 0.7),
        ("B", "The trick is a softer plastic on the inside and a stiff shell outside.", 0.7),
    ],
    "no_silence": [
        ("A", "So what are you building for the demo this weekend", 0.0),
        ("B", "a small arm that sorts screws by size using a camera", 0.0),
        ("A", "that sounds really useful for the lab downstairs", 0.0),
        ("B", "thanks, we hope so, it still drops the tiny ones", 0.0),
    ],
    "short_interruption": [
        ("B", "The enclosure is printed in two halves and then we glue the seam and sand it down for hours", 0.0),
        ("insert", "A", "wait", 0.45),
        ("A", "Actually, which glue do you use for that seam?", 0.5),
        ("B", "Just a regular epoxy, nothing special.", 0.6),
    ],
    "backchannel": [
        ("B", "We tried three different grippers before this one, and the first two kept slipping off the parts", 0.0),
        ("insert", "A", "yeah", 0.35),
        ("insert", "A", "right", 0.75),
        ("A", "That happened to us too with the cheap ones.", 0.5),
    ],
    "simultaneous": [
        ("A", "I think the best place to put the camera is right above the arm", 0.0),
        ("B", "no no, the camera has to sit on the side or the arm blocks it", -1.6),
        ("A", "Okay, the side it is then.", 0.6),
    ],
    "three_speakers": [
        ("A", "Have you two met before?", 0.0),
        ("B", "No, I don't think so. I'm on the vision team.", 0.3),
        ("C", "Nice to meet you, I do the motor control.", 0.2),
        ("B", "Oh, then we should talk.", 0.15),
        ("A", "Great, I will leave you to it.", 0.3),
    ],
}


def speak(voice, text):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.wav")
        subprocess.run(["say", "-v", voice, "-o", path, "--data-format=LEI16@16000", text], check=True)
        with wave.open(path) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    nz = np.flatnonzero(np.abs(x) > 0.003)                     # trim the voice's own silence at both ends
    return x[nz[0]:nz[-1] + 1] if len(nz) else x


def build(steps):
    """(audio, truth) for one scenario. truth: {"turns": [...], "overlaps": [[start, end], ...]}."""
    track = np.zeros(0, dtype=np.float32)
    turns, overlaps = [], []
    end = 0.0
    for step in steps:
        if step[0] == "insert":                                  # a short line inside the previous turn
            _, who, text, at = step
            prev = turns[-1]
            voice, gain = VOICES[who]
            clip = speak(voice, text) * gain
            start = prev["start"] + at * (prev["end"] - prev["start"])
            a = int(start * SR)
            if a + len(clip) > len(track):
                track = np.concatenate([track, np.zeros(a + len(clip) - len(track), dtype=np.float32)])
            track[a:a + len(clip)] += clip
            turns.append({"speaker": who, "start": round(start, 3), "end": round(start + len(clip) / SR, 3),
                          "text": text})
            overlaps.append([round(start, 3), round(start + len(clip) / SR, 3)])
            continue
        who, text, gap = step
        voice, gain = VOICES[who]
        clip = speak(voice, text) * gain
        start = max(0.0, end + gap)
        a = int(start * SR)
        if a + len(clip) > len(track):
            track = np.concatenate([track, np.zeros(a + len(clip) - len(track), dtype=np.float32)])
        track[a:a + len(clip)] += clip
        if gap < 0 and turns:
            overlaps.append([round(start, 3), round(min(end, start + len(clip) / SR), 3)])
        turns.append({"speaker": who, "start": round(start, 3), "end": round(start + len(clip) / SR, 3), "text": text})
        end = max(end, start + len(clip) / SR)
    track = np.concatenate([np.zeros(int(0.5 * SR), np.float32), track, np.zeros(int(1.5 * SR), np.float32)])
    for t in turns:                                             # shift for the 0.5 s of lead-in silence
        t["start"], t["end"] = round(t["start"] + 0.5, 3), round(t["end"] + 0.5, 3)
    overlaps = [[round(a + 0.5, 3), round(b + 0.5, 3)] for a, b in overlaps]
    return np.clip(track, -1, 1), {"turns": sorted(turns, key=lambda t: t["start"]), "overlaps": overlaps}


def names():
    """The synthetic scenarios, plus any labelled real recording placed in eval/corpus/ (<name>.wav + <name>.json)."""
    extra = sorted(f[:-5] for f in os.listdir(OUT) if f.endswith(".json") and f[:-5] not in SCENARIOS) \
        if os.path.isdir(OUT) else []
    return list(SCENARIOS) + [n for n in extra if os.path.exists(os.path.join(OUT, n + ".wav"))]


def messages_path(name):
    """Deepgram live messages for a scenario: recorded by eval/deepgram_live.py, or saved by run.py --record
    next to a real recording in eval/corpus/."""
    own = os.path.join(OUT, name + ".jsonl")
    return own if os.path.exists(own) else os.path.join(HERE, "results", "deepgram_live", name + ".jsonl")


def load(name):
    with wave.open(os.path.join(OUT, name + ".wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    with open(os.path.join(OUT, name + ".json")) as f:
        return audio, json.load(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, steps in SCENARIOS.items():
        audio, truth = build(steps)
        with wave.open(os.path.join(OUT, name + ".wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((audio * 32767).astype("<i2").tobytes())
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(dict(truth, scenario=name, seconds=round(len(audio) / SR, 2)), f, indent=1)
        print("%-18s %5.1f s  %d turns  %d overlap regions" % (name, len(audio) / SR, len(truth["turns"]),
                                                              len(truth["overlaps"])))


if __name__ == "__main__":
    main()
