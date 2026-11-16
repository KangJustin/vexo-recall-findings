"""Band-side commands for the wearer's voice baseline (see familiar/enroll.py):

    python3 run.py 64 --enroll         "Teach it your voice": read 3 sentences, save the baseline
    python3 run.py 64 --voice-test     you talk, then others talk: how well does the baseline tell you apart?
    python3 run.py --forget-voice      delete the saved baseline

No sound goes anywhere: the band's audio is processed on this laptop, and only the voiceprint (192 numbers)
is saved. The voice test saves scores only (eval/results/voice_test_*.json), never audio.
"""
import datetime
import json
import os
import sys
import threading
import time

import numpy as np

from familiar import enroll

SR = enroll.SR


class BandMic:
    """The band's microphone, decoded on this laptop. record() collects between start() and stop()."""

    def __init__(self, number):
        from familiar import live  # noqa: F401  (puts the kit on the path)
        from vexo_open.audio import PcmDecoder
        from vexo_open.sync import Band
        self.decoder = PcmDecoder()
        self.band = Band.connect(number)
        self._pcm, self._on, self._lock = bytearray(), False, threading.Lock()
        self.level = -120.0
        self.listeners = []                     # called with every decoded chunk (the live voice tracker)
        self.band.on("audio", self._audio)
        self.band.configure(audio=True)

    def _audio(self, a):
        pcm = self.decoder.feed(a.opus)
        if not pcm:
            return
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self.level = 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-6))
        for listen in self.listeners:
            listen(x)
        with self._lock:
            if self._on:
                self._pcm += pcm

    def start(self):
        with self._lock:
            self._pcm, self._on = bytearray(), True

    def peek(self, seconds):
        with self._lock:
            pcm = bytes(self._pcm[-int(seconds * SR) * 2:])
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768

    def stop(self):
        with self._lock:
            self._on = False
            pcm = bytes(self._pcm)
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768

    def close(self):
        try:
            self.band.configure(audio=False)
        finally:
            self.band.disconnect()


def meter(mic, stop, label, score=None):
    """A live level bar (and, with score, how much the last 2 s sounds like the baseline)."""
    last = 0.0
    shown = ""
    while not stop.is_set():
        bars = int(max(0, min(30, (mic.level + 60) / 2)))
        extra = ""
        if score and time.monotonic() - last > 1.0:
            last = time.monotonic()
            speech = enroll.speech_only(mic.peek(3.0), margin=6.0)[0]
            if len(speech) >= SR:
                s = score(speech[-2 * SR:])
                shown = "  sounds like you: %.2f" % s
        extra = shown
        sys.stdout.write("\r  %s [%-30s] %4.0f dB%s   " % (label, "#" * bars, mic.level, extra))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\n")


def record(mic, label, score=None):
    input("  Press Enter, then %s" % label)
    mic.start()
    stop = threading.Event()
    t = threading.Thread(target=meter, args=(mic, stop, "listening", score), daemon=True)
    t.start()
    input()
    stop.set()
    t.join()
    return mic.stop()


def load_encoder():
    print("Loading the voice model (a few seconds)...")
    from familiar.voices import load_encoder as load
    return load()


def run_enroll(number):
    encode = load_encoder()
    mic = BandMic(number)
    print("Connected to %s. The microphone is on (solid blue light); the sound stays on this laptop.\n" % mic.band.name)
    print("Step 2 of 3: Teach it your voice. Hold the band where you normally wear it, in a quiet spot.\n")
    samples = []
    try:
        for i, prompt in enumerate(enroll.PROMPTS):
            while True:
                print("Sample %d of 3. Read this out loud, then press Enter when you're done:\n\n    \"%s\"\n" % (i + 1, prompt))
                audio = record(mic, "read the sentence:")
                speech, problem = enroll.check_sample(audio)
                if problem is None:
                    print("  Got it: %.1f s of speech.\n" % (len(speech) / SR))
                    samples.append(speech)
                    break
                print("  Again, please: %s.\n" % problem)
    finally:
        mic.close()
    profile = enroll.build_profile(samples, encode)
    path = enroll.save_profile(profile)
    print("Saved your voice baseline to %s (192 numbers, no audio; delete with --forget-voice)." % path)
    print("  %.1f s of speech, consistency between samples %.2f, starting threshold %.2f." % (
        profile["seconds"], profile["consistency"], profile["threshold"]))
    if profile["consistency"] < 0.5:
        print("  The three samples sound fairly different from each other (noise, distance, or another voice?). "
              "Consider enrolling again somewhere quieter.")
    print("Next: python3 run.py %s --voice-test" % number)


def run_voice_test(number):
    profile = enroll.load_profile()
    if profile is None:
        sys.exit("No voice baseline yet. First: python3 run.py %s --enroll" % number)
    encode = load_encoder()
    score = lambda x: float(encode(x) @ profile["vector"])
    mic = BandMic(number)
    print("Connected to %s. Microphone on; the sound stays on this laptop. Baseline from %s.\n" % (
        mic.band.name, profile.get("created")))
    try:
        print("Part 1: YOU talk, about 30 seconds, about anything (ask questions, tell a story).")
        you = record(mic, "start talking. Press Enter again when done:", score)
        print("Part 2: OTHER PEOPLE talk, about 30 seconds, where they'd normally stand. You stay quiet.")
        others = record(mic, "let them start. Press Enter again when done:", score)
    finally:
        mic.close()
    you_speech = enroll.speech_only(you, margin=6.0)[0]
    other_speech = enroll.speech_only(others, margin=6.0)[0]
    print("\nSpeech heard: you %.0f s, others %.0f s." % (len(you_speech) / SR, len(other_speech) / SR))
    if len(you_speech) < 5 * SR or len(other_speech) < 5 * SR:
        print("Not enough speech in one of the parts (5 s each needed): run the test again.")
        return
    result = enroll.report(you_speech, other_speech, profile, encode)
    print("Loudness: you %s dB, others %s dB.\n" % (result["loudness_db"]["you"], result["loudness_db"]["others"]))
    for w, r in result["windows"].items():
        s, a = r["separation"], r["at_profile_threshold"]
        print("%s pieces: you score %.2f typical (lowest %.2f), others %.2f typical (highest %.2f)." % (
            w, s["you"]["median"], s["you"]["min"], s["others"]["median"], s["others"]["max"]))
        print("   best threshold %.2f -> %.0f%% right (equal error rate %.0f%%); at the baseline's %.2f -> %.0f%% right, "
              "you missed %.0f%%, others taken for you %.0f%%." % (
                  s["threshold"], 100 * s["accuracy"], 100 * s["eer"], a["threshold"], 100 * a["accuracy"],
                  100 * a["miss"], 100 * a["false_accept"]))
    two = result["windows"]["2s"]["separation"]
    if two["you"]["min"] > two["others"]["max"]:
        suggested = round((two["you"]["min"] + two["others"]["max"]) / 2, 3)
    else:
        suggested = two["threshold"]
    print("\nSuggested threshold for 2 s of speech: %.2f (the baseline uses %.2f)." % (suggested, profile["threshold"]))
    if input("Use %.2f from now on? [y/N] " % suggested).strip().lower().startswith("y"):
        saved = {k: v for k, v in profile.items() if k != "vector"}
        saved["threshold"] = suggested
        saved["tested"] = datetime.datetime.now().isoformat(timespec="seconds")
        enroll.save_profile(saved)
        print("Saved.")
    out = os.path.join(enroll.APP_DIR, "eval", "results")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "voice_test_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    with open(path, "w") as f:
        json.dump(dict(result, seconds={"you": len(you_speech) / SR, "others": len(other_speech) / SR}), f, indent=1)
    print("Scores saved to %s (no audio)." % path)


# ---------------------------------------------------------------- voice only: who is talking right now

NAMES = {"you": "You", "silence": "Quiet", "unclear": "Not sure yet"}


def display(speaker):
    return NAMES.get(speaker) or ("Person %s" % speaker[1:] if speaker.startswith("p") else speaker)


def run_voice_only(number, port=8765, open_browser=True):
    from familiar.server import make_voice_server
    from familiar.tracker import VoiceTracker
    profile = enroll.load_profile()
    encode = load_encoder()
    started = time.monotonic()

    def changed(speaker, score):
        print("  %6.1f s  %-14s %s" % (time.monotonic() - started, display(speaker),
                                       "" if score is None else "(%.2f)" % score))
    tracker = VoiceTracker(encode, me=profile, on_change=changed)
    server = make_voice_server(tracker, port, started)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/voice" % port
    mic = BandMic(number)
    mic.listeners.append(tracker.feed)
    print("Connected to %s. Microphone on; the sound stays on this laptop. Voice only: no transcript, nothing sent." % mic.band.name)
    print("Baseline: %s. Page: %s  (Ctrl+C to stop)\n" % (
        "threshold %.2f for 1 s windows" % tracker.me["threshold"] if tracker.me else "none (run --enroll for \"You\")", url))
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    stop = threading.Event()
    threading.Thread(target=tracker.run, args=(stop,), daemon=True).start()
    try:
        while True:
            mic.band.wait(0.5)
    finally:
        stop.set()
        mic.close()
        server.shutdown()


# ---------------------------------------------------------------- the turn test: real voices, known truth

def turn_schedule(rounds=3):
    plan = []
    for _ in range(rounds):
        plan += [("you", 8.0), ("quiet", 2.0), ("other", 8.0), ("quiet", 2.0)]
    plan += [("you", 4.0), ("other", 4.0), ("you", 4.0), ("other", 4.0)]    # quick hand-overs, no pause
    return plan


SAY = {"you": "YOU talk now", "other": "TEAMMATE talks now (you stay quiet)", "quiet": "everyone quiet"}
GRACE = 1.0          # seconds after each prompt not scored: people need a moment to start or stop


def score_turns(log, segments):
    """log: [(t, shown, raw_you)]; segments: [(kind, start, end)] -> the numbers."""
    out = {"you": {"right": 0, "wrong": 0, "unclear": 0, "quiet": 0, "n": 0},
           "other": {"right": 0, "wrong": 0, "unclear": 0, "quiet": 0, "n": 0},
           "quiet": {"right": 0, "wrong": 0, "unclear": 0, "quiet": 0, "n": 0}, "delays": [], "flips": 0}
    you_raw, other_raw = [], []
    for kind, a, b in segments:
        moments = [(t, s, r) for t, s, r in log if a + GRACE <= t <= b]
        prev = None
        for t, shown, raw in moments:
            c = out[kind]
            c["n"] += 1
            if kind == "quiet":
                c["right" if shown == "silence" else "wrong"] += 1
            elif shown == "silence":
                c["quiet"] += 1
            elif shown == "unclear":
                c["unclear"] += 1
            elif (kind == "you" and shown == "you") or (kind == "other" and shown.startswith("p")):
                c["right"] += 1
            else:
                c["wrong"] += 1
            if prev is not None and shown != prev:
                out["flips"] += 1
            prev = shown
            if raw is not None:
                (you_raw if kind == "you" else other_raw if kind == "other" else []).append(raw)
        if kind != "quiet":
            hit = next((t for t, s, _ in log if a <= t <= b and
                        ((kind == "you" and s == "you") or (kind == "other" and s.startswith("p")))), None)
            out["delays"].append(None if hit is None else round(hit - a, 2))
    out["calibration"] = enroll.separation(you_raw, other_raw)
    out["others_made"] = len({s for _, s, _ in log if s.startswith("p")})
    return out


def run_turn_test(number, rounds=3):
    from familiar.tracker import HOP, VoiceTracker
    profile = enroll.load_profile()
    if profile is None:
        sys.exit("No voice baseline yet. First: python3 run.py %s --enroll" % number)
    encode = load_encoder()
    tracker = VoiceTracker(encode, me=profile)
    mic = BandMic(number)
    mic.listeners.append(tracker.feed)
    plan = turn_schedule(rounds)
    print("Connected to %s. Microphone on; the sound stays on this laptop.\n" % mic.band.name)
    print("Turn test, about %d seconds. Follow the prompts: when it says YOU, talk; when it says TEAMMATE, one teammate "
          "talks (standing where they normally would) and you stay quiet. Keep talking for the whole turn." % sum(d for _, d in plan))
    log, segments = [], []
    try:
        input("Press Enter to start.")
        t0 = time.monotonic()
        t = 0.0
        for kind, seconds in plan:
            segments.append((kind, t, t + seconds))
            t += seconds
        i = 0
        next_step = t0
        while True:
            now = time.monotonic() - t0
            while i < len(segments) and now > segments[i][2]:
                i += 1
            if i >= len(segments):
                break
            kind, a, b = segments[i]
            state = tracker.step()
            log.append((round(now, 2), state["speaker"], state.get("raw", {}).get("you")))
            sys.stdout.write("\r  >> %-38s %3.0f s left   showing: %-14s" % (SAY[kind], b - now, display(state["speaker"])))
            sys.stdout.flush()
            next_step += HOP
            time.sleep(max(0.0, next_step - time.monotonic()))
        print("\n")
    finally:
        mic.close()
    r = score_turns(log, segments)
    for kind in ("you", "other", "quiet"):
        c = r[kind]
        if c["n"]:
            print("%-6s turns: right %3.0f%%  wrong %3.0f%%  not sure %3.0f%%  shown as quiet %3.0f%%  (%d moments)" % (
                kind, 100 * c["right"] / c["n"], 100 * c["wrong"] / c["n"], 100 * c["unclear"] / c["n"],
                100 * c["quiet"] / c["n"], c["n"]))
    d = sorted(x for x in r["delays"] if x is not None)
    print("Shown right after (from the prompt, including your reaction): median %s s; never: %d of %d turns." % (
        d[len(d) // 2] if d else "n/a", sum(1 for x in r["delays"] if x is None), len(r["delays"])))
    print("Changes of the shown speaker inside turns (flicker): %d. Teammate voices created: %d (1 expected)." % (
        r["flips"], r["others_made"]))
    cal = r["calibration"]
    if cal:
        print("1 s windows: you typical %.2f (lowest %.2f), teammate typical %.2f (highest %.2f); best threshold %.2f "
              "(%.0f%% right); the tracker now uses %.2f." % (
                  cal["you"]["median"], cal["you"]["min"], cal["others"]["median"], cal["others"]["max"],
                  cal["threshold"], 100 * cal["accuracy"], tracker.me["threshold"]))
        if abs(cal["threshold"] - tracker.me["threshold"]) >= 0.02 and \
                input("Use %.2f for live tracking from now on? [y/N] " % cal["threshold"]).strip().lower().startswith("y"):
            saved = {k: v for k, v in profile.items() if k != "vector"}
            saved["tracker_threshold"] = cal["threshold"]
            enroll.save_profile(saved)
            print("Saved.")
    out = os.path.join(enroll.APP_DIR, "eval", "results")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "turn_test_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    with open(path, "w") as f:
        json.dump({"result": r, "log": log, "segments": segments}, f)
    print("Saved to %s (scores and timings only, no audio)." % path)


# ---------------------------------------------------------------- enrolment for the speaker service (diart path)

def _wav(audio):
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _http(method, url, body=None, content_type="audio/wav"):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": content_type} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def run_enroll_service(number, url):
    """Record 3 clean clips on the band; the speaker service checks each with its own models and saves the
    wearer profile. url: the service's http address, e.g. http://127.0.0.1:8790"""
    url = url.replace("ws://", "http://").replace("wss://", "https://").rstrip("/")
    status, ready = _http("GET", url + "/ready")
    if status != 200:
        sys.exit("Speaker service not ready: %s" % (ready.get("models", {}).get("error") or ready))
    _http("DELETE", url + "/enrollment")
    mic = BandMic(number)
    print("Connected to %s. Microphone on. Clips go to the speaker service on this laptop.\n" % mic.band.name)
    accepted = 0
    try:
        for i, prompt in enumerate(enroll.PROMPTS):
            while True:
                print("Clip %d of 3, alone and somewhere quiet. Read, then press Enter:\n\n    \"%s\"\n" % (i + 1, prompt))
                audio = record(mic, "read the sentence:")
                status, result = _http("POST", url + "/enrollment/clips", _wav(audio))
                if result.get("ok"):
                    accepted = result.get("accepted_clips", accepted + 1)
                    print("  Accepted (%.1f s of speech).\n" % result.get("seconds", 0))
                    break
                print("  Again, please: %s.\n" % (result.get("problem") or "HTTP %d" % status))
    finally:
        mic.close()
    status, result = _http("POST", url + "/enrollment/commit", b"{}", "application/json")
    if not result.get("ok"):
        sys.exit("Enrolment not saved: %s" % result.get("problem"))
    print("Wearer profile saved by the speaker service (%s clips, %.1f s, consistency %s, model %s)." % (
        result.get("clips"), result.get("seconds", 0), result.get("consistency"), result.get("embedding_model")))
