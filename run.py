#!/usr/bin/env python3
"""Vexo Recall demo: who is talking, in the moment.

  python3 run.py --sim            scripted conversation, no band and no key needed (safe demo fallback)
  python3 run.py --type           type lines yourself in this terminal:   1: Hi, I'm Maya
  python3 run.py 42               your band (the number on its bag) + Deepgram, live
  python3 run.py 42 --flash       also blink the band's light green when a name is picked up
  add --no-gemini to any of them  names from the built-in rules only (no text goes to Google)
  python3 run.py 42 --no-voices   live without voice matching (voiceprints on this laptop)
  python3 run.py 42 --second-pass deepgram     re-check risky moments on recorded audio (suggestions only)
  python3 run.py 42 --gemini-review            experimental: Gemini reviews risky audio (suggestions only)
  python3 run.py 42 --enroll      "Teach it your voice": 3 sentences -> your voice baseline (on this laptop)
  python3 run.py 42 --voice-test  you talk, then others talk: how well the baseline tells you apart
  python3 run.py --forget-voice   delete your saved voice baseline
  python3 run.py 42 --voice-only  who is talking right now, from the sound alone (page /voice; no transcript)
  python3 run.py 42 --turn-test   guided you/teammate turns: real-voice accuracy and reaction time of --voice-only

Open http://127.0.0.1:8765 (it opens by itself). Needs DEEPGRAM_API_KEY for the live mode, and
GEMINI_API_KEY (optional) for Gemini's second opinion on names: see README.md.
"""
import argparse
import os
import queue
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from familiar.resolver import Resolver  # noqa: E402
from familiar.server import Hub, make_server  # noqa: E402
from familiar import gemini  # noqa: E402

# A made-up conversation that shows every kind of clue. Speaker 0 is the person wearing the band.
SCRIPT = [
    (0, "Hey, welcome! Are you here for the hardware meetup?", 1.2),
    (1, "Yeah, first time. Hi, I'm Maya, I work at Northline.", 2.2),
    (0, "Nice to meet you, Maya. What do you do there?", 2.0),
    (1, "I'm a product designer. I'm working on a robot gripper enclosure.", 2.6),
    (0, "Oh nice, we've been struggling with gripper housings too.", 2.4),
    (2, "Sorry to interrupt. Do you know where the coffee is?", 2.4),
    (0, "Over by the door. Hi Dev, by the way, good to see you again.", 2.6),
    (2, "Thanks! I study robotics at Berkeley, I'm from Oakland.", 2.4),
    (3, "Is this seat taken?", 2.0),
    (0, "Go ahead. What's your name?", 1.6),
    (3, "Priya.", 1.4),
]


def play_script(book, speed=1.0):
    book.set_me(0)
    time.sleep(1.0 / speed)
    for speaker, text, pause in SCRIPT:
        words = text.split()
        for i in range(1, len(words) + 1):                      # guesses arrive word by word, like Deepgram
            book.feed(speaker, " ".join(words[:i]), final=False)
            time.sleep(0.11 / speed)
        book.feed(speaker, text, final=True)
        time.sleep(pause / speed)
    print("Scripted conversation finished. The page stays open; Ctrl+C to quit.")


def type_lines(book):
    print("Type a line as   voice: text   (for example   1: Hi, I'm Maya).  Voice 0 is you.  Ctrl+C to quit.")
    last = 0
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        head, sep, tail = raw.partition(":")
        if sep and head.strip().isdigit():
            last, text = int(head), tail.strip()
        else:
            text = raw
        book.feed(last, text, final=True)


def start_voices(book, enabled=True):
    """Who-said-what for live mode: (VoicePrints, decoder). Deepgram's word-level labels always come first;
    with enabled and the packages installed, voiceprints on this laptop check them."""
    from familiar.voices import VoicePrints
    if not enabled:
        print("Voice matching: off (--no-voices). Speakers come from Deepgram's labels only.")
        return VoicePrints(book, enabled=False).start(), None
    try:
        from vexo_open.audio import PcmDecoder
        decoder = PcmDecoder()                                   # needs: python3 -m pip install av
        import torch, speechbrain  # noqa: F401,E401
    except Exception as problem:
        print("Voice matching is off (%s). Install with: python3 -m pip install torch speechbrain av"
              % (str(problem).splitlines()[0][:120] or type(problem).__name__))
        return VoicePrints(book, enabled=False).start(), None
    from familiar import enroll
    me = enroll.load_profile()
    print("Voice matching: on this laptop only. Voiceprints stay in memory for this conversation and are never "
          "saved or sent. Turn it off with --no-voices. (Loading the voice model...)")
    if me:
        print("Your voice baseline: loaded (enrolled %s, threshold %.2f). Your card is \"You\" by voice." % (
            me.get("created"), me["threshold"]))
    else:
        print("No voice baseline yet: \"You\" is guessed from loudness. Teach it your voice: python3 run.py "
              "<band> --enroll")
    return VoicePrints(book, me=me).start(), decoder


def start_second_pass(book, voices, engine, apply, review):
    """Off unless asked: the measured results (eval/README.md) do not justify automatic corrections."""
    if engine == "off" and not review:
        return None
    from familiar import second_pass
    if engine == "deepgram":
        print("Second pass: risky moments (with ~4 s around them) are re-sent as AUDIO to Deepgram's recorded-audio "
              "diarization. %s" % ("Sure corrections are applied (undo on the page)." if apply else
                                   "Its disagreements are shown as suggestions."))
    elif engine == "gemini-transcribe":
        print("Second pass: risky moments are sent as AUDIO to Google Gemini (%s). Suggestions only."
              % second_pass.GEMINI_TRANSCRIBE_MODEL)
    rv = None
    if review:
        rv = second_pass.GeminiReview()
        print("Gemini review (experimental): risky moments are sent as AUDIO plus the transcript to Google Gemini "
              "(%s). Its answers are hypotheses, shown for you to check, never applied by themselves." % rv.model)
    sp = second_pass.SecondPass(book, voices, engine=None if engine == "off" else engine,
                                apply=apply and engine == "deepgram", review=rv)
    voices.on_segment = sp.note
    return sp


def start_speaker_service(book, voices, url, number):
    """Behind --speaker-service: diart + segmentation-3.0 + ECAPA in speaker_service/ (its own environment)."""
    from familiar.diart_align import DiartAttribution
    from familiar.speaker_client import SpeakerClient
    align = DiartAttribution(book, voices, apply=True)
    session = "band%s-%s" % (number, time.strftime("%Y%m%d-%H%M%S"))
    client = SpeakerClient(url, session, on_event=align.on_event)

    def on_change(snapshot, last=[book.epoch]):         # "Start a new conversation": a fresh service session
        if book.epoch != last[0]:
            last[0] = book.epoch
            client.restart("%s-%d" % (session, book.epoch))
    old = book.on_change
    book.on_change = lambda snap: (old(snap) if old else None, on_change(snap))
    print("Speaker service: %s (diart, pyannote/segmentation-3.0, SpeechBrain ECAPA). The band's AUDIO goes to "
          "this service on this laptop; if it is down or behind, transcription carries on and the page says "
          "\"Speaker AI unavailable\"." % url)
    return client.start()


def run_live(book, number, flash, voices_on=True, engine="off", apply=False, review=False, record=None,
             speaker_url=None):
    from familiar.live import DiarizedDeepgram
    from vexo_open.sync import Band

    flashes = queue.Queue()
    if flash:
        book.on_event = lambda event: flashes.put(event)

    voices, decoder = start_voices(book, voices_on)
    ears = DiarizedDeepgram(on_sentence=voices.settle, on_interim=voices.interim)
    speaker_client = None
    if speaker_url:
        if not decoder:
            from vexo_open.audio import PcmDecoder
            decoder = PcmDecoder()
        speaker_client = start_speaker_service(book, voices, speaker_url, number)
    if engine != "off" or review:
        if not decoder:                                          # the second pass needs the buffered sound
            from vexo_open.audio import PcmDecoder
            decoder = PcmDecoder()
        start_second_pass(book, voices, engine, apply, review)

    def deepgram_labels():
        seen = ears.labels
        book.diagnostics["deepgram"] = {"words": seen["words"], "with_speaker_label": seen["with_speaker"],
                                        "different_voices": len(seen["speakers"])}
    recording = None
    if record:
        # Only with everyone's consent: the band's AUDIO and Deepgram's raw messages, for labelling and eval/.
        import json as _json
        import wave as _wave
        os.makedirs(os.path.dirname(os.path.abspath(record)) or ".", exist_ok=True)
        if not decoder:
            from vexo_open.audio import PcmDecoder
            decoder = PcmDecoder()
        recording = {"pcm": bytearray(), "sent": 0.0, "t0": time.monotonic(), "file": open(record + ".jsonl", "w")}
        print("RECORDING: the band's audio and the transcript are saved to %s.wav / .jsonl on this laptop. Only with "
              "everyone's consent; delete them after labelling (eval/README.md)." % record)
        plain = ears._message

        def logged(message):
            recording["file"].write(_json.dumps({"received": round(time.monotonic() - recording["t0"], 3),
                                                 "audio_sent": round(recording["sent"], 3), "message": message}) + "\n")
            return plain(message)
        ears._message = logged
    ears.start()                                                 # prints who gets the sound; raises with a clear sentence if no key
    band = Band.connect(number)
    print("Connected to %s. Tell the people near you that the microphone is on (the band's light is solid blue)." % band.name)

    def on_audio(a):
        ears.feed(a.opus, a.seq)
        if decoder:
            # Decoded WITHOUT seq on purpose: no made-up sound for lost packets, so this clock matches
            # the packets Deepgram got, and its word timings point at the right samples.
            pcm = decoder.feed(a.opus)
            voices.add_audio(pcm)
            if speaker_client is not None:
                speaker_client.feed(pcm)                         # same samples, same clock as Deepgram
            if recording is not None:
                recording["pcm"] += pcm
                recording["sent"] += len(pcm) / 32000.0
    band.on("audio", on_audio)
    band.configure(audio=True)
    try:
        while True:
            band.wait(0.25)
            deepgram_labels()
            if not flashes.empty():
                while not flashes.empty():
                    flashes.get()
                # The band refuses light commands while its microphone is on, so pause the mic for half a second.
                band.configure(audio=False)
                band.set_led("#22c55e")
                band.wait(0.5)
                band.led_off()
                band.configure(audio=True)
    except KeyboardInterrupt:
        pass
    finally:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)            # a second Ctrl+C must not cut the clean-up in half
        print("\nStopping: microphone off, disconnecting...")
        for step in (lambda: band.configure(audio=False), band.disconnect):
            try:
                step()
            except Exception as problem:                         # the band may already be gone
                print("While stopping: %s" % problem, file=sys.stderr)
        if speaker_client is not None:
            speaker_client.stop()
        try:
            ears.stop()
        except Exception as problem:                             # a closing problem should not hide the result
            print("While stopping: %s" % problem, file=sys.stderr)
        if recording is not None:
            recording["file"].close()
            with _wave.open(record + ".wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(bytes(recording["pcm"]))
            print("Saved %s.wav and %s.jsonl. Label who spoke when in %s.json (eval/README.md)." % (record, record, record))


def start_gemini(book, off):
    """Gemini's second opinion on names, if there is a key. The regexes work either way."""
    if off:
        print("Gemini: off (--no-gemini). Names come from the built-in rules only.")
        return None
    key = gemini.find_key()
    if not key:
        print("Gemini: no %s found, so names come from the built-in rules only. "
              "(Add it to .env to get a second opinion.)" % gemini.KEY_NAME)
        return None
    client = gemini.make_client(key)
    # Say who gets what, before anything is sent. Keep this line.
    print("Note: to help name speakers, the transcript TEXT (not the audio) is sent to %s, model %s. "
          "Turn it off with --no-gemini." % (client.company, client.model))
    return gemini.Advisor(book, client).start()


def band_number(text):
    """"64", "64.", " 064", "Vexo-HACK-64" -> 64; anything else -> None."""
    import re
    m = re.fullmatch(r"\s*(?:vexo-hack-)?0*(\d{1,3})\W*\s*", str(text), re.I)
    return int(m.group(1)) if m and 1 <= int(m.group(1)) <= 999 else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("number", nargs="?", help="the number on your band, e.g. 42")
    p.add_argument("--sim", action="store_true", help="play a scripted conversation (no band, no key)")
    p.add_argument("--type", action="store_true", help="type lines in this terminal")
    p.add_argument("--flash", action="store_true", help="blink the band green when a name is picked up (live mode)")
    p.add_argument("--no-gemini", action="store_true", help="names from the built-in rules only; send nothing to Gemini")
    p.add_argument("--no-voices", action="store_true", help="live mode: no voice matching (voiceprints on this laptop)")
    p.add_argument("--second-pass", choices=["off", "deepgram", "gemini-transcribe"], default="off",
                   help="live mode: re-check risky moments on recorded audio (default off)")
    p.add_argument("--second-pass-apply", action="store_true",
                   help="with --second-pass deepgram: apply corrections Deepgram is sure of (reversible)")
    p.add_argument("--gemini-review", action="store_true",
                   help="live mode, experimental: Gemini reviews risky audio; suggestions only")
    p.add_argument("--no-notes", action="store_true", help="no Gemini notes (topics, mentioned names, follow-ups)")
    p.add_argument("--enroll", action="store_true", help="teach it your voice: record 3 sentences, save a baseline")
    p.add_argument("--voice-test", action="store_true", help="test the voice baseline: you talk, then others talk")
    p.add_argument("--forget-voice", action="store_true", help="delete the saved voice baseline")
    p.add_argument("--voice-only", action="store_true", help="who is talking now, from the sound alone (no transcript)")
    p.add_argument("--turn-test", action="store_true", help="guided turns: measure --voice-only on real voices")
    p.add_argument("--speaker-service", metavar="URL", help="live mode, experimental: who-said-what from the diart "
                   "speaker service, e.g. ws://127.0.0.1:8790 (see speaker_service/README.md)")
    p.add_argument("--enroll-service", metavar="URL", help="enrol the wearer with the speaker service, e.g. "
                   "http://127.0.0.1:8790 (3 clips recorded on the band)")
    p.add_argument("--record", metavar="PATH", help="live mode, ONLY with everyone's consent: save the audio and "
                   "Deepgram's raw messages to PATH.wav/.jsonl for labelling (see eval/README.md)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--speed", type=float, default=1.0, help="with --sim: 2 plays twice as fast")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if args.number is not None:
        number = band_number(args.number)
        if number is None:
            p.error("%r is not a band number: use the number on your band's bag, e.g. python3 run.py 42" % args.number)
        args.number = str(number)
    if args.forget_voice:
        from familiar import enroll
        print("Deleted your voice baseline." if enroll.forget_profile() else "There was no voice baseline saved.")
        return
    if args.enroll or args.voice_test or args.voice_only or args.turn_test or args.enroll_service:
        if not args.number:
            p.error("give your band number, e.g. python3 run.py 42 --enroll")
        from familiar import live  # noqa: F401  (puts the kit on the path)
        from familiar import band_voice
        from vexo_open.sync import VexoError
        try:
            if args.enroll_service:
                band_voice.run_enroll_service(int(args.number), args.enroll_service)
            elif args.enroll:
                band_voice.run_enroll(int(args.number))
            elif args.voice_test:
                band_voice.run_voice_test(int(args.number))
            elif args.turn_test:
                band_voice.run_turn_test(int(args.number))
            else:
                band_voice.run_voice_only(int(args.number), args.port, not args.no_browser)
        except VexoError as problem:
            sys.exit("Problem: %s" % problem)
        except KeyboardInterrupt:
            print("\nStopped.")
        return
    if not (args.sim or args.type or args.number):
        p.error("give your band number (python3 run.py 42), or use --sim or --type")

    if not (args.sim or args.type):
        # Check the kit and the key before opening the page, so a missing key does not leave a dead tab.
        from familiar.live import speech
        from vexo_open.sync import VexoError
        try:
            speech.find_key()
        except VexoError as problem:
            sys.exit("Problem: %s" % problem)

    book, hub = Resolver(), Hub()
    try:
        server = make_server(book, hub, args.port)
    except OSError as problem:
        if problem.errno in (48, 98, 10048):          # address in use: macOS, Linux, Windows
            sys.exit("Port %d is taken, most likely by Vexo Recall already running in another terminal. "
                     "Open http://127.0.0.1:%d, or press Ctrl+C in that terminal first. "
                     "(Something else? Use --port 8766.)" % (args.port, args.port))
        raise
    threading.Thread(target=server.serve_forever, daemon=True).start()
    advisor = start_gemini(book, args.no_gemini)
    notes = None
    if advisor and not args.no_notes:
        notes = gemini.NotesWorker(book, gemini.ConversationNotes()).start()
        print("Gemini notes: topics, mentioned names and follow-ups from the transcript TEXT (model %s). "
              "Turn off with --no-notes." % gemini.NOTES_MODEL)
    url = "http://127.0.0.1:%d" % args.port
    print("Demo page: %s" % url)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        if args.sim:
            play_script(book, args.speed)
            while True:
                time.sleep(1)
        elif args.type:
            type_lines(book)
        else:
            from familiar import live  # noqa: F401  (finds the kit before the band is used)
            from vexo_open.sync import VexoError
            try:
                run_live(book, int(args.number.lower().replace("vexo-hack-", "")), args.flash, not args.no_voices,
                         args.second_pass, args.second_pass_apply, args.gemini_review, args.record,
                         args.speaker_service)
            except VexoError as problem:
                sys.exit("Problem: %s" % problem)
    except KeyboardInterrupt:
        pass
    finally:
        if advisor:
            advisor.stop()
        if notes:
            notes.stop()
        server.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)          # background threads (Bluetooth, speech, Gemini) must never keep the program alive


if __name__ == "__main__":
    main()
