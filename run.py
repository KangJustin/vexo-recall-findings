#!/usr/bin/env python3
"""Vexo Recall demo: who is talking, in the moment.

  python3 run.py --sim            scripted conversation, no band and no key needed (safe demo fallback)
  python3 run.py --type           type lines yourself in this terminal:   1: Hi, I'm Maya
  python3 run.py 42               your band (the number on its bag) + Deepgram, live
  python3 run.py 42 --flash       also blink the band's light green when a name is picked up

Open http://127.0.0.1:8765 (it opens by itself). Needs DEEPGRAM_API_KEY for the live mode:
see README.md.
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


def run_live(book, number, flash):
    from familiar.live import DiarizedDeepgram
    from vexo_open.sync import Band

    flashes = queue.Queue()
    if flash:
        book.on_event = lambda event: flashes.put(event)

    ears = DiarizedDeepgram(on_piece=lambda speaker, text, final: book.feed(speaker, text, final))
    ears.start()                                                 # prints who gets the sound; raises with a clear sentence if no key
    band = Band.connect(number)
    print("Connected to %s. Tell the people near you that the microphone is on (the band's light is solid blue)." % band.name)
    band.on("audio", lambda a: ears.feed(a.opus, a.seq))
    band.configure(audio=True)
    try:
        while True:
            band.wait(0.25)
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
        band.configure(audio=False)
        band.disconnect()
        try:
            ears.stop()
        except Exception as problem:                             # a closing problem should not hide the result
            print("While stopping: %s" % problem, file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("number", nargs="?", help="the number on your band, e.g. 42")
    p.add_argument("--sim", action="store_true", help="play a scripted conversation (no band, no key)")
    p.add_argument("--type", action="store_true", help="type lines in this terminal")
    p.add_argument("--flash", action="store_true", help="blink the band green when a name is picked up (live mode)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--speed", type=float, default=1.0, help="with --sim: 2 plays twice as fast")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if not (args.sim or args.type or args.number):
        p.error("give your band number (python3 run.py 42), or use --sim or --type")

    book, hub = Resolver(), Hub()
    server = make_server(book, hub, args.port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
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
                run_live(book, int(args.number.lower().replace("vexo-hack-", "")), args.flash)
            except VexoError as problem:
                sys.exit("Problem: %s" % problem)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
