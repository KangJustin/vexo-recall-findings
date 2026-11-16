"""Deepgram with speaker labels, on top of the band kit's own speech.py.

The kit's DeepgramSpeech already sends the band's Opus packets and handles keys, errors and
keep-alives. This subclass only adds `diarize=true` and reads the per-word `speaker` field,
so each finished piece of speech arrives as (voice number, text).

The sound still goes to Deepgram (see docs/deepgram.md in the kit): tell the people near the band.
"""
import os
import sys

KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vexo-open-kit")
sys.path.insert(0, os.path.join(KIT, "examples", "python"))

try:
    import speech  # the kit's speech.py (it also puts the kit's library on the path)
except ImportError as problem:  # pragma: no cover
    raise SystemExit("Could not find the Vexo Open kit next to this folder (vexo-open-kit/). "
                     "Run:  git clone https://github.com/VexoAI-Org/vexo-open-kit.git   (%s)" % problem)


def group_words(words):
    """[(speaker, text), ...] from Deepgram's word list: neighbours with the same voice are joined."""
    pieces = []
    for word in words or []:
        text = str(word.get("punctuated_word") or word.get("word") or "").strip()
        if not text:
            continue
        speaker = word.get("speaker")
        if pieces and pieces[-1][0] == speaker:
            pieces[-1][1].append(text)
        else:
            pieces.append((speaker, [text]))
    return [(speaker, " ".join(parts)) for speaker, parts in pieces]


class DiarizedDeepgram(speech.DeepgramSpeech):
    """on_piece(speaker, text, final) is called from another thread, for guesses and settled text."""

    def __init__(self, on_piece=None, **options):
        options.setdefault("diarize", "true")
        super().__init__(**options)
        self.on_piece = on_piece

    def _message(self, message):
        if message.get("type") != "Results":
            return super()._message(message)
        try:
            best = (message.get("channel", {}).get("alternatives") or [{}])[0]
            final = bool(message.get("is_final"))
            pieces = group_words(best.get("words"))
            if not pieces:
                text = str(best.get("transcript") or "").strip()
                pieces = [(None, text)] if text else []
            for speaker, text in pieces:
                if final:
                    self.finals.append(text)
                if self.on_piece:
                    self.on_piece(speaker, text, final)
        except Exception as problem:  # the reader thread swallows errors silently; say it
            print("Could not read an answer from Deepgram: %s" % problem, file=sys.stderr)
