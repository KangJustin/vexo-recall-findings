"""Deepgram with speaker labels, on top of the band kit's own speech.py.

The kit's DeepgramSpeech already sends the band's Opus packets and handles keys, errors and
keep-alives. This subclass only adds `diarize=true` and reads the per-word `speaker` field,
so each finished piece of speech arrives as (voice number, text).

The sound still goes to Deepgram (see docs/deepgram.md in the kit): tell the people near the band.
"""
import itertools
import os
import sys

KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vexo-open-kit")
sys.path.insert(0, os.path.join(KIT, "examples", "python"))

try:
    import speech  # the kit's speech.py (it also puts the kit's library on the path)
except ImportError as problem:  # pragma: no cover
    raise SystemExit("Could not find the Vexo Open kit next to this folder (vexo-open-kit/). "
                     "Run:  git clone https://github.com/VexoAI-Org/vexo-open-kit.git   (%s)" % problem)


_CONNECTIONS = itertools.count(1)


def _seconds(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_spans(words):
    """[(speaker, text, start, end, timed), ...] from Deepgram's word list: neighbours with the same voice are
    joined. start and end are seconds since the stream began (None if Deepgram sent no timing); timed is
    [(word, start, end), ...] so a sentence can later be split where the voice changes."""
    pieces = []
    for word in words or []:
        text = str(word.get("punctuated_word") or word.get("word") or "").strip()
        if not text:
            continue
        speaker, start, end = word.get("speaker"), _seconds(word.get("start")), _seconds(word.get("end"))
        if pieces and pieces[-1][0] == speaker:
            pieces[-1][1].append(text)
            pieces[-1][3] = end if end is not None else pieces[-1][3]
        else:
            pieces.append([speaker, [text], start, end, []])
        if start is not None and end is not None:
            pieces[-1][4].append((text, start, end))
    return [(speaker, " ".join(parts), start, end, timed) for speaker, parts, start, end, timed in pieces]


def group_words(words):
    """[(speaker, text), ...]: group_spans without the timing."""
    return [(speaker, text) for speaker, text, _, _, _ in group_spans(words)]


class DiarizedDeepgram(speech.DeepgramSpeech):
    """on_piece(speaker, text, final) is called from another thread, for guesses and settled text.

    Deepgram settles a sentence in chunks ("Hi, I'm" ... "Maya, I work at Northline."). The name
    rules need the whole sentence, so settled chunks are held per voice and handed on as one final
    line when Deepgram says the sentence ended (speech_final / UtteranceEnd) or another voice starts."""

    def __init__(self, on_piece=None, on_sentence=None, on_interim=None, **options):
        # Deepgram's current switch for speaker labels (diarize=true is deprecated; never send both: rejected).
        # On a live stream "latest" is the v1 diarizer today; streaming words carry `speaker` but no
        # `speaker_confidence` (that exists only for recorded audio).
        if "diarize" not in options:
            options.setdefault("diarize_model", "latest")
        options.setdefault("endpointing", 300)          # the kit's 100 ms splits "I'm ... Maya" too often
        options.setdefault("utterance_end_ms", "1000")  # a sentence end that still arrives in a noisy room
        super().__init__(**options)
        self.on_piece = on_piece
        # on_sentence(speaker, text, start, end, words, connection): if given, settled sentences go there
        # INSTEAD of on_piece (voice matching decides who said them, then feeds the Resolver itself).
        self.on_sentence = on_sentence
        # on_interim(pieces, connection): the whole provisional text, [(speaker, text), ...], REPLACING the
        # previous one on every update (Deepgram revises interim results; they are never appended).
        self.on_interim = on_interim
        self.connection = 0                             # speaker numbers mean something only inside one connection
        self.labels = {"words": 0, "with_speaker": 0, "speakers": set()}   # is Deepgram labelling voices at all?
        self._held = None                               # [speaker, [chunks], start, end, [timed words]] not yet handed on
        self._interim = []                              # pieces of the latest interim result (superseded by the next)

    def start(self, *args, **kwargs):
        self.connection = next(_CONNECTIONS)            # a new connection: its "speaker 0" is nobody we know yet
        return super().start(*args, **kwargs)

    def _emit(self, speaker, text, final):
        if final:
            self.finals.append(text)
        if self.on_piece:
            self.on_piece(speaker, text, final)

    def provisional(self):
        """What is not settled yet: the held finalized chunks, then the latest interim guess."""
        out = []
        if self._held and self._held[1]:
            out.append((self._held[0], " ".join(self._held[1])))
        for speaker, text in self._interim:
            if out and (speaker is None or speaker == out[-1][0]):
                out[-1] = (out[-1][0], out[-1][1] + " " + text)
            else:
                out.append((speaker, text))
        return out

    def _show(self):
        pieces = self.provisional()
        if self.on_interim:
            self.on_interim(pieces, self.connection)
        else:
            for speaker, text in pieces:
                self._emit(speaker, text, False)

    def _flush(self):
        held, self._held = self._held, None
        if held and held[1]:
            text = " ".join(held[1])
            if not self.on_sentence:
                return self._emit(held[0], text, True)
            self.finals.append(text)
            try:
                self.on_sentence(held[0], text, held[2], held[3], held[4], self.connection)
            except Exception as problem:                # voice matching must never stop the transcript
                print("Voice matching: %s" % type(problem).__name__, file=sys.stderr)
                if self.on_piece:
                    self.on_piece(held[0], text, True)

    def _sentence_ends(self):                           # the kit calls this when the connection closes
        self._interim = []
        self._flush()

    def _message(self, message):
        kind = message.get("type")
        if kind == "UtteranceEnd":
            self._flush()
            return self._show() if self.on_interim else None
        if kind != "Results":
            return super()._message(message)
        try:
            best = (message.get("channel", {}).get("alternatives") or [{}])[0]
            words = best.get("words") or []
            if message.get("is_final"):
                self.labels["words"] += len(words)
                for word in words:
                    if word.get("speaker") is not None:
                        self.labels["with_speaker"] += 1
                        self.labels["speakers"].add(word.get("speaker"))
            pieces = group_spans(words)
            if not pieces:
                text = str(best.get("transcript") or "").strip()
                pieces = [(None, text, None, None, [])] if text else []
            if not message.get("is_final"):             # a guess: it replaces the previous guess
                self._interim = [(speaker, text) for speaker, text, _, _, _ in pieces]
                return self._show()
            self._interim = []                          # the final result supersedes the guesses for this audio
            for speaker, text, start, end, timed in pieces:
                if self._held and speaker is not None and speaker != self._held[0]:
                    self._flush()                       # another voice took over: the last one is done
                if self._held:
                    self._held[1].append(text)
                    self._held[3] = end if end is not None else self._held[3]
                    self._held[4].extend(timed)
                else:
                    self._held = [speaker, [text], start, end, list(timed)]
            if message.get("speech_final"):
                self._flush()
            self._show()
        except Exception as problem:  # the reader thread swallows errors silently; say it
            print("Could not read an answer from Deepgram: %s" % problem, file=sys.stderr)
