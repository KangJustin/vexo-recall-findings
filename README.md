> **Archived experiment (Vexo Hacks, Nov 2026).** Live "who said what" on a Vexo wristband did not work well
> enough on real voices to demo, and the team moved on to Familiar. **Read [FINDINGS.md](FINDINGS.md) first**:
> what was built, what was measured, and what to try next. The rest of this README is how to run it.

# Vexo Recall — who is talking, in the moment

Hackathon demo. The band's mic goes to Deepgram with speaker labels (diarization). Names come from what people say
("Hi, I'm Maya", "Nice to meet you, Dev", "What's your name?" → "Priya"). Anything unsure is marked "possible";
anything missing can be typed in on the page. No names or voices are stored.

## Run it
```
git clone https://github.com/VexoAI-Org/vexo-open-kit.git      # next to run.py (git-ignored)
# create .env yourself (git-ignored; never commit or paste a key) with the line  DEEPGRAM_API_KEY=...
# and, optionally, the line  GEMINI_API_KEY=...  for Gemini's second opinion on names
python3 run.py --sim          # scripted demo, no band or key: the safe fallback
python3 run.py --type         # type lines yourself:  1: Hi, I'm Maya
python3 run.py 42             # live: your band number (double-tap the band first, close the kit dashboard tab)
python3 run.py 42 --flash     # also blink the band green on a new name (untested on hardware)
python3 run.py --sim --no-gemini   # names from the built-in rules only (works with every mode)
python3 -m unittest discover -s tests
```
Page: http://127.0.0.1:8765 (local only).

## Things to know
- The band has no vibration or screen, only a light. It also refuses light commands while its mic is on, so
  the page is the main feedback; `--flash` pauses the mic for 0.5 s to blink.
- Diarization labels voices only within one stream and can split one person in two: use "Same as…" on the page.
- Tell people near the band the mic is on; the audio goes to Deepgram.
- `familiar/resolver.py` is the logic (name rules, tested); `familiar/live.py` adds `diarize=true` to the kit's Deepgram class.

## Gemini's second opinion on names (Gemini Decision 1)
The regexes in `familiar/resolver.py` stay the instant, offline answer. With `GEMINI_API_KEY` set (environment
or `.env`), `familiar/gemini.py` also sends the last ~20 lines of **transcript text, never audio** to Google's
**Gemini Decisions API** (model `gemini-decision-1`), at most once every ~2 s, on a background thread with a
4 s timeout. The program prints a line saying so at startup; `--no-gemini` turns it off.
- Decision 1 writes no text: it answers closed questions with calibrated confidence (~150-250 ms, input
  tokens only). So the app collects candidate names from the transcript and asks, in one request: for each
  voice, which candidate is its name (or `no_name_yet`); for each pair of voices, whether they are one
  person split in two (yes/no); and, if nobody said yet, which voice wears the band.
- Its limit: it can only choose names that appear in the transcript. A misheard name stays misheard; type it.
- Typed names are never asked about and never offered to another voice; rejected names are never offered.
- Merge rules (`Resolver.apply_model_opinion`): typed names always win; confidence >= 0.8 gives "heard",
  0.5-0.8 gives "possible", lower is ignored; the evidence shown is the transcript line with the name;
  a "heard" name is never replaced (the other one goes to alts).
- "Same person" at >= 0.85 joins two voices by itself only if they never spoke at the same time and do not
  have two different names; 0.5-0.85, or overlapping voices, shows a "Same person?" card on the page.
- Each name shows its source on the page (rules, Gemini, or typed by you). If Gemini is slow or fails, the
  page shows "Rules only" and carries on.
- `GEMINI_ENGINE=text` uses a text model on the Interactions API with structured output instead
  (`gemini-3.5-flash-lite`, or `GEMINI_MODEL`); it can propose names the rules never saw, but is slower.

## Who said what (live mode)
Deepgram's word-level speaker labels come first (`diarize_model=latest`; on a live stream that is Deepgram's v1
diarizer, and streaming words carry `speaker` but no `speaker_confidence`). `familiar/live.py` splits a line
wherever the label changes, even mid-sentence, so a short "wait" or "yeah" keeps its own line. Interim results
are provisional and are REPLACED on every update, never appended. Labels are scoped to their connection: a new
connection's "speaker 0" is not assumed to be anyone already seen.

`familiar/voices.py` maps each (connection, label) to a person of this conversation. Voiceprints (SpeechBrain
ECAPA on this laptop; `python3 -m pip install torch speechbrain av`; `--no-voices` turns them off) may only:
recognise a NEW label as someone already heard (1.5 s+ of speech), split a long line where the voice clearly
changes inside it, or leave a line **unresolved** ("Unclear who") when they clearly disagree with the label.
They never overrule a label on a short clip, and they learn only from clear single-speaker audio.

Each line keeps its audio times, words and attribution status (provider, voice, corrected, uncertain),
separate from whether its text is final. Overlap is marked only when two speakers' words overlap in time.

Optional, off by default, measured in `eval/` (see `eval/results/report.md`):
- `--second-pass deepgram`: re-sends risky moments (~4 s either side) to Deepgram's recorded-audio
  diarizer and shows disagreements as suggestions (Apply / Dismiss / Undo on the page).
  `--second-pass-apply` applies what Deepgram is sure of; measured, it made results WORSE with voice matching on.
- `--gemini-review` (experimental): an audio-capable Gemini model (`GEMINI_REVIEW_MODEL`, default
  gemini-4-hydrogen) reviews risky audio with Deepgram's words and must answer in a fixed JSON schema;
  validated answers are shown as unverified suggestions only.
- `--second-pass gemini-transcribe`: Gemini 3.5 Transcribe as the second pass (`GEMINI_TRANSCRIBE_MODEL`);
  it returned 404 for this project on 14 Nov, so it is wired but not measured.
- Gemini notes (on with a Gemini key, `--no-notes` to turn off): topics, names MENTIONED, follow-ups. Who said
  something comes from the app's attribution of the cited line, never from Gemini; a name that is mentioned
  never becomes anyone's identity, and statements on unresolved lines stay unattributed.
- `--record PATH`: ONLY with everyone's consent, saves the audio and Deepgram's raw messages for labelling.

`http://127.0.0.1:8765/debug` shows each line's attribution decision and the voiceprint numbers.

## Your voice baseline ("Teach it your voice")
```
python3 run.py 64 --enroll        # read 3 sentences through the band -> profiles/me.json (on this laptop)
python3 run.py 64 --voice-test    # you talk ~30 s, then others talk ~30 s: how well are you told apart?
python3 run.py --forget-voice     # delete it
```
Nothing is trained: the pretrained speaker model turns your three samples into one voiceprint (192 numbers, no
audio is kept). Each sample is checked (enough speech, not distorted, not drowned in noise) and re-asked if
not. The voice test scores 1, 2 and 3 s pieces against the baseline, reports how well "you" and "not you"
separate (best threshold, accuracy, equal error rate) and can save a better threshold; it saves scores only.
In live mode the baseline makes your card "You" by voice from your first clear line (instead of guessing from
loudness); it is fixed and never updated by live audio. `eval/voice_baseline.py` runs the same check on macOS
voices without a band (synthetic voices separate far more easily than real ones).
