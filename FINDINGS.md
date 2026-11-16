# Who said what, on a wristband: findings

Vexo Hacks, 14–15 Nov 2026. Justin Kang.

**Goal:** a Vexo band on the wearer's wrist hears a conversation; the app shows, live, who is talking and
puts each person's words on the right card, especially "You" (the wearer) versus everyone else, including when
people interrupt or talk over each other.

**Outcome:** it did not work well enough on real voices to demo, and the team pivoted to Familiar, which names
people by their face (camera) and only uses the voice to tell the wearer apart. This repository keeps the
experiments, the tools to measure them, and what we learned, so the next attempt starts from evidence.

## What was built

| Layer | What it does | Where |
|---|---|---|
| Live transcription | Deepgram streaming with word-level speaker labels; settled chunks held into whole sentences, interim results replaced (never appended) | `familiar/live.py` |
| Names from speech | Regex rules ("I'm Maya", "nice to meet you, Dev"), then Gemini Decision 1 as a grounded second opinion | `familiar/resolver.py`, `familiar/gemini.py` |
| Voice matching | SpeechBrain ECAPA voiceprints on the laptop: label-to-person mapping, splitting lines where the voice changes, "unresolved" instead of guessing | `familiar/voices.py` |
| Wearer baseline | "Teach it your voice": 3 read-aloud samples, quality checks, a voice test that measures separation | `familiar/enroll.py`, `familiar/band_voice.py` |
| Second pass | Deepgram recorded-audio diarization and a Gemini audio review on risky regions, as reversible suggestions | `familiar/second_pass.py` |
| Voice-only tracker | Who is talking right now from sound alone, ~4 decisions a second | `familiar/tracker.py`, `--voice-only` |
| Speaker service | diart + pyannote/segmentation-3.0 + ECAPA as a FastAPI service with overlap detection and wearer matching | `speaker_service/` |
| Measurement | Labelled synthetic conversations, a replay of recorded Deepgram output through the app, scorers | `eval/` |

## Measured results

Synthetic = six scripted conversations made with macOS text-to-speech voices (alternating turns, no pause,
short interruption, back-channel "yeah", simultaneous speech, three speakers; 214 words). Synthetic voices are
far more distinct than real people on a wrist mic, so these numbers are upper bounds.

### Deepgram live diarization (the provider itself)

| | Wrong words | Missed words | Notes |
|---|---|---|---|
| Live streaming | 39 / 214 | 19 | Both "no pause" and "three speakers" collapsed into ONE speaker label |
| Recorded audio, whole clip (v2 diarizer) | 9 / 214 | 19 | Much better, but needs the whole clip |
| Recorded audio, 8–12 s windows, applied automatically | worse than none | | Short windows lose the context that made it good |

- Overlapped interjections ("wait" said over someone) were never transcribed at all; "yeah"/"right" said over
  someone were given to the other speaker. Overlapping speech lost words in every engine tried.
- Streaming words carry `speaker` but no `speaker_confidence` (that exists only for recorded audio).
- Settled words arrive 2.3 s (median) / 4.4 s (p90) after they are spoken.

### Our attribution on top of Deepgram (synthetic)

| Version | Wrong | Unresolved | Extra / empty cards |
|---|---|---|---|
| First version (voiceprints override labels) | 3 | 0 | 8 |
| Label-first redesign | 0 | 12 | 0 |
| + Gemini audio review, if all its suggestions were applied | 9 | 0 | 0 |

The first version failed on REAL speech (14 Nov live session): voiceprints from 0.3–0.7 s clips moved
correctly-labelled Deepgram words to the wrong person, and the label-to-person map rewrote itself, so one error
spread. Synthetic voices hid this, because even short clips of synthetic voices match cleanly.

### The wearer's voice (real voice, band on the wrist)

Enrolled ECAPA baseline, then the wearer talked ~30 s and teammates ~30 s:

| Piece of speech | Wearer typical | Others highest | Best accuracy |
|---|---|---|---|
| 1 s | 0.38 | 0.27 | 94% (equal error 9%) |
| 2 s | 0.47 | 0.31 | 98% (equal error 4%) |
| 3 s | 0.54 | 0.21 | 100% |

- Telling the wearer apart works from about 2 s of speech; 1 s is not reliable.
- The teammates were LOUDER at the band than the wearer (−21.8 dB vs −23.5 dB): "the loudest voice is the
  wearer" is not a safe rule on a wrist.
- Chatting scored lower than reading the enrolment sentences (0.47 vs 0.61): enrol the way you will talk.

### Voice-only tracker (synthetic)

Tuning window and smoothing took moments shown correctly from 41% to 72% (5% wrong, 15% "not sure"), with
the reaction delay going from 2.0 s to 0.9 s. Not measured on real voices.

### diart speaker service (real models, synthetic conversations)

Models: pyannote/segmentation-3.0 (10 s window, 16 kHz, powerset, 589 frames) + speechbrain/spkrec-ecapa-voxceleb.
Startup 8.3 s; 0.185 s of compute per 0.5 s step on a MacBook (real-time factor 0.37).

| Latency setting | Speech missed | Overlap caught |
|---|---|---|
| 0.5 s | 44% | 25 / 26 |
| 2 s | 21–30% | 25 / 26 |
| 3 s | 16% | 24 / 26 |
| 4 s | 8% | 24 / 26 |

- Overlap detection was the strong point (24–25 of 26 overlap moments).
- Wearer shown as someone else: 0. Someone shown as the wearer: 7–9 words, all from two similar female voices
  merged into one speaker; the new-speaker threshold made no difference.
- Most misses were at the start of a turn (56% in a turn's first second vs 17% later): diart averages several
  chunks, and a new speaker is not identified in the early ones.
- Never validated on real band audio before the pivot.

## What did not work, and why

1. **Real voices on a wrist mic are hard.** The same person scored 0.4–0.65 against themselves, different people
   0.0–0.25, versus 0.8 / under 0.5 for synthetic voices. Anything tuned on synthetic audio was optimistic.
2. **Deepgram's live speaker labels break on fast hand-overs**, the exact moments the demo needed.
3. **Short replies cannot be attributed by voice.** Under ~1 s there is not enough sound for a voiceprint.
4. **Overlapping speech loses words** before any attribution happens.
5. **Latency adds up**: ~2.3 s for settled words, plus 3 s for diart to be accurate, plus Gemini.
6. **Every fix moved errors into "unresolved"** rather than making them right. Honest, but a demo of
   "Unclear who" is not a demo.

## What we kept for Familiar

- The enrolled wearer baseline idea (2–3 s of speech separates the wearer well).
- Faces (camera) to say WHO someone is, voice only to say which voice is the wearer, and a name only when exactly
  one registered face is in view.
- Gemini 4 for summaries of what was said (Hydrogen live, Carbon at the end).

## If someone picks this up again

- Record and label real conversations first (`python3 run.py <band> --record ...`, `eval/README.md`) and
  measure before tuning; synthetic audio misled us twice.
- Try the speaker service on real audio with `SPEAKER_LATENCY=3` and calibrate the wearer thresholds on
  recordings that are not used for evaluation (`speaker_service/README.md`).
- Gemini 3.5 Transcribe (recorded-audio diarization) was not available to our project (404); it is wired in
  `familiar/second_pass.py` but never measured.

---

The commit history was put together after the event, grouped by area of work in the order it was built; files
that changed many times appear in their final form.
