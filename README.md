# Vexo Recall — who is talking, in the moment

Hackathon demo. The band's mic goes to Deepgram with speaker labels (diarization). Names come from what people say
("Hi, I'm Maya", "Nice to meet you, Dev", "What's your name?" → "Priya"). Anything unsure is marked "possible";
anything missing can be typed in on the page. No names or voices are stored.

## Run it
```
git clone https://github.com/VexoAI-Org/vexo-open-kit.git      # next to run.py (git-ignored)
echo "DEEPGRAM_API_KEY=your-team-key" > .env                   # git-ignored; never commit or paste the key
python3 run.py --sim          # scripted demo, no band or key: the safe fallback
python3 run.py --type         # type lines yourself:  1: Hi, I'm Maya
python3 run.py 42             # live: your band number (double-tap the band first, close the kit dashboard tab)
python3 run.py 42 --flash     # also blink the band green on a new name (untested on hardware)
python3 -m unittest discover -s tests
```
Page: http://127.0.0.1:8765 (local only).

## Things to know
- The band has no vibration or screen, only a light. It also refuses light commands while its mic is on, so
  the page is the main feedback; `--flash` pauses the mic for 0.5 s to blink.
- Diarization labels voices only within one stream and can split one person in two: use "Same as…" on the page.
- Tell people near the band the mic is on; the audio goes to Deepgram.
- `familiar/resolver.py` is the logic (name rules, tested); `familiar/live.py` adds `diarize=true` to the kit's Deepgram class.
