"""One-off check: Gemini API reachable with project .env. Delete after use."""
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY set:", bool(key and str(key).strip()))
if not key or not str(key).strip():
    print("RESULT: FAIL — no GEMINI_API_KEY in .env")
    raise SystemExit(1)

import google.generativeai as genai

genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-1.5-flash")
r = model.generate_content('Reply with exactly one word: OK')
text = (r.text or "").strip()
print("Response:", repr(text[:300]))
print("RESULT: OK" if text else "RESULT: FAIL — empty response")
