"""Offline checks for the physics/chemistry/math and math/science-figure sources.

The rows below are the real field shapes pulled from the Hub datasets-server, so this catches the
mistake that actually bites here: pointing `_txt` at a dataset that has no `text` column. `_stream`
opens lazily, so such a source does not fail at import - it just yields nothing, forever, and shows
up as a modality quietly missing from the mix.
"""
import sys

from omni.producer import GROUPS, SOURCES, _img_text, _scienceqa, _two, _txt

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


# camel-ai/physics, camel-ai/chemistry, camel-ai/math: role_1, topic;, sub_topic, message_1, message_2
CAMEL = {"role_1": "Physicist_RoleType.ASSISTANT", "topic;": "Quantum mechanics",
         "sub_topic": "The Schrodinger equation", "message_1": "What is the probability of ...",
         "message_2": "To find the probability we first normalise ..."}

fmt = _two("message_1", "message_2")
got = fmt(CAMEL)
check("camel-ai rows produce text via _two", got.get("text", "").startswith("What is the probability"))
check("_two joins question and answer", "\n\n" in got["text"] and "normalise" in got["text"])
check("_txt would have yielded nothing on the same row (why _two is required)", _txt(CAMEL) == {})
check("_two survives a missing field", fmt({"message_1": "only a question"}) == {"text": "only a question"})
check("_two drops an empty row", fmt({}) == {})

# linxy/LaTeX_OCR: image, text  (the text is a LaTeX transcription)
LATEX = {"image": {"src": "x.png"}, "text": "d s ^ { 2 } = ( 1 - \\frac { q } { r } ) ^ { 2 }"}
lt = _img_text(img_keys=("image",), cap_keys=("text",))(LATEX)
check("LaTeX_OCR keeps the image", lt.get("image") is not None)
check("LaTeX_OCR keeps the LaTeX transcription", "\\frac" in lt.get("text", ""))
check("LaTeX_OCR drops a row with no image",
      _img_text(img_keys=("image",), cap_keys=("text",))({"text": "no image here"}) == {})

# derek-thomas/ScienceQA: image, question, choices, answer(index), ...
SQ = {"image": {"src": "x.png"}, "question": "Which of these states is farthest north?",
      "choices": ["West Virginia", "Louisiana", "Arizona", "Oklahoma"], "answer": 0,
      "hint": "", "task": "closed choice", "grade": "grade2", "subject": "social science",
      "solution": "Look at the compass rose ..."}
r = _scienceqa(SQ)
check("ScienceQA keeps the image", r.get("image") is not None)
check("ScienceQA renders the options", "A. West Virginia" in r.get("text", ""), r.get("text", "")[:60])
check("ScienceQA resolves the index answer to its text", "Answer: West Virginia" in r.get("text", ""))
check("ScienceQA tolerates an out-of-range answer index",
      "Answer:" not in _scienceqa(dict(SQ, answer=99)).get("text", ""))
check("ScienceQA falls back to the solution when there is no question",
      "compass rose" in _scienceqa(dict(SQ, question="", choices=[], answer=None)).get("text", ""))
check("ScienceQA drops a row with no image", _scienceqa({"question": "q", "choices": ["a"]}) == {})

# Wiring: every new source exists, is grouped, and uses the formatter it should.
check("text sources registered",
      all(k in SOURCES for k in ("text_physics", "text_chemistry", "text_math_inst")))
check("image sources registered", all(k in SOURCES for k in ("image_latex", "image_scienceqa")))
check("physics/chemistry/math are the text group",
      all(GROUPS[k] == "text" for k in ("text_physics", "text_chemistry", "text_math_inst")))
check("latex/scienceqa are the image group",
      all(GROUPS[k] == "image" for k in ("image_latex", "image_scienceqa")))
check("every SOURCE has exactly one GROUP", all(k in GROUPS for k in SOURCES),
      str([k for k in SOURCES if k not in GROUPS]) or "none missing")

print("\n" + ("ALL DOMAIN-SOURCE CHECKS PASS" if ok else "DOMAIN-SOURCE CHECKS FAILED"))
sys.exit(0 if ok else 1)
