#!/usr/bin/env python3
"""Flag metadiscourse: sentences whose topic is the site rather than the subject.

The six rules are in CLAUDE.md under "Write about the subject, not about the
writing". This catches the mechanical cases. It cannot catch all of them and it
will occasionally be wrong, so a hit is a prompt to reread the sentence, not a
verdict.

Most of the prose on this site is assembled in assets/*-page.js rather than
written in the HTML, so both are scanned.

    python3 tools/check_prose.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RULES = [
    ("narrative pointer",
     r"\bnext section\b|\bas we will see\b|\bfurther down\b|\brecall (?:from|that)\b"
     r"|\bsections? \d+ (?:to|and) \d+\b|\bthe rest of this page\b|\bthe previous page\b"
     r"|\bwill be covered\b|\bwe saw (?:above|earlier)\b|\bas mentioned\b"),
    ("position narration",
     r"(?:^|[.!?]\s+|>|`)(?:Here is|Here are|Now[,\s]|First we|Let us|Having established|Start with)\b"),
    ("self-justification",
     r"\bworth (?:noting|remembering|singling|mentioning|being able|putting|knowing)\b"
     r"|\bit is worth\b|\bimportantly\b|\bnotably\b|\bdeserves (?:stating|mention)\b"
     r"|\bneedless to say\b"),
    ("artifact self-reference",
     r"\bthis (?:page|site|figure|chart|section|demo|document|write-?up)\b"
     r"|\bthe present\b|\bthese notes\b|\bas (?:described|explained) (?:above|below)\b"),
    ("claim of rigour",
     r"\bcarefully (?:measured|chosen|validated)\b|\brigorous(?:ly)?\b|\bgenuinely comput\w*"
     r"|\bhonestly\b|\bproperly measured\b|\brather than merely\b"),
    ("borrowed timeline",
     r"\b(?:findings?|results?|patterns?|coefficients?|assumptions?) (?:emerge|emerges|come back)\b"
     r"|\bthe story (?:so far|of)\b|\bour journey\b"),
]

TARGETS = ["index.html", "regression.html", "trees.html", "classification.html",
           "assets/regression-page.js", "assets/trees-page.js",
           "assets/classification-page.js"]


def prose_of(path: Path) -> str:
    """Strip out everything that is not read by a visitor."""
    t = path.read_text()
    if path.suffix == ".html":
        t = re.sub(r"<script[\s\S]*?</script>", " ", t)
        t = re.sub(r"<style[\s\S]*?</style>", " ", t)
        t = re.sub(r"<!--[\s\S]*?-->", " ", t)
        t = re.sub(r"<[^>]+>", " ", t)
    else:
        # only the template literals in a page script carry prose
        t = re.sub(r"/\*[\s\S]*?\*/", " ", t)
        t = re.sub(r"^\s*//.*$", " ", t, flags=re.M)
    return t


def main():
    hits = []
    for name in TARGETS:
        text = prose_of(ROOT / name)
        for rule, pattern in RULES:
            for m in re.finditer(pattern, text, re.I | re.M):
                start = max(0, m.start() - 60)
                ctx = re.sub(r"\s+", " ", text[start:m.end() + 60]).strip()
                hits.append((name, rule, m.group(0).strip(), ctx))

    if not hits:
        print("prose: clean")
        return
    print(f"prose: {len(hits)} possible instances of metadiscourse\n")
    for name, rule, found, ctx in hits:
        print(f"  {name}  [{rule}]  {found!r}")
        print(f"    …{ctx}…\n")
    print("See CLAUDE.md, \"Write about the subject, not about the writing\".")
    sys.exit(1)


if __name__ == "__main__":
    main()
