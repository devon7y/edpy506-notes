#!/usr/bin/env python3
"""Load every page in a real browser and fail on anything a reader would notice.

Catches what the unit tests cannot: a script that throws, a figure that renders
empty, a page wider than the screen it is read on, an id collision between a
section anchor and a control. Run it before committing a page change.

    python3 tools/check_pages.py            # all pages, light and dark
    python3 tools/check_pages.py trees.html # just one

Needs playwright with chromium:  pip install playwright && playwright install chromium
"""
import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765
WIDTHS = (390, 768, 1280)


def static_checks(pages):
    """Things that can be caught without a browser."""
    problems = []
    for page in pages:
        html = (ROOT / page).read_text()
        ids = re.findall(r'id="([^"]+)"', html)
        for i in {x for x in ids if ids.count(x) > 1}:
            problems.append(f"{page}: duplicate id {i!r}")
        # A <section id="x"> shadows an <input id="x"> for getElementById, which
        # fails silently and leaves a figure blank. This bit once already.
        sections = set(re.findall(r'<section[^>]*id="([^"]+)"', html))
        for script in ROOT.glob("assets/*-page.js"):
            if script.stem.split("-")[0] not in page:
                continue
            used = set(re.findall(r"getElementById\('([^']+)'\)", script.read_text()))
            for i in used & sections:
                problems.append(f"{page}: script reads id {i!r}, which is also a <section> anchor")
    return problems


async def browser_checks(pages):
    from playwright.async_api import async_playwright

    problems = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for page_name in pages:
            for theme in ("light", "dark"):
                pg = await browser.new_page(viewport={"width": 1280, "height": 900},
                                            color_scheme=theme, reduced_motion="reduce")
                errs = []
                pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
                pg.on("console", lambda m: errs.append(f"console.error: {m.text}")
                      if m.type == "error" else None)
                await pg.goto(f"http://localhost:{PORT}/{page_name}", wait_until="networkidle")
                await pg.wait_for_timeout(1500)

                # drive every control; buttons are re-queried because several
                # figures rebuild their own controls on click
                for h in await pg.query_selector_all("input[type=range]"):
                    lo = await h.get_attribute("min") or "0"
                    hi = await h.get_attribute("max") or "1"
                    for v in (hi, lo, str(round((float(lo) + float(hi)) / 2))):
                        await h.fill(v)
                        await h.dispatch_event("input")
                        await pg.wait_for_timeout(70)
                n = await pg.evaluate("document.querySelectorAll('main button').length")
                for i in range(n):
                    loc = pg.locator("main button").nth(i)
                    if await loc.count():
                        try:
                            await loc.click(timeout=3000, force=True)
                            await pg.wait_for_timeout(120)
                        except Exception as e:
                            errs.append(f"button {i} would not click: {str(e)[:90]}")

                empty = await pg.evaluate(
                    """() => [...document.querySelectorAll('.figure')]
                        .filter(f => !f.querySelector('svg, table, .traits, .card, ul, .bars'))
                        .map(f => (f.querySelector('.figure__title') || {}).textContent)""")
                if empty:
                    errs.append(f"figures rendered empty: {empty}")
                for name, sel in (("select", "select"),):
                    for h in await pg.query_selector_all(sel):
                        opts = await h.query_selector_all("option")
                        if not opts:
                            errs.append(f"{name} has no options")

                for w in WIDTHS:
                    await pg.set_viewport_size({"width": w, "height": 900})
                    await pg.wait_for_timeout(350)
                    sw = await pg.evaluate("document.documentElement.scrollWidth")
                    if sw > w + 1:
                        errs.append(f"page is wider than the viewport at {w}px (scrollWidth {sw})")

                if errs:
                    problems += [f"{page_name} [{theme}]: {e}" for e in errs[:6]]
                else:
                    print(f"  ok   {page_name} [{theme}]")
                await pg.close()
        await browser.close()
    return problems


def main():
    # The instructor's demos are reference material in the working directory,
    # gitignored by the same pattern, and not part of the site.
    pages = sys.argv[1:] or sorted(
        p.name for p in ROOT.glob("*.html") if "demo" not in p.name.lower())
    print(f"checking: {', '.join(pages)}")

    problems = static_checks(pages)
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.2)
        problems += asyncio.run(browser_checks(pages))
    finally:
        server.terminate()

    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nall pages OK")


if __name__ == "__main__":
    main()
