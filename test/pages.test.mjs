/* Structural checks on the pages themselves, with no browser.
 *
 * Ported from the Classifiers site's inline_modules test, which exists because
 * a broken module fails in the worst way available: the HTML renders, the prose
 * reads correctly, and only the figures are gone. A screenshot of the text looks
 * perfect. tools/check_pages.py catches this too, but it needs playwright and
 * about forty seconds; these run in one and belong in `npm test`.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, writeFileSync, unlinkSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;

/* The instructor's own demos sit in the working directory as reference and are
   gitignored by the same pattern. They are not ours and not part of the site,
   so they are not held to its conventions. tools/check_pages.py skips them the
   same way. */
const pages = readdirSync(ROOT)
  .filter((f) => f.endsWith('.html') && !/demo/i.test(f));

test('there are pages to check', () => {
  assert.ok(pages.length >= 3, `found ${pages.length} pages`);
});

test('every inline <script type="module"> parses', () => {
  const failures = [];
  for (const page of pages) {
    const html = readFileSync(join(ROOT, page), 'utf8');
    const re = /<script(?![^>]*\bsrc=)[^>]*type=["']module["'][^>]*>([\s\S]*?)<\/script>/g;
    let m, n = 0;
    while ((m = re.exec(html)) !== null) {
      n += 1;
      /* Pad with the preceding newlines so a syntax error's reported line
         number matches the .html file rather than the extracted fragment. */
      const before = html.slice(0, m.index).split('\n').length - 1;
      const tmp = join(tmpdir(), `edpy_${page}_${n}.mjs`);
      writeFileSync(tmp, '\n'.repeat(before) + m[1]);
      try {
        execFileSync(process.execPath, ['--check', tmp], { stdio: 'pipe' });
      } catch (err) {
        const msg = (err.stderr?.toString() || err.message).split('\n').slice(0, 4).join('\n');
        failures.push(`${page} (inline module ${n}):\n${msg}`);
      } finally {
        unlinkSync(tmp);
      }
    }
  }
  assert.deepEqual(failures, []);
});

test('every module file the pages load exists and parses', () => {
  const missing = [], broken = [];
  const srcs = new Set();
  for (const page of pages) {
    const html = readFileSync(join(ROOT, page), 'utf8');
    for (const m of html.matchAll(/<script[^>]*type=["']module["'][^>]*\bsrc=["']([^"']+)["']/g)) {
      srcs.add(m[1]);
    }
  }
  /* The shared modules are not referenced by a <script src>, only imported. */
  for (const f of readdirSync(join(ROOT, 'assets'))) if (f.endsWith('.js')) srcs.add(`assets/${f}`);
  for (const src of srcs) {
    const p = join(ROOT, src);
    if (!existsSync(p)) { missing.push(src); continue; }
    try {
      execFileSync(process.execPath, ['--check', p], { stdio: 'pipe' });
    } catch (err) {
      broken.push(`${src}: ${(err.stderr?.toString() || '').split('\n').slice(0, 3).join(' ')}`);
    }
  }
  assert.deepEqual(missing, [], 'a page loads a module that is not on disk');
  assert.deepEqual(broken, []);
});

test('a page that imports initChrome also calls it', () => {
  /* Importing it and not calling it is silent: the page renders, every figure
     draws, and only the top bar and the theme toggle are dead. */
  const bad = [];
  for (const page of pages) {
    const html = readFileSync(join(ROOT, page), 'utf8');
    const src = (html.match(/<script[^>]*\bsrc=["']([^"']+)["']/) || [])[1];
    const js = src && existsSync(join(ROOT, src)) ? readFileSync(join(ROOT, src), 'utf8') : '';
    const all = html + js;
    if (/\binitChrome\b/.test(all) && !/initChrome\s*\(\s*\)/.test(all)) bad.push(page);
  }
  assert.deepEqual(bad, []);
});

test('every page carries the chrome the shared CSS and JS expect', () => {
  for (const page of pages) {
    const html = readFileSync(join(ROOT, page), 'utf8');
    assert.match(html, /<link[^>]+assets\/site\.css/, `${page}: no stylesheet`);
    assert.match(html, /class="topbar"/, `${page}: no top bar`);
    assert.match(html, /class="topbar__nav"/, `${page}: nav has nowhere to render`);
    assert.match(html, /class="theme-toggle"/, `${page}: no theme toggle`);
    assert.match(html, /<title>[^<]+<\/title>/, `${page}: no title`);
    assert.match(html, /<meta name="description"/, `${page}: no description`);
  }
});

test('every in-page anchor a page links to exists on the page it names', () => {
  const dangling = [];
  for (const page of pages) {
    const html = readFileSync(join(ROOT, page), 'utf8');
    for (const m of html.matchAll(/href="([^"]*#[^"]+)"/g)) {
      const [file, frag] = m[1].split('#');
      const target = file === '' ? page : file;
      if (!existsSync(join(ROOT, target))) { dangling.push(`${page} -> ${m[1]} (no such file)`); continue; }
      const targetHtml = readFileSync(join(ROOT, target), 'utf8');
      if (!targetHtml.includes(`id="${frag}"`)) dangling.push(`${page} -> ${m[1]}`);
    }
  }
  assert.deepEqual(dangling, []);
});

test('every page listed in CHAPTERS exists', () => {
  const js = readFileSync(join(ROOT, 'assets/site.js'), 'utf8');
  const hrefs = [...js.matchAll(/href:\s*'([^']+)'/g)].map((m) => decodeURIComponent(m[1]));
  assert.ok(hrefs.length >= 2);
  for (const h of hrefs) assert.ok(existsSync(join(ROOT, h)), `CHAPTERS lists ${h}, which is not on disk`);
});
