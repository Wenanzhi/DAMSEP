/* Asset integrity and real-browser interaction checks. Requires Playwright. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const http = require('node:http');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

const repo = path.resolve(__dirname, '..');
const docs = path.join(repo, 'docs');
const preview = path.join(repo, '.demo-cache/previews');
fs.mkdirSync(preview, {recursive: true});
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(docs, 'data/demo-data.js'), 'utf8'), context);
const data = context.window.DAMSEP_DEMO;
const metadata = new Map(JSON.parse(fs.readFileSync(path.join(repo, 'data/metadata/distance_test.json'))).map(row => [row.utterance_id, row]));

function checkWav(file) {
  const bytes = fs.readFileSync(file);
  assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
  assert.equal(bytes.toString('ascii', 8, 12), 'WAVE');
  let rate, align, frames, peak = 0;
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const type = bytes.toString('ascii', offset, offset + 4);
    const size = bytes.readUInt32LE(offset + 4), start = offset + 8;
    if (type === 'fmt ') {
      assert.equal(bytes.readUInt16LE(start), 1, 'PCM encoding');
      assert.equal(bytes.readUInt16LE(start + 2), 1, 'Mono');
      rate = bytes.readUInt32LE(start + 4); align = bytes.readUInt16LE(start + 12);
      assert.equal(bytes.readUInt16LE(start + 14), 16, 'PCM-16');
    }
    if (type === 'data') {
      frames = size / align;
      for (let i = start; i < start + size; i += 2) peak = Math.max(peak, Math.abs(bytes.readInt16LE(i) / 32768));
    }
    offset = start + size + size % 2;
  }
  assert.equal(rate, 8000); assert.equal(frames, 32000);
  assert(peak > 0 && peak < .941, `Non-silent and unclipped: ${file}`);
}

assert.equal(data.scenes.length, 6);
let audioCount = 0;
for (const sample of data.scenes) {
  assert(metadata.has(sample.utterance));
  assert.equal(JSON.stringify(sample.geometry), JSON.stringify(metadata.get(sample.utterance)));
  for (const source of ['s1', 's2']) {
    const point = sample.geometry[source], mic = sample.geometry.left_mic;
    const distance = Math.hypot(point.x - mic.x, point.y - mic.y, point.z - mic.z);
    assert(Math.abs(distance - point.distance_to_left_mic) < 1e-9);
  }
  assert.equal(Object.keys(sample.audio).length, 11);
  for (const asset of Object.values(sample.audio)) {
    assert(!path.isAbsolute(asset.src)); checkWav(path.join(docs, asset.src)); audioCount++;
  }
  assert.equal(sample.responses.length, 2);
  for (const source of sample.responses) {
    assert.equal(Object.keys(source).length, 7);
    for (const curve of Object.values(source)) {
      assert(Number.isFinite(curve.drr)); assert.equal(curve.early.length, 421);
      assert(curve.early.every(Number.isFinite));
      assert(curve.full.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && Math.abs(y) <= 1.000001));
      assert(curve.edc.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && y >= -60 && y <= 0));
      assert(curve.edc.every((p, i) => !i || p[1] <= curve.edc[i - 1][1] + 1e-5), 'EDC must not increase');
    }
  }
}

const types = {'.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.wav': 'audio/wav', '.md': 'text/plain'};
const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  if (!urlPath.startsWith('/DAMSEP/')) { res.writeHead(404); res.end(); return; }
  const relative = urlPath.slice('/DAMSEP/'.length) || 'index.html';
  const file = path.resolve(docs, relative);
  if (!file.startsWith(docs + path.sep) || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.setHeader('Content-Type', types[path.extname(file)] || 'application/octet-stream');
  const size = fs.statSync(file).size;
  const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '');
  res.setHeader('Accept-Ranges', 'bytes');
  if (range) {
    const start = Number(range[1]), end = range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    if (start >= size || start > end) { res.writeHead(416); res.end(); return; }
    res.statusCode = 206;
    res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
    res.setHeader('Content-Length', end - start + 1);
    fs.createReadStream(file, {start, end}).pipe(res);
  } else {
    res.setHeader('Content-Length', size);
    fs.createReadStream(file).pipe(res);
  }
});

(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}/DAMSEP/`;
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [], failed = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400) failed.push(response.url()); });
    await page.goto(base);
    await page.waitForSelector('html[data-ready="true"]');
    assert.equal(await page.locator('.scene-button').count(), 6);
    assert.equal(await page.locator('audio').count(), 7);

    // Native media controls must load correct durations and preserve seeking.
    await page.waitForFunction(() => [...document.querySelectorAll('audio')].every(a => a.readyState >= 1 && a.duration === 4));
    assert(await page.evaluate(() => [...document.querySelectorAll('audio')].every(a => a.controls)));
    await page.locator('.comparison-cell audio').first().evaluate(audio => { audio.currentTime = 1.25; });
    await page.waitForFunction(() => document.querySelector('.comparison-cell audio').currentTime >= 1.24);
    await page.locator('#mixture-player audio').click({position: {x: 20, y: 21}});
    await page.waitForFunction(() => document.querySelector('#mixture-player audio').currentTime > .05);
    await page.locator('.comparison-cell audio').first().click({position: {x: 20, y: 21}});
    await page.waitForFunction(() => document.querySelector('#mixture-player audio').paused && !document.querySelector('.comparison-cell audio').paused);
    await page.locator('.scene-button').nth(1).click();
    assert(await page.evaluate(() => [...document.querySelectorAll('audio')].every(a => a.paused)));

    for (let i = 0; i < data.scenes.length; i++) {
      await page.locator('.scene-button').nth(i).click();
      assert.equal(await page.locator('#near-distance').innerText(), data.scenes[i].geometry.s1.distance_to_left_mic.toFixed(2));
      assert.equal(await page.locator('#far-distance').innerText(), data.scenes[i].geometry.s2.distance_to_left_mic.toFixed(2));
      assert((await page.locator('#scene-id').innerText()).includes(data.scenes[i].utterance));
      await page.locator('[data-audio-mode="reverberant"]').click();
      assert.equal(await page.locator('audio').count(), 5);
      assert.equal(await page.locator('.audio-table tbody tr').count(), 2);
      await page.locator('[data-audio-mode="clean"]').click();
      assert.equal(await page.locator('audio').count(), 7);
      assert.equal(await page.locator('.audio-table tbody tr').count(), 3);
      for (const method of Object.keys(data.methods)) {
        await page.selectOption('#baseline-select', method);
        assert.equal(await page.locator('#baseline-legend').innerText(), data.methods[method]);
        for (const source of [0, 1]) {
          await page.locator(`[data-source="${source}"]`).click();
          assert((await page.locator('#rir-plot svg').getAttribute('aria-label')).includes(source ? 'far' : 'near'));
          assert.equal(await page.locator('#rir-plot path').count(), 3);
          assert.equal(await page.locator('#edc-plot path').count(), 3);
          assert.equal(await page.locator('.drr-results tbody tr').count(), 3);
        }
      }
    }
    await page.locator('[data-window="1000"]').click();
    assert.equal(await page.locator('[data-window="1000"]').getAttribute('aria-pressed'), 'true');
    assert((await page.locator('#rir-plot').innerText()).includes('1000'));
    await page.locator('[data-window="50"]').click();
    assert(await page.locator('.architecture img').isVisible());
    assert(await page.locator('.architecture img').evaluate(img => img.complete && img.naturalWidth > 0));
    await page.locator('.protocol summary').click();
    assert(await page.locator('.protocol strong').filter({hasText: 'ground-truth direct-path RIR'}).isVisible());
    await page.locator('.protocol summary').click();

    await page.locator('.scene-button').first().click();
    await page.selectOption('#baseline-select', 'recrir');
    await page.locator('[data-source="0"]').click();
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForFunction(() => [...document.querySelectorAll('audio')].every(a => a.readyState >= 1));
    await page.screenshot({path: path.join(preview, 'desktop-top.png')});
    await page.screenshot({path: path.join(preview, 'desktop.png'), fullPage: true});
    for (const width of [768, 390, 320]) {
      await page.setViewportSize({width, height: 844});
      await page.waitForTimeout(100);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No overflow at ${width}px`);
      if (width === 390) await page.screenshot({path: path.join(preview, 'mobile.png'), fullPage: true});
    }
    assert.deepEqual(errors, []); assert.deepEqual(failed, []);

    // The folder is also a portable offline demo, not just a hosted page.
    await page.goto(pathToFileURL(path.join(docs, 'index.html')).href);
    await page.waitForSelector('html[data-ready="true"]');
    await page.locator('#mixture-player audio').click({position: {x: 20, y: 21}});
    await page.waitForFunction(() => document.querySelector('#mixture-player audio').currentTime > .05);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({scenes: 6, audioFiles: audioCount, rirSourceMethodPairs: 84,
      baselineSourceSwitches: 60, nativePlayback: 'passed', nativeSeek: 'passed', exclusivePlayback: 'passed',
      responsiveWidths: [1440, 768, 390, 320], projectSubpath: 'passed', offlinePlayback: 'passed',
      browserErrors: errors.length, assetErrors: failed.length, previews: preview}, null, 2));
  } finally { await browser.close(); server.close(); }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });
