/* No network requests: the small exported dataset also works from file://. */
(() => {
  'use strict';
  const data = window.DAMSEP_DEMO;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const colors = {reference: '#333333', damsep: '#1f77b4', baseline: '#d97926', near: '#1f77b4', far: '#d97926'};
  const state = {scene: 0, audioMode: 'clean', source: 0, baseline: 'recrir', window: 50};
  let players = [];
  const cleanupPlayers = [];
  const NS = 'http://www.w3.org/2000/svg';
  let svgCounter = 0;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svg(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svgText(x, y, text, attrs = {}) {
    return svg('text', {x, y, fill: '#666666', 'font-size': 11,
      'font-family': '-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif', ...attrs}, text);
  }

  function announce(message) { $('#announcer').textContent = message; }
  function scene() { return data.scenes[state.scene]; }
  function pressGroup(selector, value, key) {
    $$(selector).forEach(button => button.setAttribute('aria-pressed', String(button.dataset[key] === String(value))));
  }

  function drawGeometry() {
    const geometry = scene().geometry;
    const mic = geometry.left_mic;
    const points = [geometry.s1, geometry.s2].map((source, i) => ({
      x: source.x - mic.x, y: source.y - mic.y, z: source.z - mic.z,
      color: i ? colors.far : colors.near, label: i ? 'Far source' : 'Near source', source
    }));
    const w = Math.max(300, Math.min(610, $('#spatial-plot').clientWidth));
    const h = 306, pad = w < 450 ? 40 : 53;
    const xs = [0, ...points.map(p => p.x)], ys = [0, ...points.map(p => p.y)];
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const scale = Math.min((w - 2 * pad) / Math.max(3, maxX - minX + 1),
      (h - 2 * pad) / Math.max(3, maxY - minY + 1));
    const centerX = (minX + maxX) / 2, centerY = (minY + maxY) / 2;
    const x = value => w / 2 + (value - centerX) * scale;
    const y = value => h / 2 - (value - centerY) * scale;
    const chart = svg('svg', {viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-labelledby': 'geometry-title geometry-description'});
    chart.append(svg('title', {id: 'geometry-title'}, 'Ground-truth microphone and source positions'));
    chart.append(svg('desc', {id: 'geometry-description'}, `XY projection relative to the microphone. Near source: ${geometry.s1.distance_to_left_mic.toFixed(2)} metres in 3D. Far source: ${geometry.s2.distance_to_left_mic.toFixed(2)} metres in 3D. Source heights are listed beside the diagram.`));
    const grid = svg('g', {stroke: '#e6e6e6', 'stroke-width': 1});
    const boundsX = [centerX - (w / 2 - 22) / scale, centerX + (w / 2 - 22) / scale];
    const boundsY = [centerY - (h / 2 - 20) / scale, centerY + (h / 2 - 20) / scale];
    for (let i = Math.ceil(boundsX[0]); i <= Math.floor(boundsX[1]); i++) {
      grid.append(svg('line', {x1: x(i), x2: x(i), y1: 22, y2: h - 25}));
      if (i !== 0) chart.append(svgText(x(i), h - 8, i, {'text-anchor': 'middle', 'font-size': 10, fill: '#888888'}));
    }
    for (let i = Math.ceil(boundsY[0]); i <= Math.floor(boundsY[1]); i++) {
      grid.append(svg('line', {x1: 25, x2: w - 25, y1: y(i), y2: y(i)}));
      if (i !== 0) chart.append(svgText(11, y(i) + 3, i, {'text-anchor': 'middle', 'font-size': 10, fill: '#888888'}));
    }
    chart.prepend(grid);
    chart.append(svg('line', {x1: 25, x2: w - 25, y1: y(0), y2: y(0), stroke: '#bbbbbb', 'stroke-dasharray': '3 5'}));
    chart.append(svg('line', {x1: x(0), x2: x(0), y1: 22, y2: h - 25, stroke: '#bbbbbb', 'stroke-dasharray': '3 5'}));
    chart.append(svgText(w - 17, h - 8, 'x', {'font-size': 10, fill: '#555555'}));
    chart.append(svgText(10, 13, 'y', {'font-size': 10, fill: '#555555'}));
    points.forEach((p, i) => {
      chart.append(svg('line', {x1: x(0), y1: y(0), x2: x(p.x), y2: y(p.y), stroke: p.color, 'stroke-width': 1.6, 'stroke-dasharray': '5 5', opacity: .7}));
      const point = svg('g');
      point.append(svg('circle', {cx: x(p.x), cy: y(p.y), r: 13, fill: p.color}));
      point.append(svgText(x(p.x), y(p.y) + 4, i + 1, {'text-anchor': 'middle', fill: 'white', 'font-size': 11, 'font-weight': 600}));
      const labelY = y(p.y) + (p.y >= 0 ? -33 : 37);
      point.append(svgText(x(p.x), labelY, p.label, {'text-anchor': 'middle', fill: p.color, 'font-size': 10, 'font-weight': 550}));
      point.append(svg('title', {}, `${p.label}: x ${p.source.x.toFixed(3)}, y ${p.source.y.toFixed(3)}, z ${p.source.z.toFixed(3)} m`));
      chart.append(point);
    });
    chart.append(svg('circle', {cx: x(0), cy: y(0), r: 9, fill: '#333333'}));
    chart.append(svg('circle', {cx: x(0), cy: y(0), r: 3, fill: 'white'}));
    // Put the microphone label on the less crowded side of its two connections.
    const labelAbove = points.reduce((sum, p) => sum + p.y, 0) < 0;
    chart.append(svgText(x(0), y(0) + (labelAbove ? -27 : 30), 'MIC', {'text-anchor': 'middle', fill: '#444444', 'font-size': 10, 'font-weight': 600, 'letter-spacing': 1}));
    $('#spatial-plot').replaceChildren(chart);
    const rows = $('#geometry-rows'); rows.replaceChildren();
    [['Microphone', mic, null], ['Source 1 (near)', geometry.s1, 'near'], ['Source 2 (far)', geometry.s2, 'far']].forEach(([label, point, role]) => {
      const row = el('tr');
      const name = el('th', role || '', label); name.scope = 'row'; row.append(name);
      ['x', 'y', 'z'].forEach(axis => row.append(el('td', '', point[axis].toFixed(2))));
      const distance = el('td', '', role ? point.distance_to_left_mic.toFixed(2) : '—');
      if (role) distance.id = role + '-distance';
      row.append(distance); rows.append(row);
    });
    $('#scene-id').textContent = `Sample ${state.scene + 1}: ${scene().utterance}`;
  }

  function makePlayer(asset, label) {
    const wrapper = el('div', 'audio-player');
    const audio = document.createElement('audio');
    audio.src = asset.src;
    audio.controls = true;
    audio.preload = 'metadata';
    audio.setAttribute('aria-label', label);
    const onPlay = () => players.forEach(other => { if (other !== audio) other.pause(); });
    const onError = () => {
      if (!wrapper.querySelector('.audio-error')) wrapper.append(el('p', 'audio-error', 'Audio unavailable.'));
      announce(`Audio unavailable: ${label}. Check the audio folder.`);
    };
    audio.addEventListener('play', onPlay);
    audio.addEventListener('error', onError);
    wrapper.append(audio);
    players.push(audio);
    cleanupPlayers.push(() => {
      audio.removeEventListener('error', onError);
      audio.removeEventListener('play', onPlay);
      audio.pause(); audio.removeAttribute('src'); audio.load();
    });
    return wrapper;
  }

  function drawAudio() {
    cleanupPlayers.splice(0).forEach(fn => fn());
    players = [];
    const current = scene();
    $('#mixture-player').replaceChildren(makePlayer(current.audio.mixture, 'Input mixture'));
    const table = el('table', 'audio-table');
    const thead = el('thead'); const headings = el('tr');
    const methodHeading = el('th', '', 'Method'); methodHeading.scope = 'col'; headings.append(methodHeading);
    ['Near source', 'Far source'].forEach((label, i) => {
      const heading = el('th', '', label); heading.scope = 'col';
      heading.append(el('small', '', current.geometry['s' + (i + 1)].distance_to_left_mic.toFixed(2) + ' m'));
      headings.append(heading);
    });
    thead.append(headings); table.append(thead);
    const tbody = el('tbody');
    const methods = [['reference', 'Reference', 'Ground truth'], ['damsep', 'DAMSEP', 'Ours']];
    if (state.audioMode === 'clean') methods.push(['spmamba', 'SPMamba', 'Baseline']);
    methods.forEach(([key, label, subtitle]) => {
      const row = el('tr', key === 'damsep' ? 'method-ours' : '');
      const name = el('th', '', label); name.scope = 'row'; name.append(el('small', '', subtitle)); row.append(name);
      [0, 1].forEach(source => {
        const cell = el('td', 'comparison-cell');
        cell.dataset.label = source ? 'Far source' : 'Near source';
        cell.append(makePlayer(current.audio[`${key}_${state.audioMode}_${source}`],
          `${label}, ${source ? 'far' : 'near'} ${state.audioMode} source`));
        row.append(cell);
      });
      tbody.append(row);
    });
    table.append(tbody); $('#audio-comparison').replaceChildren(table);
    $('#audio-note').textContent = state.audioMode === 'clean'
      ? 'Clean-source estimates are compared with anechoic references. All clips use the same crop and a shared playback gain per sample. Headphones are recommended.'
      : 'DAMSEP’s reverberant source estimates are compared with the reverberant references. All clips use the same crop and a shared playback gain per sample. Headphones are recommended.';
  }

  function plot(container, config) {
    const w = Math.max(290, Math.min(480, $(container).clientWidth)), h = 247;
    const margin = {left: 49, right: 15, top: 15, bottom: 39};
    const x = v => margin.left + (v - config.xmin) / (config.xmax - config.xmin) * (w - margin.left - margin.right);
    const y = v => h - margin.bottom - (v - config.ymin) / (config.ymax - config.ymin) * (h - margin.top - margin.bottom);
    const root = svg('svg', {viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': config.label});
    root.append(svg('title', {}, config.label));
    const clipId = 'plot-clip-' + (++svgCounter);
    const clip = svg('clipPath', {id: clipId});
    clip.append(svg('rect', {x: margin.left, y: margin.top, width: w - margin.left - margin.right, height: h - margin.top - margin.bottom}));
    const defs = svg('defs'); defs.append(clip); root.append(defs);
    config.yticks.forEach(v => {
      root.append(svg('line', {x1: margin.left, x2: w - margin.right, y1: y(v), y2: y(v), stroke: '#e6e6e6', 'stroke-width': 1}));
      root.append(svgText(margin.left - 10, y(v) + 3, v, {'text-anchor': 'end', 'font-size': 10}));
    });
    config.xticks.forEach(v => {
      root.append(svg('line', {x1: x(v), x2: x(v), y1: margin.top, y2: h - margin.bottom, stroke: '#ededed', 'stroke-width': 1}));
      root.append(svgText(x(v), h - margin.bottom + 16, v, {'text-anchor': 'middle', 'font-size': 10}));
    });
    root.append(svgText((w + margin.left - margin.right) / 2, h - 5, 'Time after direct arrival (ms)', {'text-anchor': 'middle', 'font-size': 10}));
    root.append(svgText(12, h / 2, config.ylabel, {transform: `rotate(-90 12 ${h / 2})`, 'text-anchor': 'middle', 'font-size': 10}));
    const curves = svg('g', {'clip-path': `url(#${clipId})`});
    // Draw the reference last so early-arrival agreement remains visible.
    [...config.lines].reverse().forEach(line => {
      const path = line.points.map(([tx, value], i) => `${i ? 'L' : 'M'}${x(tx).toFixed(2)},${y(value).toFixed(2)}`).join(' ');
      const attrs = {d: path, fill: 'none', stroke: line.color, 'stroke-width': line.key === 'reference' ? 1.3 : 1.6, 'stroke-linejoin': 'round', opacity: line.key === 'reference' ? .8 : .88};
      if (line.key === 'reference') attrs['stroke-dasharray'] = '4 3';
      const curve = svg('path', attrs); curve.append(svg('title', {}, line.label)); curves.append(curve);
    });
    root.append(curves);
    $(container).replaceChildren(root);
  }

  function drawResponses() {
    const responses = scene().responses[state.source];
    const selected = [
      {key: 'reference', label: 'Ground truth', color: colors.reference},
      {key: 'damsep', label: 'DAMSEP', color: colors.damsep},
      {key: state.baseline, label: data.methods[state.baseline], color: colors.baseline}
    ];
    $('#baseline-legend').textContent = data.methods[state.baseline];
    const short = state.window === 50;
    plot('#rir-plot', {label: `Peak-normalized impulse responses for the ${state.source ? 'far' : 'near'} source: ground truth, DAMSEP, ${data.methods[state.baseline]}`,
      xmin: -2.5, xmax: state.window, ymin: -1.05, ymax: 1.05, ylabel: 'Amplitude',
      yticks: [-1, -.5, 0, .5, 1], xticks: short ? [0, 10, 20, 30, 40, 50] : [0, 200, 400, 600, 800, 1000],
      lines: selected.map(line => ({...line, points: short
        ? responses[line.key].early.map((v, i) => [(i - 20) / 8, v]) : responses[line.key].full}))});
    plot('#edc-plot', {label: `Energy decay curves for the ${state.source ? 'far' : 'near'} source: ground truth, DAMSEP, ${data.methods[state.baseline]}`,
      xmin: 0, xmax: 1000, ymin: -60, ymax: 0, ylabel: 'Energy (dB)', yticks: [0, -15, -30, -45, -60],
      xticks: [0, 200, 400, 600, 800, 1000], lines: selected.map(line => ({...line, points: responses[line.key].edc}))});
    const table = el('table', 'drr-results');
    const thead = el('thead'); const heading = el('tr');
    ['Method', 'Near DRR (dB)', 'Far DRR (dB)', 'Inferred nearer source'].forEach(label => {
      const cell = el('th', '', label); cell.scope = 'col'; heading.append(cell);
    });
    thead.append(heading); table.append(thead);
    const tbody = el('tbody');
    selected.forEach(method => {
      const near = scene().responses[0][method.key].drr;
      const far = scene().responses[1][method.key].drr;
      const tied = Math.abs(near - far) <= 1e-6;
      const agrees = near > far && !tied;
      const row = el('tr', method.key === 'damsep' ? 'ours' : '');
      const name = el('th'); name.scope = 'row';
      const methodLabel = el('span', 'method-cell');
      const line = el('i', 'line'); line.style.color = method.color;
      if (method.key === 'reference') line.style.borderTopStyle = 'dashed';
      methodLabel.append(line, document.createTextNode(method.label)); name.append(methodLabel);
      const result = el('td', 'order-result' + (agrees ? '' : ' disagrees'), tied ? 'Tie' : agrees ? 'Source 1' : 'Source 2');
      result.append(el('small', '', tied ? 'No order' : agrees ? 'Matches geometry' : 'Differs from geometry'));
      row.append(name, el('td', '', near.toFixed(2)), el('td', '', far.toFixed(2)), result);
      tbody.append(row);
    });
    table.append(tbody); $('#drr-table').replaceChildren(table);
  }

  function selectScene(index, notify = true) {
    state.scene = index;
    pressGroup('.scene-button', index, 'scene');
    drawGeometry(); drawAudio(); drawResponses();
    if (notify) announce(`Sample ${index + 1} selected. Geometry, separation audio and RIR comparisons updated.`);
  }

  try {
    if (!data || !Array.isArray(data.scenes) || data.scenes.length !== 6 || !data.scenes.every(s => s.responses?.length === 2)) {
      throw new Error('Demo data are missing or incomplete.');
    }
    const picker = $('#scene-picker');
    data.scenes.forEach((sample, index) => {
      const button = el('button', 'scene-button'); button.type = 'button'; button.dataset.scene = index;
      const near = sample.geometry.s1.distance_to_left_mic.toFixed(2);
      const far = sample.geometry.s2.distance_to_left_mic.toFixed(2);
      button.setAttribute('aria-label', `Sample ${index + 1}: near ${near} metres, far ${far} metres`);
      button.textContent = `Sample ${index + 1}`;
      button.addEventListener('click', () => selectScene(index)); picker.append(button);
    });
    Object.entries(data.methods).forEach(([value, label]) => {
      const option = el('option', '', label); option.value = value; $('#baseline-select').append(option);
    });
    $$('[data-audio-mode]').forEach(button => button.addEventListener('click', () => {
      state.audioMode = button.dataset.audioMode;
      pressGroup('[data-audio-mode]', state.audioMode, 'audioMode'); drawAudio();
      announce(`Showing ${state.audioMode} source comparisons.`);
    }));
    $$('[data-source]').forEach(button => button.addEventListener('click', () => {
      state.source = Number(button.dataset.source); pressGroup('[data-source]', state.source, 'source'); drawResponses();
      announce(`Showing ${state.source ? 'far' : 'near'} source room responses.`);
    }));
    $$('[data-window]').forEach(button => button.addEventListener('click', () => {
      state.window = Number(button.dataset.window); pressGroup('[data-window]', state.window, 'window'); drawResponses();
    }));
    $('#baseline-select').addEventListener('change', event => {
      state.baseline = event.target.value; drawResponses();
      announce(`Comparing room responses with ${data.methods[state.baseline]}.`);
    });
    $('#scene-layout').hidden = false;
    selectScene(0, false);
    $('#loading-message').hidden = true;
    let resizeFrame;
    window.addEventListener('resize', () => {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => { drawGeometry(); drawResponses(); });
    });
    document.documentElement.dataset.ready = 'true';
  } catch (error) {
    console.error(error);
    const message = $('#loading-message'); message.className = 'error-message';
    message.textContent = 'The demo assets could not be loaded. Keep index.html, app.js and the data and audio folders together, then reload the page.';
  }
})();
