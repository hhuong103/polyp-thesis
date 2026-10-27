// Exercise the actual frontend event handlers without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.childNodes = []; this.value = ''; this._text = '';
    this.files = []; this.hidden = false; this.disabled = false;
    this.style = {};
    this.handlers = {}; this.classList = {toggle() {}};
  }
  set textContent(value) { this._text = String(value); this.childNodes = []; }
  get textContent() { return this._text + this.childNodes.map(c => c.textContent).join(''); }
  set value(value) { this._value = value; if (this.tagName === 'input' && value === '') this.files = []; }
  get value() { return this._value; }
  append(...values) {
    for (const value of values) {
      if (typeof value === 'string') { const n = new Element('#text'); n.textContent = value; this.childNodes.push(n); }
      else this.childNodes.push(value);
    }
  }
  replaceChildren(...values) { this._text = ''; this.childNodes = []; this.append(...values); }
  addEventListener(type, handler) { (this.handlers[type] ||= []).push(handler); }
  setAttribute(key, value) { this[key] = value; }
  scrollIntoView() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
}
class Form {
  constructor() { this.values = new Map(); }
  append(key, value) { this.values.set(key, value); }
  get(key) { return this.values.get(key); }
}

const nodes = {};
const html = fs.readFileSync('frontend/index.html', 'utf8');
for (const match of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
  nodes[match[2]] = new Element(match[1]);
  nodes[match[2]].hidden = /\bhidden\b/.test(match[0]);
}
nodes['prediction-mode'].value = 'single';
const calls = [];
let deferred = null, failNext = false;
const models = [
  {id: 'final-unet', name: 'U-Net', available: true},
  {id: 'unetplusplus', name: 'U-Net++', available: true},
  {id: 'smoke-unet', name: 'Smoke', available: true, evaluation_enabled: false},
];
function prediction(model) {
  return {id: model, name: model, mask: 'mask', overlay: 'overlay', inference_ms: 10,
    processing_ms: 12, foreground_pixels: 30, foreground_percent: 3, empty_mask: false,
    parameter_count: 123456, architecture: model === 'final-unet' ? 'unet' : 'unetplusplus',
    encoder: 'resnet34', checkpoint: model + '.pth', metrics: null};
}
function payload(url, form) {
  const data = {original: 'original', total_ms: 25, results: url.endsWith('/compare') || url.endsWith('/evaluate-image')
    ? [prediction(form.get('model_a')), prediction(form.get('model_b'))]
    : [prediction(form.get('model'))]};
  if (form.get('ground_truth')) {
    data.ground_truth = 'truth';
    data.results.forEach(r => { r.metrics = {dice: 0.8, iou: 2/3, false_positive_pixels: 3, false_negative_pixels: 2}; });
    if (form.get('resize_ground_truth') === 'true') {
      const truth = form.get('ground_truth'), image = form.get('file');
      data.ground_truth_alignment = {resized: true, method: 'nearest',
        original_width: truth.width, original_height: truth.height,
        target_width: image.width || 640, target_height: image.height || 480};
    }
  }
  return data;
}
const context = vm.createContext({
  Node: Element, FormData: Form, console,
  createImageBitmap: async file => {
    if (file.invalid) throw new Error('Invalid image');
    return {width: file.width || 640, height: file.height || 480, close() {}};
  },
  document: {hidden: false, getElementById: id => nodes[id], querySelectorAll: () => [],
    createElement: tag => new Element(tag), createTextNode: text => { const n = new Element('#text'); n.textContent = text; return n; }},
  setInterval() {}, clearTimeout() {}, setTimeout() {},
  fetch: async (url, options) => {
    if (url === '/api/study') return {ok: true, json: async () => ({status: {phase: 'not_started'}, summary: null})};
    if (url === '/api/models') return {ok: true, json: async () => ({device: 'cpu', models})};
    if (url === '/api/evaluation') return {ok: true, json: async () => ({job: {status: 'idle'}, report: null})};
    calls.push({url, form: options.body});
    if (failNext) {
      const failure = failNext; failNext = false;
      return {ok: false, status: failure.status || 503, json: async () => ({detail: failure.detail || 'Checkpoint unavailable'})};
    }
    if (deferred) { const wait = deferred; deferred = null; await wait; }
    return {ok: true, json: async () => payload(url, options.body)};
  },
});
function change(id, value) {
  if (value !== undefined) nodes[id].value = value;
  for (const handler of nodes[id].handlers.change || []) handler();
}
function submit() { return nodes['compare-form'].onsubmit({preventDefault() {}}); }
function descendants(node) { return node.childNodes.flatMap(child => [child, ...descendants(child)]); }
function sections(node) { return descendants(node).filter(child => child.className === 'model-result'); }

async function run() {
  vm.runInContext(fs.readFileSync('frontend/app.js', 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(nodes['model-a'].childNodes.length, 2, 'Smoke checkpoint must not appear');
  assert.equal(nodes['second-model'].hidden, true);
  nodes['image-file'].files = [{name: 'test.jpg', size: 100}];

  for (const model of ['final-unet', 'unetplusplus']) {
    change('model-a', model);
    await submit();
    assert.equal(calls.at(-1).url, '/api/predict');
    assert.equal(calls.at(-1).form.get('model'), model);
    assert.equal(calls.at(-1).form.get('model_b'), undefined);
    assert.equal(sections(nodes.results).length, 1);
    assert.ok(nodes.results.textContent.includes('Prediction Mask'));
  }
  change('prediction-mode', 'compare');
  assert.equal(nodes['second-model'].hidden, false);
  assert.notEqual(nodes['model-a'].value, nodes['model-b'].value);
  await submit();
  assert.equal(calls.at(-1).url, '/api/compare');
  assert.equal(sections(nodes.results).length, 2);
  assert.ok(nodes.metrics.textContent.includes('Cần Ground Truth'));
  assert.ok(nodes.metrics.textContent.includes('Số tham số'));
  assert.ok(nodes.metrics.textContent.includes('123.456'));

  // Evaluation is a separate workflow and never silently runs without a mask.
  change('prediction-mode', 'evaluation');
  assert.equal(nodes['truth-file'].required, true);
  assert.equal(nodes['truth-input'].open, true);
  assert.equal(nodes['predict-button'].disabled, true);
  assert.equal(nodes.reports.hidden, false);
  const beforeEvaluation = calls.length;
  await submit();
  assert.equal(calls.length, beforeEvaluation);
  assert.ok(nodes.message.textContent.includes('Ground Truth'));
  nodes['truth-file'].files = [{name: 'mask.png', size: 80}];
  change('truth-file');
  assert.equal(nodes['predict-button'].disabled, false);
  await submit();
  assert.equal(calls.at(-1).url, '/api/evaluate-image');
  assert.equal(calls.at(-1).form.get('ground_truth').name, 'mask.png');
  assert.equal(sections(nodes.results).length, 2);
  assert.ok(nodes.metrics.textContent.includes('0.8000'));
  assert.ok(nodes.results.textContent.includes('Ground Truth'));

  // Exercise image controls, including independent overlays for each model.
  const stages = descendants(nodes.results).filter(n => n.className === 'before-after-stage');
  const controls = descendants(nodes.results).filter(n => n.className === 'viewer-controls');
  const slider = descendants(controls[0]).find(n => n.type === 'range');
  slider.value = '75'; slider.oninput();
  assert.equal(stages[0].childNodes[1].style.clipPath, 'inset(0 0 0 75%)');
  const toggle = descendants(controls[0]).find(n => n.type === 'checkbox');
  toggle.checked = false; toggle.onchange();
  assert.equal(stages[0].childNodes[1].hidden, true);
  assert.equal(slider.disabled, true);
  assert.equal(stages[1].childNodes[1].hidden, false);
  descendants(nodes.results).find(n => n.className === 'image-view').onclick();
  assert.equal(nodes['image-dialog'].open, true);
  nodes['zoom-level'].value = '200'; nodes['zoom-level'].oninput();
  assert.equal(nodes['zoom-image'].style.width, '200%');
  nodes['close-zoom'].onclick();
  assert.equal(nodes['image-dialog'].open, false);

  // Reject mismatched uploads before making an HTTP request; clearing GT restores prediction.
  nodes['truth-file'].files = [{name: 'small-mask.png', size: 80, width: 2, height: 2}];
  change('truth-file');
  const beforeMismatch = calls.length;
  await submit();
  assert.equal(calls.length, beforeMismatch);
  assert.ok(nodes.message.textContent.includes('640 × 480'));
  assert.ok(nodes.message.textContent.includes('2 × 2'));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(nodes['resize-tools'].hidden, false);
  nodes['resize-truth'].onclick();
  await submit();
  assert.equal(calls.at(-1).url, '/api/evaluate-image');
  assert.equal(calls.at(-1).form.get('resize_ground_truth'), 'true');
  assert.ok(nodes.message.textContent.includes('Đã căn chỉnh Ground Truth'));
  assert.ok(nodes.results.textContent.includes('nearest-neighbor'));
  assert.ok(nodes.metrics.textContent.includes('Đã căn chỉnh kích thước'));
  nodes['reset-resize'].onclick();
  const beforeReset = calls.length;
  await submit();
  assert.equal(calls.length, beforeReset, 'User can restore strict size validation');
  nodes['resize-truth'].onclick();
  nodes['truth-file'].files = [{name: 'another-mask.png', size: 80, width: 3, height: 4}];
  change('truth-file');
  await submit();
  assert.equal(calls.length, beforeReset, 'New upload must reset the resize choice');
  nodes['clear-truth'].onclick();
  assert.equal(nodes['truth-file'].files.length, 0);
  assert.equal(nodes['predict-button'].disabled, true, 'Evaluation still needs GT after clearing');
  change('prediction-mode', 'single');
  await submit();
  assert.equal(calls.at(-1).url, '/api/predict');
  assert.equal(calls.at(-1).form.get('ground_truth'), undefined);
  nodes['image-file'].files = [{name: 'broken.png', size: 100, invalid: true}];
  change('image-file');
  const beforeInvalid = calls.length;
  await submit();
  assert.equal(calls.length, beforeInvalid);
  assert.ok(nodes.message.textContent.includes('không hợp lệ'));
  nodes['image-file'].files = [{name: 'test.jpg', size: 100}];
  change('image-file');
  change('prediction-mode', 'compare');

  change('model-b', nodes['model-a'].value);
  assert.equal(nodes['predict-button'].disabled, true);
  const count = calls.length;
  await submit();
  assert.equal(calls.length, count, 'Duplicate models must not submit');

  change('prediction-mode', 'single');
  assert.equal(nodes['truth-file'].required, false);
  assert.equal(nodes.reports.hidden, true);
  let release;
  deferred = new Promise(resolve => { release = resolve; });
  const pending = submit();
  assert.equal(nodes['predict-button'].disabled, true);
  await new Promise(resolve => setImmediate(resolve));
  await vm.runInContext('refreshModels()', context);
  assert.equal(nodes['predict-button'].disabled, true, 'Refresh must not allow concurrent requests');
  change('model-a', 'final-unet');
  release();
  await pending;
  assert.equal(sections(nodes.results).length, 0, 'Outdated response must not replace changed selection');

  change('prediction-mode', 'evaluation');
  nodes['truth-file'].files = [{name: 'mask.png', size: 80}];
  change('truth-file');
  failNext = {status: 404, detail: 'Not Found'};
  await submit();
  assert.ok(nodes.message.textContent.includes('khởi động lại backend'));
  change('prediction-mode', 'single');

  failNext = true;
  await submit();
  assert.equal(nodes.message.textContent, 'Checkpoint unavailable');
  assert.equal(nodes.message.className, 'error');
  assert.equal(sections(nodes.results).length, 0);
  assert.equal(nodes['predict-button'].disabled, false);
  console.log('PASS: prediction, evaluation, explicit GT resizing with result labels, reset on new uploads, strict/corrupt validation, clear GT, viewers, stale requests, API errors.');
}
run().catch(error => { console.error(error); process.exitCode = 1; });
