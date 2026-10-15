// Exercise the actual frontend event handlers without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.childNodes = []; this.value = ''; this._text = '';
    this.files = []; this.hidden = false; this.disabled = false;
    this.handlers = {}; this.classList = {toggle() {}};
  }
  set textContent(value) { this._text = String(value); this.childNodes = []; }
  get textContent() { return this._text + this.childNodes.map(c => c.textContent).join(''); }
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
    processing_ms: 12, foreground_percent: 3, empty_mask: false, metrics: null};
}
function payload(url, form) {
  return {original: 'original', total_ms: 25, results: url.endsWith('/compare')
    ? [prediction(form.get('model_a')), prediction(form.get('model_b'))]
    : [prediction(form.get('model'))]};
}
const context = vm.createContext({
  Node: Element, FormData: Form, console,
  document: {hidden: false, getElementById: id => nodes[id], querySelectorAll: () => [],
    createElement: tag => new Element(tag), createTextNode: text => { const n = new Element('#text'); n.textContent = text; return n; }},
  setInterval() {}, clearTimeout() {}, setTimeout() {},
  fetch: async (url, options) => {
    if (url === '/api/models') return {ok: true, json: async () => ({device: 'cpu', models})};
    calls.push({url, form: options.body});
    if (failNext) { failNext = false; return {ok: false, status: 503, json: async () => ({detail: 'Checkpoint unavailable'})}; }
    if (deferred) { const wait = deferred; deferred = null; await wait; }
    return {ok: true, json: async () => payload(url, options.body)};
  },
});
function change(id, value) {
  if (value !== undefined) nodes[id].value = value;
  for (const handler of nodes[id].handlers.change || []) handler();
}
function submit() { return nodes['compare-form'].onsubmit({preventDefault() {}}); }
function sections(node) { return node.childNodes.filter(child => child.className === 'model-result'); }

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

  change('model-b', nodes['model-a'].value);
  assert.equal(nodes['predict-button'].disabled, true);
  const count = calls.length;
  await submit();
  assert.equal(calls.length, count, 'Duplicate models must not submit');

  change('prediction-mode', 'single');
  let release;
  deferred = new Promise(resolve => { release = resolve; });
  const pending = submit();
  assert.equal(nodes['predict-button'].disabled, true);
  await vm.runInContext('refreshModels()', context);
  assert.equal(nodes['predict-button'].disabled, true, 'Refresh must not allow concurrent requests');
  change('model-a', 'final-unet');
  release();
  await pending;
  assert.equal(sections(nodes.results).length, 0, 'Outdated response must not replace changed selection');

  failNext = true;
  await submit();
  assert.equal(nodes.message.textContent, 'Checkpoint unavailable');
  assert.equal(nodes.message.className, 'error');
  assert.equal(sections(nodes.results).length, 0);
  assert.equal(nodes['predict-button'].disabled, false);
  console.log('PASS: both single models, comparison, duplicate guard, pending refresh, stale response, API failure.');
}
run().catch(error => { console.error(error); process.exitCode = 1; });
