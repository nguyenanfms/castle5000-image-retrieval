const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/app.js', 'utf8');
function extract(name, next) {
  const start = source.indexOf(`  function ${name}(`);
  const end = source.indexOf(`  function ${next}(`, start);
  if (start < 0 || end < 0) throw new Error(`missing ${name}`);
  return source.slice(start, end);
}
function element() {
  return {
    children: [], attrs: {}, hidden: false,
    style: {display: '', removeProperty(name) { this[name] = ''; }, setProperty() {}},
    replaceChildren(...items) { this.children = items; },
    setAttribute(name, value) { this.attrs[name] = value; },
    removeAttribute(name) { delete this.attrs[name]; },
    appendChild(item) { this.children.push(item); },
    append(...items) { this.children.push(...items); },
  };
}
const nodes = Object.fromEntries(['matched-overlay', 'search-region-evidence', 'lightbox-visual',
                                  'matched-toggle', 'matched-description'].map(id => [id, element()]));
const lightboxImg = {complete: false, naturalWidth: 0, naturalHeight: 0};
const context = {$: id => nodes[id], lightboxImg, document: {createElementNS: () => element()},
                 console, Array, Object, String, Math};
vm.createContext(context);
vm.runInContext(extract('getNaturalOverlaySize', 'abortLightboxQa') +
                extract('renderDemoEvidence', 'closeLightbox') +
                extract('highlightEvidenceBox', 'statusLabel'), context);
const ev = {image_id: 'a.jpg', regions: [{label: 'P1', box: [.1, .1, .4, .9]}]};
for (let i = 0; i < 4; i++) {
  lightboxImg.complete = false;
  context.renderDemoEvidence(null, ev);
  lightboxImg.complete = true;
  lightboxImg.naturalWidth = 100;
  lightboxImg.naturalHeight = 100;
  lightboxImg.onload();
  if (nodes['matched-overlay'].children.length !== 2 || 'hidden' in nodes['matched-overlay'].attrs)
    throw new Error('QA overlay missing after image load');
}
console.log('QA evidence overlay survived 4 open/load cycles');

// The same image can exist in the previous KIS results and in QA. Only the
// current task's evidence may be handed to the image inspector.
const start = source.indexOf('  function openLightbox(');
const end = source.indexOf('  const conditionStatus', start);
const captured = [];
const qaEvidence = {image_id: 'a.jpg', regions: [{label: 'D1', box: [.1, .1, .4, .9]}]};
const kisResult = {id: 'a.jpg', conditions: [{subject: 'old search'}]};
const qaPanel = {hidden: false};
const ctx = {currentImageId: null, currentResults: [kisResult], lightboxImg: {},
  submitResult: {}, lightboxId: {}, lightbox: {}, retrievalQaPanel: qaPanel,
  searchV2: true, selectedSearchImage: null, grid: {querySelectorAll: () => []},
  resetLightboxQaPanel() {}, renderDemoEvidence: (result, evidence) => captured.push([result, evidence]),
  renderVerification() {}, renderNeighbors() {},
  $: () => ({focus() {}})};
vm.createContext(ctx);
vm.runInContext(source.slice(start, end), ctx);
ctx.openLightbox('a.jpg', null, qaEvidence);
if (captured[0][0] !== null || captured[0][1] !== qaEvidence) throw new Error('QA inherited KIS conditions');
ctx.openLightbox('a.jpg');
if (captured[1][0] !== null) throw new Error('QA image inherited KIS conditions');
qaPanel.hidden = true;
ctx.openLightbox('a.jpg');
if (captured[2][0] !== kisResult) throw new Error('KIS conditions missing in search');
console.log('QA and KIS image evidence remain separate');
