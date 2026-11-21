const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/app.js', 'utf8');
function element() {
  return {children: [], attrs: {}, dataset: {}, hidden: false,
    append(...items) { this.children.push(...items); },
    appendChild(item) { this.children.push(item); },
    setAttribute(key, value) { this.attrs[key] = value; },
    addEventListener() {}, focus() {}};
}
const nodes = Object.fromEntries(['qa-images-panel', 'qa-comparison-panel', 'qa-images-tab', 'qa-comparison-tab'].map(id => [id, element()]));
const context = {$: id => nodes[id], document: {createElement: () => element()},
  createEvidenceActionBtn: () => element(), Array, String, Number};
vm.createContext(context);
const start = source.indexOf('  function createCosineHistory(');
const end = source.indexOf('  function renderRetrievalPanel(', start);
if (start < 0 || end < 0) throw new Error('Missing QA side tab functions');
vm.runInContext(source.slice(start, end), context);
const estimate = {threshold: .85, groups: [{group_id: 'G1', members: [
  {sequence: 1, image_id: 'a.jpg', label: 'D1', box: [0,0,1,1], comparison: {decision: 'first', comparisons: []}},
  {sequence: 2, image_id: 'b.jpg', label: 'D1', box: [0,0,1,1], comparison: {decision: 'matched', comparisons: [
    {image_id:'a.jpg', label:'D1', box:[0,0,1,1], group_id:'G1', similarity:.9, eligible:true, above_threshold:true}]}}
]}]};
const history = context.createCosineHistory(estimate);
if (history.children.length !== 4 || !history.children[2].children[0].textContent.includes('#1') ||
    !history.children[3].children[0].textContent.includes('#2')) throw new Error('Comparison order missing');
context.selectQaSideTab('comparison');
if (!nodes['qa-images-panel'].hidden || nodes['qa-comparison-panel'].hidden ||
    nodes['qa-comparison-tab'].attrs['aria-selected'] !== 'true') throw new Error('Comparison tab not selected');
context.selectQaSideTab('images');
if (nodes['qa-images-panel'].hidden || !nodes['qa-comparison-panel'].hidden) throw new Error('Images tab not selected');
console.log('QA side tabs and comparison order verified');
