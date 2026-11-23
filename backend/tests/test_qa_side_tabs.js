// "Xem đối tượng": every answer lists the objects it rests on as crops only (no whole frames).
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/app.js', 'utf8');
function element(tag) {
  return {tag, children: [], attrs: {}, dataset: {}, hidden: false, textContent: '', className: '',
    append(...items) { this.children.push(...items); },
    appendChild(item) { this.children.push(item); },
    setAttribute(key, value) { this.attrs[key] = value; },
    addEventListener() {}, focus() {}};
}
const crops = [];
const context = {document: {createElement: tag => element(tag)}, Array, String, Number, Map,
  regionSvg: (imageId, box) => { crops.push([imageId, box]); return element('svg'); },
  openLightbox: () => {}};
vm.createContext(context);
const start = source.indexOf('  function answerObjects(');
const end = source.indexOf('  function ', source.indexOf('  function createObjectsPanel(') + 1);
if (start < 0 || end < 0) throw new Error('Missing the object panel functions');
vm.runInContext(source.slice(start, end), context);

// Roster answer: 3 people, one without a box.
const roster = {people_roster: [
  {label: 'P1', appearance: 'man in grey', image_id: 'a.jpg', box: null},
  {label: 'P2', appearance: 'woman in white', image_id: 'a.jpg', box: [0.5, 0.2, 0.6, 0.5]},
  {label: 'P3', appearance: 'man in a jersey', image_id: 'b.jpg', box: [0.1, 0.1, 0.3, 0.9]}],
  evidence: [{image_id: 'a.jpg'}, {image_id: 'b.jpg'}]};
const panel = context.createObjectsPanel(roster);
const title = panel.children[0].textContent;
const tiles = panel.children[1].children;
if (title !== 'Xem đối tượng · 3 người (2 có ảnh cắt)') throw new Error(`Wrong title: ${title}`);
if (tiles.length !== 2 || crops.length !== 2 || crops[0][0] !== 'a.jpg' || crops[1][0] !== 'b.jpg')
  throw new Error('Only boxed people may become crops');
if (!panel.children[2] || !panel.children[2].textContent.includes('1 người chưa có khung'))
  throw new Error('Missing note for people without a box');

// Ordinary answer: regions of the evidence frames are the objects; frames without regions are skipped.
crops.length = 0;
const counted = context.createObjectsPanel({evidence: [
  {image_id: 'c.jpg', regions: [{label: 'P1', box: [0, 0, 0.5, 0.5]}, {label: 'P2', box: [0.5, 0.5, 1, 1]}]},
  {image_id: 'd.jpg', regions: []}]});
if (counted.children[0].textContent !== 'Xem đối tượng · 3 đối tượng (2 có ảnh cắt)' || crops.length !== 2)
  throw new Error(`Wrong evidence objects: ${counted.children[0].textContent}`);
console.log('QA object panel shows only object crops');
