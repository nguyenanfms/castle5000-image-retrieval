// "Xem đối tượng": one grid of object crops (no whole frames) with the duplicate checks on the tiles.
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/app.js', 'utf8');
function element(tag) {
  const classes = new Set();
  return {tag, children: [], attrs: {}, dataset: {}, hidden: false, textContent: '', className: '', title: '',
    classList: {toggle(name, on) { if (on) classes.add(name); else classes.delete(name); }, contains: n => classes.has(n)},
    append(...items) { this.children.push(...items); },
    appendChild(item) { this.children.push(item); },
    setAttribute(key, value) { this.attrs[key] = value; },
    addEventListener(type, fn) { this.attrs[`on${type}`] = fn; }, focus() {}};
}
const crops = [];
const context = {document: {createElement: tag => element(tag)}, Array, String, Number, Map, Set, Boolean,
  regionSvg: (imageId, box) => { crops.push([imageId, box]); return element('svg'); },
  openLightbox: () => {}};
vm.createContext(context);
const start = source.indexOf('  function answerObjects(');
const end = source.indexOf('  function ', source.indexOf('  function createObjectsPanel(') + 1);
if (start < 0 || end < 0) throw new Error('Missing the object panel functions');
vm.runInContext(source.slice(start, end), context);

// Roster: P1 has no box, P4 repeats P2 (merged), P3 and P2 are a suspected pair.
const roster = {people_roster: [
  {label: 'P1', appearance: 'man in grey', image_id: 'a.jpg', box: null},
  {label: 'P2', appearance: 'woman in white', image_id: 'a.jpg', box: [0.5, 0.2, 0.6, 0.5]},
  {label: 'P3', appearance: 'man in a jersey', image_id: 'b.jpg', box: [0.1, 0.1, 0.3, 0.9]},
  {label: 'P4', appearance: 'woman in white', image_id: 'c.jpg', box: [0.2, 0.1, 0.4, 0.9]}],
  people_duplicates: [{a: 'P2', b: 'P4', merged: true, reid: 0.7, attributes: 0.9},
                      {a: 'P2', b: 'P3', merged: false, reid: 0.6, attributes: 0.6}],
  evidence: [{image_id: 'a.jpg'}, {image_id: 'b.jpg'}, {image_id: 'c.jpg'}]};
const panel = context.createObjectsPanel(roster);
const title = panel.children[0].textContent;
if (title !== 'Xem đối tượng · 3 người · gộp 1 trùng · 1 cặp nghi trùng · 1 chưa có khung')
  throw new Error(`Wrong title: ${title}`);
const tiles = panel.children[1].children;
const names = tiles.map(t => t.children[1].children[0].textContent);
if (crops.length !== 3 || names.join(',') !== 'P2,P4 = P2,P3') throw new Error(`Wrong tiles: ${names}`);
if (!tiles[1].className.includes('is-duplicate')) throw new Error('Merged person not marked');
const chip = tiles[0].children.find(c => c.className === 'qa-dup-chip');
if (!chip || chip.textContent !== 'nghi trùng P3') throw new Error('Missing suspected-pair chip');
chip.attrs.onclick({stopPropagation() {}});
if (!tiles[0].classList.contains('is-paired') || !tiles[2].classList.contains('is-paired') || tiles[1].classList.contains('is-paired'))
  throw new Error('Chip must light up exactly the pair');

// Ordinary answer: evidence regions are the objects; frames without regions are not shown.
crops.length = 0;
const counted = context.createObjectsPanel({evidence: [
  {image_id: 'c.jpg', regions: [{label: 'P1', box: [0, 0, 0.5, 0.5]}, {label: 'P2', box: [0.5, 0.5, 1, 1]}]},
  {image_id: 'd.jpg', regions: []}]});
if (counted.children[0].textContent !== 'Xem đối tượng · 3 đối tượng · 1 chưa có khung' || crops.length !== 2)
  throw new Error(`Wrong evidence objects: ${counted.children[0].textContent}`);
console.log('QA object panel shows crops with duplicate checks');
