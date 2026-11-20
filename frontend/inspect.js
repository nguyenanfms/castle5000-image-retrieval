(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const colors = ['#4ade80','#60a5fa','#fbbf24','#f472b6','#c4b5fd','#2dd4bf'];
  let data = null, selected = null, requestVersion = 0;
  const kindNames = {person:'Người',object:'Đồ vật',interaction_candidate:'Vùng ghép ứng viên'};
  const svgNode = (name, attrs) => {
    const node = document.createElementNS(NS,name);
    for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,String(value));
    return node;
  };
  function items() {
    if (!data) return [];
    const objects = $('inspect-mode').value === 'objects';
    return (objects ? data.objects : data.regions).map((item,index) => ({
      box:item.box,
      score:objects ? item.score : item.confidence,
      label:objects ? item.label : `${kindNames[item.kind] || item.kind}: ${(item.objects || []).map(o=>o.label).join(' + ')}`,
      index
    })).filter(item => item.score >= Number($('inspect-threshold').value));
  }
  function draw() {
    $('inspect-boxes').replaceChildren();
    for (const item of items().filter(item=>selected===null || selected===item.index)) {
      const [x1,y1,x2,y2]=item.box;
      const color=colors[item.index % colors.length];
      const caption=`${item.index+1}. ${item.label} · ${item.score.toFixed(2)}`;
      const width=Math.min(1260,caption.length*9+16);
      const labelX=Math.max(0,Math.min(x1*1280,1280-width));
      const labelY=Math.max(0,y1*720-25);
      const group=svgNode('g',{});
      group.append(svgNode('rect',{x:x1*1280,y:y1*720,width:(x2-x1)*1280,height:(y2-y1)*720,fill:'none',stroke:color,'stroke-width':3}));
      group.append(svgNode('rect',{x:labelX,y:labelY,width,height:24,fill:color,rx:3}));
      const text=svgNode('text',{x:labelX+6,y:labelY+17,fill:'#10141b','font-size':16,'font-family':'sans-serif'});
      text.textContent=caption;group.append(text);$('inspect-boxes').append(group);
    }
  }
  function render() {
    $('threshold-value').textContent=Number($('inspect-threshold').value).toFixed(2);
    $('inspect-list').replaceChildren();
    const visible=items();
    if (selected!==null && !visible.some(item=>item.index===selected)) selected=null;
    for (const item of visible) {
      const button=document.createElement('button');button.type='button';
      button.textContent=`${item.index+1}. ${item.label} · ${item.score.toFixed(2)}`;
      button.setAttribute('aria-pressed',String(selected===item.index));
      button.onclick=()=>{selected=item.index;render();};$('inspect-list').append(button);
    }
    $('inspect-status').textContent=`${data.id} · ${visible.length} khung đang hiển thị${selected===null ? '' : ' · đang chọn một khung'}`;
    if (!visible.length) $('inspect-list').textContent='Không có khung ở ngưỡng này.';
    draw();
  }
  async function load() {
    const id=$('image-id').value.trim();if(!id)return;
    const version=++requestVersion;
    $('inspect-status').textContent='Đang tải vùng…';$('auth-notice').hidden=true;
    $('inspect-content').hidden=true;data=null;selected=null;$('inspect-boxes').replaceChildren();
    try {
      const response=await fetch(`/api/regions/${encodeURIComponent(id)}`,{credentials:'same-origin'});
      const body=await response.json();if(version!==requestVersion)return;
      if(!response.ok) {
        $('auth-notice').hidden=response.status!==401;
        throw new Error(body.message || body.detail || 'Không tải được vùng ảnh.');
      }
      data=body;$('inspect-image').setAttribute('href',`/images/${encodeURIComponent(id)}`);
      $('inspect-content').hidden=false;
      history.replaceState(null,'',`?id=${encodeURIComponent(id)}`);render();
    } catch(error) {if(version===requestVersion)$('inspect-status').textContent=error.message;}
  }
  $('image-form').onsubmit=event=>{event.preventDefault();load();};
  $('inspect-mode').onchange=()=>{selected=null;if(data)render();};
  $('inspect-threshold').oninput=()=>{if(data)render();};
  $('show-all').onclick=()=>{selected=null;if(data)render();};
  $('image-id').value=new URLSearchParams(location.search).get('id') || '09_1574.jpg';
  load();
})();
