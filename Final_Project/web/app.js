'use strict';
const $ = id => document.getElementById(id);
const state = {selected:null, results:[], session:null, status:null, searchSerial:0, imageSerial:0,
 task:null, taskError:'', taskLoading:false, selecting:false, submitting:false,
 autoSerial:0, autoToggling:false};
let toastTimer;
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6000);}
function notice(message){$('notice').textContent=message;$('notice').hidden=!message;}
async function api(path,body){
 const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 let data;try{data=await response.json();}catch{throw Error(`HTTP ${response.status}: không đọc được phản hồi`);}
 if(!response.ok){
  const error=Error(typeof data.detail==='string'?data.detail:`HTTP ${response.status}: yêu cầu không hợp lệ`);
  error.status=response.status;
  throw error;
 }
 return data;
}
function busy(button,yes,label){button.disabled=yes;if(label)button.textContent=label;}
function searchModeLabel(mode){return mode==='hybrid'?'CLIP + BM25':'CLIP';}
function renderCards(records){
 state.results=records;$('results').replaceChildren();$('empty').hidden=records.length>0;
 records.forEach(record=>{
  const button=document.createElement('button');button.className='card';button.type='button';
  const img=document.createElement('img');img.src=record.thumb;img.loading='lazy';img.alt=record.filename;
  const body=document.createElement('div');body.className='card-body';
  const text=document.createElement('span');text.className='filename';text.textContent=record.filename;
  const meta=document.createElement('small');meta.textContent=`${record.batch} · ${record.hour===null?'?':String(record.hour).padStart(2,'0')}h`;text.append(meta);body.append(text);
  if(record.score!==null){const score=document.createElement('span');score.className='score';score.textContent=record.score.toFixed(3);body.append(score);}
  button.append(img,body);button.addEventListener('click',()=>openImage(record));$('results').append(button);
 });
}
async function search(imageKey=null){
 const serial=++state.searchSerial;busy($('searchButton'),true,'Đang tìm…');
 $('resultsInfo').textContent='Lần tìm đầu có thể lâu hơn vì cần nạp model vào RAM/GPU.';
 try{
  const query=$('query').value;
  const data=await api('/api/search',{query,k:Number($('limit').value),batch:$('batch').value,
   hour:$('hour').value===''?null:Number($('hour').value),image_key:imageKey,mode:$('searchMode').value});
  if(serial!==state.searchSerial)return;renderCards(data.results);
  $('resultsTitle').textContent=imageKey?'Ảnh tương tự':$('query').value.trim()?'Kết quả tìm kiếm':'Duyệt theo thứ tự frame';
  const method=query.trim()&&!imageKey?` · ${searchModeLabel(data.search_mode)}`:'';
  $('resultsInfo').textContent=`${data.results.length} ảnh · ${data.took_ms} ms${method}`;
  notice('');if($('query').value.length>250)notice('Mô tả dài có thể bị CLIP cắt bớt. Hãy ưu tiên hành động và đồ vật quan trọng.');
 }catch(error){if(serial!==state.searchSerial)return;notice(error.message);$('resultsInfo').textContent='Chưa tìm được.';}
 finally{if(serial===state.searchSerial)busy($('searchButton'),false,'Tìm ảnh ↗');}
}
async function openImage(record){
 const serial=++state.imageSerial;state.selected=record;
 $('dialogTitle').textContent=`${record.batch} / ${record.filename}`;$('selectedFilename').textContent=record.filename;
 $('selectedInfo').textContent=`${record.batch} · giờ ${record.hour??'?'} · frame ${record.frame??'?'}`;
 $('selectedItem').textContent=record.item_name||'Chưa ánh xạ được mã DRES';$('selectedDescription').textContent=record.description||'';
 $('largeImage').src=`/api/image/${record.key}`;$('largeImage').alt=record.filename;
 $('submissionResult').textContent='';updateSubmit();
 if(!$('imageDialog').open)$('imageDialog').showModal();
 $('neighbors').replaceChildren();
 try{const data=await api(`/api/neighbors/${record.key}`);if(serial!==state.imageSerial)return;
  data.results.forEach(r=>{const b=document.createElement('button');b.className='neighbor'+(r.key===record.key?' selected':'');
   const im=document.createElement('img');im.src=r.thumb;im.alt=r.filename;const s=document.createElement('span');s.textContent=r.filename;
   b.append(im,s);b.onclick=()=>openImage(r);$('neighbors').append(b);});
 }catch(error){toast(error.message);}
}
function updateSubmit(){
 const logged=state.session?.logged_in;
 const real=state.session?.mode==='real';
 const hasEvaluation=logged&&!!state.session.evaluation;
 const busy=state.taskLoading||state.selecting||state.submitting;
 const kind=state.task?.kind;
 const ready=hasEvaluation&&!!state.task?.ref&&!state.taskError;
 const isQa=hasEvaluation&&kind==='QA';
 $('evaluation').disabled=busy;
 $('logout').disabled=busy;
 $('refreshTask').disabled=busy||!hasEvaluation;
 $('taskLabel').textContent=!logged?'':!hasEvaluation?'Chọn buổi thi để xem task.':state.taskLoading?'Đang kiểm tra task…':
  state.taskError||(!state.task?.task?'Chưa có task đang chạy.':kind?state.task.task.name||kind:'Task hiện tại chưa xác định được loại.');
 $('qaForm').hidden=!isQa;
 const enabled=!!state.session?.auto_enabled;
 $('autoControls').hidden=!hasEvaluation;
 $('toggleAuto').disabled=busy||state.autoToggling;
 $('toggleAuto').textContent=enabled?'Tắt tự nộp':'Bật tự nộp';
 $('toggleAuto').setAttribute('aria-pressed',String(enabled));
 $('autoForm').hidden=!enabled;
 $('runAuto').disabled=busy||!ready||!kind||state.autoToggling;
 $('qaAnswer').readOnly=state.submitting;
 $('submitQa').textContent=real?'Nộp đáp án lên DRES thật':'Nộp thử QA';
 $('submitQa').disabled=busy||!ready||!isQa||!$('qaAnswer').value.trim();
 $('submitButton').textContent=real?'Nộp ảnh lên DRES thật':'Nộp thử local';
 $('submitButton').disabled=busy||!ready||kind!=='KIS'||!state.selected?.item_name||!!state.selected?.ambiguous_item;
 $('submissionHelp').textContent=!logged?'Kết nối ở thanh bên trước khi nộp.':!hasEvaluation?'Chọn buổi thi trước khi nộp.':
  isQa?'Dùng ảnh làm bằng chứng, rồi nhập đáp án QA ở thanh bên.':
  !ready||kind!=='KIS'?'Cập nhật task ở thanh bên trước khi nộp ảnh.':
  state.selected?.ambiguous_item?'Mã DRES trùng: cần sửa metadata.':
  real?'Đang ở chế độ thật. Nộp sai có thể bị trừ điểm.':'Mô phỏng thao tác; không gửi DRES và không chấm đúng/sai.';
}
function renderHistory(){
 const entries=state.session?.history||[];$('history').replaceChildren();
 if(!entries.length){$('history').textContent='Chưa có lượt nộp.';return;}
 entries.forEach(e=>{
  const p=document.createElement('p');
  const answer=e.kind==='QA'?`QA · ${e.text}`:`KIS · ${e.filename}`;
  p.textContent=`${e.time} · ${answer} · ${e.verdict}${e.automatic?' · Tự nộp':''}`;
  if(e.evidence_item)p.title=`${e.evidence_item} · ${e.evidence}`;
  $('history').append(p);
 });
}
function updateSession(){
 const s=state.session;const logged=s?.logged_in;$('evaluationControls').hidden=!logged;
 $('connectionLabel').textContent=!logged?'Chưa kết nối DRES':s.mode==='mock'?'● Luyện local · không tính điểm':`● DRES thật · ${s.username}`;
 $('mode').disabled=!!logged;$('loginForm').hidden=!!logged;
 if(logged)$('mode').value=s.mode;
 $('viewer').hidden=!(logged&&s.mode==='real'&&s.evaluation);
 if(!$('viewer').hidden){const base=state.status.dres_url.replace(/\/api\/v2\/?$/,'');$('viewer').href=`${base}/evaluation/viewer/${encodeURIComponent(s.evaluation)}`;}
 updateSubmit();renderHistory();
}
async function reloadEvaluations(){
 try{const list=await api('/api/dres/evaluations');$('evaluation').replaceChildren(new Option('Chọn buổi thi',''));
  list.forEach(e=>$('evaluation').append(new Option(`${e.name} · ${e.status}`,e.id)));
  $('evaluation').value=state.session?.evaluation||'';
 }catch(e){toast(e.message);}
}
function setTask(task){
 if(state.task?.ref!==task?.ref){
  $('qaAnswer').value='';
  $('qaResult').textContent='';
  $('autoHint').value='';
  resetAutoPreview();
 }
 state.task=task;
 state.taskError='';
}
function resetAutoPreview(message=''){
 state.autoSerial++;
 $('autoStatus').textContent=message;
 $('autoEvidence').hidden=true;
 $('autoEvidence').onclick=null;
}
async function toggleAuto(){
 const enabled=!state.session.auto_enabled;
 const real=state.session.mode==='real';
 if(enabled&&real){
  const provider=state.status.auto_provider;
  const transfer=provider==='gemini'?'\nGemini sẽ nhận đề và ảnh ứng viên để kiểm chứng.':'';
  if(!confirm(`Bật tự nộp cho buổi DRES thật? App sẽ gửi bài có tính điểm khi model chọn được đáp án.${transfer}`))return;
 }else if(enabled&&state.status.auto_provider==='gemini'){
  if(!confirm('Gemini sẽ nhận đề và ảnh ứng viên qua Internet để kiểm chứng. Tiếp tục?'))return;
 }
 resetAutoPreview();
 state.autoToggling=true;
 updateSubmit();
 try{
  await api('/api/dres/auto/mode',{evaluation:state.session.evaluation,enabled});
  await refreshSession();
  if(enabled)$('autoHint').focus();
 }catch(error){
  toast(error.message);
 }finally{
  state.autoToggling=false;
  updateSubmit();
 }
}
async function runAuto(event){
 event.preventDefault();
 const hint=$('autoHint').value.trim();
 const expected=state.task?.ref;
 if(!hint||!expected||!state.session?.auto_enabled)return;
 resetAutoPreview('Đang dịch và tìm ảnh để kiểm chứng…');
 const serial=state.autoSerial;
 const stillCurrent=()=>serial===state.autoSerial&&expected===state.task?.ref&&state.session?.auto_enabled;
 let committing=false;
 try{
  if((await refreshTask())?.ref!==expected){
   $('autoStatus').textContent='Task đã đổi. Dán gợi ý của task mới.';
   return;
  }
  if(!stillCurrent())return;
  const data=await api('/api/dres/auto/prepare',{hint,task_ref:expected,mode:$('searchMode').value});
  if(!stillCurrent())return;
  renderCards(data.results);
  $('resultsTitle').textContent='Ứng viên tự nộp';
  $('resultsInfo').textContent=`${data.query} · ${searchModeLabel(data.search_mode)}`;
  if(!data.preview){
   $('autoStatus').textContent='Chưa đủ bằng chứng. Thêm gợi ý tiếp theo.';
   return;
  }
  const {answer,evidence,observation}=data.preview;
  $('autoEvidence').hidden=false;
  $('autoEvidence').onclick=()=>openImage(evidence);
  $('autoStatus').textContent=`Đang tự gửi ${answer}. ${observation}`;
  state.submitting=true;
  committing=true;
  updateSubmit();
  const result=await api('/api/dres/auto/commit',{ticket:data.ticket});
  if(stillCurrent())$('autoStatus').textContent=`${result.verdict} · ${answer}. ${observation}`;
 }catch(error){
  if(stillCurrent())$('autoStatus').textContent=error.message;
 }finally{
  if(committing){
   await refreshSession().catch(error=>toast(error.message));
   state.submitting=false;
   updateSubmit();
  }
 }
}
async function refreshSession(){
 const session=await api('/api/dres/status');
 if(session.evaluation!==state.session?.evaluation||session.mode!==state.session?.mode)setTask(null);
 state.session=session;
 updateSession();
}
async function refreshTask(){
 if(!state.session?.evaluation||state.taskLoading)return null;
 state.taskLoading=true;
 updateSubmit();
 try{
  const task=await api('/api/dres/task');
  setTask(task);
  return task;
 }catch(error){
  state.taskError=error.message;
  throw error;
 }finally{
  state.taskLoading=false;
  updateSubmit();
 }
}
async function initialize(){
 try{const status=await api('/api/status');
  const firstLoad=!state.status;
  state.status=status;
  $('readyDot').classList.toggle('ready',status.ready);$('readyLabel').textContent=status.ready?'Index sẵn sàng':'Chưa có index';
  $('imageCount').textContent=status.stats?.indexed.toLocaleString('vi-VN')??'0';$('batchCount').textContent=status.batches.length;
  $('modelLabel').textContent=`${status.model.model_name} / ${status.model.pretrained}`;$('deviceLabel').textContent=`Thiết bị: ${status.device_setting}`;
  $('batch').replaceChildren(new Option('Tất cả Batch',''));status.batches.forEach(b=>$('batch').append(new Option(b,b)));
  const metadata=status.metadata;
  $('searchMode').options[1].disabled=!metadata?.available;
  if(!metadata?.available||firstLoad)$('searchMode').value=metadata?.available?'hybrid':'clip';
  $('metadataInfo').textContent=metadata?.available?
   `${metadata.coverage.toLocaleString('vi-VN')} ảnh có mô tả`:
   metadata?.error?'Chưa tải được metadata. Tìm kiếm đang dùng CLIP.':'';
  $('metadataInfo').title=metadata?.error||'';
  await refreshSession();
  if(state.session.logged_in){
   await reloadEvaluations();
   await refreshTask().catch(error=>toast(error.message));
  }
  if(status.ready)await search();else{notice('Chưa có index hợp lệ. Trong terminal chạy: python -m castle build. Sau đó khởi động lại server.');$('resultsInfo').textContent='Chờ tạo chỉ mục.';}
 }catch(e){notice(e.message);}
}
$('searchForm').onsubmit=e=>{e.preventDefault();search();};
for(const id of ['batch','hour','limit'])$(id).onchange=()=>search();
$('searchMode').onchange=()=>{
 resetAutoPreview();
 if($('query').value.trim())search();
};
document.querySelectorAll('[data-query]').forEach(b=>b.onclick=()=>{$('query').value=b.dataset.query;search();});
$('browse').onclick=()=>{$('query').value='';search();};$('refresh').onclick=initialize;
$('closeDialog').onclick=()=>$('imageDialog').close();
$('similarButton').onclick=()=>{const key=state.selected.key;$('imageDialog').close();search(key);};
$('mode').onchange=()=>{const real=$('mode').value==='real';$('credentials').hidden=!real;$('connect').textContent=real?'Đăng nhập DRES thật':'Kết nối luyện local';};
$('loginForm').onsubmit=async e=>{e.preventDefault();busy($('connect'),true);
 try{await api('/api/dres/login',{mode:$('mode').value,username:$('username').value,password:$('password').value});$('password').value='';await refreshSession();await reloadEvaluations();toast('Đã kết nối. Chọn buổi thi để nộp.');}catch(error){toast(error.message);}finally{busy($('connect'),false);}};
$('evaluation').onchange=async()=>{
 const evaluation=$('evaluation').value;
 if(!evaluation){$('evaluation').value=state.session?.evaluation||'';return;}
 state.selecting=true;
 updateSubmit();
 try{
  await api('/api/dres/select',{evaluation});
  await refreshSession();
  await refreshTask();
 }catch(error){
  $('evaluation').value=state.session?.evaluation||'';
  toast(error.message);
 }finally{
  state.selecting=false;
  updateSubmit();
 }
};
$('reloadEvaluations').onclick=reloadEvaluations;
$('refreshTask').onclick=()=>refreshTask().catch(async error=>{
 toast(error.message);
 if(error.status===401)await refreshSession().catch(err=>toast(err.message));
});
$('logout').onclick=async()=>{
 try{await api('/api/dres/logout',{});await refreshSession();$('mode').dispatchEvent(new Event('change'));}
 catch(error){toast(error.message);}
};
async function confirmSubmission(kind, value){
 if(state.submitting||state.taskLoading||state.selecting||state.taskError||state.task?.kind!==kind||!state.task.ref)return;
 const isQa=kind==='QA';
 const real=state.session?.mode==='real';
 const expected=state.task.ref;
 const resultBox=$(isQa?'qaResult':'submissionResult');
 const send=isQa?{text:value}:{key:value.key};
 const answer=isQa?value:`${value.filename}\n${value.item_name}`;
 const canShowResult=()=>isQa||state.selected?.key===value.key;
 state.submitting=true;
 updateSubmit();
 let attempted=false;
 try{
  if(real&&(await refreshTask())?.ref!==expected){
   throw Error('Task đã đổi. Kiểm tra lại câu hỏi trước khi nộp.');
  }
  const title=real?'NỘP DRES THẬT · có tính điểm':'NỘP MÔ PHỎNG LOCAL';
  const note=real?'Nộp sai có thể bị trừ điểm.':'Không gửi lên máy chủ DRES.';
  if(!confirm(`${title}\n${state.task.task.name||kind}\n\n${answer}\n\n${note}`))return;
  attempted=true;
  resultBox.textContent='Đang gửi…';
  delete resultBox.dataset.verdict;
  const result=await api(isQa?'/api/dres/submit-text':'/api/dres/submit',{
   ...send, confirmed:true, task_ref:expected
  });
  const message=`${result.verdict}: ${result.message}`;
  if(canShowResult()){
   resultBox.textContent=message;
   resultBox.dataset.verdict=result.verdict;
  }else toast(message);
  if(isQa)$('qaAnswer').value='';
 }catch(error){
  if(canShowResult()&&state.task?.ref===expected){resultBox.textContent=error.message;delete resultBox.dataset.verdict;}
  else toast(error.message);
  if(error.status===409&&real){
   const updated=await refreshTask().catch(()=>null);
   if(updated&&updated.ref!==expected)toast(error.message);
  }
  if(error.status===401)attempted=true;
 }finally{
  if(attempted)await refreshSession().catch(error=>toast(error.message));
  state.submitting=false;
  updateSubmit();
 }
}
$('submitButton').onclick=()=>confirmSubmission('KIS',state.selected);
$('toggleAuto').onclick=toggleAuto;
$('autoHint').oninput=()=>resetAutoPreview();
$('autoForm').onsubmit=runAuto;
$('qaAnswer').oninput=updateSubmit;
$('qaForm').onsubmit=event=>{
 event.preventDefault();
 const answer=$('qaAnswer').value.trim();
 if(answer)confirmSubmission('QA',answer);
};
for(let hour=0;hour<24;hour++)$('hour').append(new Option(`${String(hour).padStart(2,'0')}:00`,String(hour)));
initialize();
