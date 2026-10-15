const $ = id => document.getElementById(id);
let entries = [], generation = 0, polling = null, reportStamp = null, modelSignature = null;
let requestPending = false, apiAvailable = false;
function el(tag, text, cls) { const node = document.createElement(tag); if(text != null) node.textContent = text; if(cls) node.className = cls; return node; }
async function api(url, options) { const r = await fetch(url, options); const data = await r.json(); if(!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Yêu cầu không hợp lệ (${r.status})`); if(url === '/api/evaluation' && !options) $('stale-warning').hidden = !data.report?.stale; return data; }
const fmt = n => n == null ? '—' : Number(n).toFixed(4);
function table(headers, rows) { const wrap=el('div',null,'table-wrap'), t=el('table'), head=el('tr'); headers.forEach(x=>head.append(el('th',x))); const thead=el('thead');thead.append(head);t.append(thead);const body=el('tbody');rows.forEach(row=>{const tr=el('tr');row.forEach(value=>{const td=el('td');td.append(value instanceof Node ? value : document.createTextNode(String(value)));tr.append(td)});body.append(tr)});t.append(body);wrap.append(t);return wrap; }
function card(title, data) { const f=el('figure',null,'image-card'), img=el('img'), cap=el('figcaption',title), link=el('a','Tải PNG ↓');img.src='data:image/png;base64,'+data;img.alt=title;link.href=img.src;link.download=title+'.png';cap.append(link);f.append(img,cap);return f; }
function renderResults(target, data) {
  target.replaceChildren();
  target.className = '';
  const source = el('div', null, 'image-grid');
  source.append(card('Ảnh gốc', data.original));
  if (data.ground_truth) source.append(card('Mask chuẩn', data.ground_truth));
  target.append(source);
  data.results.forEach(r => {
    const section = el('article', null, 'model-result'), heading = el('h3', r.name);
    heading.append(el('span', `Suy luận ${r.inference_ms} ms · xử lý ${r.processing_ms} ms`));
    section.append(heading);
    const grid = el('div', null, 'image-grid');
    grid.append(card('Prediction Mask — ' + r.name, r.mask), card('Overlay — ' + r.name, r.overlay));
    section.append(grid);
    if (r.empty_mask) {
      section.append(el('p', 'Mask rỗng: không có pixel đạt ngưỡng 0,5. Overlay giữ nguyên ảnh gốc; cần mask chuẩn để xác định có bỏ sót hay không.', 'notice'));
    } else if (r.foreground_percent != null) {
      section.append(el('p', `Vùng dự đoán chiếm ${r.foreground_percent}% diện tích ảnh.`, 'note'));
    }
    if (r.errors) {
      section.append(card('Bản đồ sai lệch — ' + r.name, r.errors));
      const legend = el('p', null, 'error-legend');
      legend.append(el('b', '● Đúng (TP)', 'tp'), el('b', '● Dương tính giả (FP)', 'fp'), el('b', '● Bỏ sót (FN)', 'fn'));
      section.append(legend);
    }
    target.append(section);
  });
}
function metrics(results){return table(['Mô hình','Dice ↑','IoU ↑','Suy luận (ms)','FP (pixel)','FN (pixel)'],results.map(r=>[r.name,fmt(r.metrics?.dice),fmt(r.metrics?.iou),r.inference_ms,r.metrics?.false_positive_pixels??'—',r.metrics?.false_negative_pixels??'—']));}
document.querySelectorAll('[data-tab]').forEach(button=>button.onclick=()=>{document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!==button.dataset.tab);document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b===button));if(button.dataset.tab==='reports'){loadHistory();poll();}});
function notes() {
  ['a', 'b'].forEach(side => {
    $('note-' + side).textContent = entries.find(e => e.id === $('model-' + side).value)?.note || '';
  });
}
function isComparison() { return $('prediction-mode').value === 'compare'; }
function updateControls() {
  const comparing = isComparison();
  $('second-model').hidden = !comparing;
  $('model-b').required = comparing;
  const a = entries.find(e => e.id === $('model-a').value);
  const b = entries.find(e => e.id === $('model-b').value);
  const ready = a?.available && (!comparing || (b?.available && a.id !== b.id));
  $('predict-button').disabled = requestPending || !apiAvailable || !ready;
  $('predict-button').textContent = requestPending ? 'Đang phân đoạn…' : comparing ? 'Phân đoạn và so sánh ↗' : 'Phân đoạn polyp ↗';
}
function invalidate() {
  generation++;
  $('results').replaceChildren(el('p', 'Thiết lập đã thay đổi. Nhấn Phân đoạn để xem kết quả mới.'));
  $('metrics').replaceChildren();
  $('timing').textContent = 'CHƯA CHẠY';
  $('message').className = '';
  $('message').textContent = isComparison() && $('model-a').value === $('model-b').value
    ? 'Hãy chọn hai mô hình khác nhau để so sánh.'
    : 'Chọn ảnh và mô hình, sau đó nhấn Phân đoạn.';
  notes();
  updateControls();
}
['image-file', 'truth-file', 'model-a', 'model-b', 'prediction-mode'].forEach(id => {
  $(id).addEventListener('change', () => {
    if (id === 'prediction-mode' && isComparison() && $('model-a').value === $('model-b').value) {
      $('model-b').value = entries.find(e => e.available && e.id !== $('model-a').value)?.id || '';
    }
    invalidate();
  });
});
$('compare-form').onsubmit = async event => {
  event.preventDefault();
  if (requestPending) return;
  const token = ++generation, comparing = isComparison();
  const file = $('image-file').files[0], truth = $('truth-file').files[0], msg = $('message');
  if (!file) return;
  msg.className = '';
  if ([file, truth].some(f => f && f.size > 10 * 1024 * 1024)) {
    msg.textContent = 'Mỗi file tối đa 10 MB'; msg.className = 'error'; return;
  }
  if (comparing && $('model-a').value === $('model-b').value) {
    msg.textContent = 'Hãy chọn hai mô hình khác nhau.'; msg.className = 'error'; return;
  }
  const form = new FormData();
  form.append('file', file);
  if (truth) form.append('ground_truth', truth);
  if (comparing) {
    form.append('model_a', $('model-a').value);
    form.append('model_b', $('model-b').value);
  } else {
    form.append('model', $('model-a').value);
  }
  requestPending = true;
  updateControls();
  $('results').replaceChildren(el('p', 'Đang xử lý ảnh…'));
  $('metrics').replaceChildren();
  $('timing').textContent = 'ĐANG CHẠY';
  msg.textContent = comparing ? 'Đang chạy hai mô hình trên cùng ảnh…' : 'Đang chạy mô hình đã chọn…';
  try {
    const data = await api(comparing ? '/api/compare' : '/api/predict', {method: 'POST', body: form});
    if (token !== generation) return;
    renderResults($('results'), data);
    $('metrics').replaceChildren(metrics(data.results));
    $('timing').textContent = `TỔNG ${data.total_ms} ms`;
    msg.textContent = truth ? 'Đã tính Dice/IoU trên kích thước ảnh gốc.' : 'Chưa có mask chuẩn — không tính Dice/IoU.';
  } catch (error) {
    if (token === generation) {
      msg.textContent = error.message; msg.className = 'error'; $('timing').textContent = 'LỖI';
      $('results').replaceChildren(el('p', 'Không có kết quả mới. Kiểm tra thông báo lỗi rồi thử lại.'));
    }
  } finally {
    requestPending = false;
    updateControls();
  }
};
function chart(rows,keys,title){const box=el('div'),canvas=el('canvas');canvas.width=640;canvas.height=260;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',title);box.append(el('h3',title),canvas);const c=canvas.getContext('2d'),colors=['#087c64','#c17a24'];let max=Math.max(1,...rows.flatMap(r=>keys.map(k=>r[k])));const x=i=>46+i*550/Math.max(1,rows.length-1),y=v=>210-v/max*170;c.fillStyle='#71847a';c.font='12px sans-serif';for(let i=0;i<=4;i++){const v=max*i/4;c.fillText(v.toFixed(2),4,y(v)+4);c.strokeStyle='#e6ede8';c.beginPath();c.moveTo(44,y(v));c.lineTo(610,y(v));c.stroke()}keys.forEach((key,k)=>{c.strokeStyle=colors[k];c.lineWidth=2;c.beginPath();rows.forEach((r,i)=>i?c.lineTo(x(i),y(r[key])):c.moveTo(x(i),y(r[key])));c.stroke();rows.forEach((r,i)=>{c.fillStyle=colors[k];c.beginPath();c.arc(x(i),y(r[key]),2,0,Math.PI*2);c.fill()});c.fillText(key,50+k*200,20)});c.fillStyle='#71847a';c.fillText('Epoch '+rows[0].epoch,45,240);c.fillText('Epoch '+rows.at(-1).epoch,530,240);return box;}
async function loadHistory(){try{const histories=await api('/api/history');$('histories').replaceChildren();histories.forEach(h=>{const box=el('article',null,'history');box.append(el('h3',h.name));if(!h.rows.length){box.append(el('p',h.error||'Chưa có lịch sử huấn luyện. Không tạo biểu đồ từ dữ liệu giả.'),el('code',h.path));}else{box.append(chart(h.rows,['train_loss','val_loss'],'Loss theo epoch'),chart(h.rows,['val_dice','val_iou'],'Validation Dice / IoU'));const detail=el('details');detail.append(el('summary','Xem số liệu từng epoch'),table(['Epoch','Train loss','Val loss','Dice','IoU'],h.rows.map(r=>[r.epoch,fmt(r.train_loss),fmt(r.val_loss),fmt(r.val_dice),fmt(r.val_iou)])));box.append(detail)}$('histories').append(box)})}catch(e){$('histories').textContent=e.message}}
$('refresh-history').onclick=loadHistory;
async function showCase(id,index){$('case-results').textContent='Đang tải trường hợp kiểm thử…';try{const d=await api(`/api/evaluation/case/${encodeURIComponent(id)}/${index}`);renderResults($('case-results'),d);$('case-results').append(metrics(d.results));$('case-results').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){$('case-results').textContent=e.message}}
function report(data){if(!data){$('test-results').textContent='Chưa có báo cáo. Nhấn đánh giá để chạy toàn bộ test.csv.';return;}if(reportStamp===data.created_at)return;reportStamp=data.created_at;$('test-results').replaceChildren(el('p','Báo cáo: '+new Date(data.created_at).toLocaleString('vi-VN')),table(['Mô hình','Số ảnh','Mean Dice ↑','Mean IoU ↑','Suy luận TB (ms)'],data.models.map(m=>[m.name,m.count,fmt(m.dice),fmt(m.iou),m.inference_ms.toFixed(2)])),el('p','Hai mask cùng rỗng: Dice = IoU = 1. Trung bình theo ảnh (macro), không gộp toàn bộ pixel. Báo cáo lưu theo checkpoint tại thời điểm chạy; chạy lại khi đổi weights hoặc dữ liệu.'));$('failures').replaceChildren();data.models.forEach(m=>{const box=el('div');box.append(el('h3',m.name));box.append(table(['Ảnh test','Dice','IoU','FP','FN',''],m.samples.slice(0,5).map(s=>{const b=el('button','Xem sai lệch ↗');b.onclick=()=>showCase(m.id,s.index);return[s.image.split(/[\\/]/).at(-1),fmt(s.dice),fmt(s.iou),s.false_positive_pixels,s.false_negative_pixels,b]})));$('failures').append(box)})}
async function poll(){clearTimeout(polling);try{const d=await api('/api/evaluation'),j=d.job;$('evaluate').disabled=j.status==='running';$('progress').hidden=j.status!=='running';$('progress').max=j.total||1;$('progress').value=j.completed;$('evaluation-status').textContent=j.status==='running'?`Đang đánh giá ${j.completed}/${j.total} lượt suy luận…`:j.status==='error'?'Lỗi đánh giá: '+j.error:j.status==='complete'?'Đã đánh giá xong.':'';report(d.report);if(j.status==='running')polling=setTimeout(poll,1800)}catch(e){$('evaluation-status').textContent=e.message;$('evaluate').disabled=false}}
$('evaluate').onclick=async()=>{$('evaluate').disabled=true;try{await api('/api/evaluation',{method:'POST'});poll()}catch(e){$('evaluation-status').textContent=e.message;$('evaluate').disabled=false}};
function renderTraining() {
  const target = $('training-status');
  target.replaceChildren();
  entries.filter(e => e.progress).forEach(entry => {
    const p = entry.progress;
    const text = p.state === 'complete' ? 'Đã hoàn tất theo cấu hình' : p.state === 'checkpoint_available' ? 'Có checkpoint tạm thời · chưa hoàn tất' : 'Chưa có checkpoint để dự đoán';
    target.append(el('h3', entry.name), el('p', `${text} · Epoch đã ghi: ${p.epochs}/${p.max_epochs} · Best validation Dice: ${fmt(p.best_val_dice)}`));
    const phases = {training: 'Đang huấn luyện', resuming: 'Đang tiếp tục từ checkpoint', verifying: 'Đang hậu kiểm checkpoint', evaluating: 'Đang đánh giá tập test', complete: 'Đã hoàn tất huấn luyện và đánh giá', error: 'Quy trình gặp lỗi'};
    if (p.phase) target.append(el('p', `${phases[p.phase] || p.phase}${p.updated_at ? ' · cập nhật ' + new Date(p.updated_at).toLocaleString('vi-VN') : ''}`));
    const bar = el('progress'); bar.max = p.max_epochs; bar.value = p.epochs; bar.setAttribute('aria-label', 'Số epoch đã ghi'); target.append(bar);
    if (p.error) target.append(el('p', p.error, 'error'));
  });
  if (!target.childNodes.length) target.textContent = 'Không có run huấn luyện mới trong cấu hình.';
}
async function refreshModels() {
  try {
    const data = await api('/api/models');
    entries = data.models.filter(entry => entry.evaluation_enabled !== false);
    apiAvailable = true;
    $('connection').textContent = `● API kết nối · ${data.device.toUpperCase()}`;
    const signature = JSON.stringify(entries);
    if (signature !== modelSignature) {
      ['a', 'b'].forEach((side, index) => {
        const select = $('model-' + side), previous = select.value;
        select.replaceChildren();
        entries.forEach(entry => {
          const suffix = !entry.available ? ' · chờ checkpoint' : entry.progress && entry.progress.state !== 'complete' ? ' · tạm thời' : '';
          const option = el('option', entry.name + suffix); option.value = entry.id; option.disabled = !entry.available; select.append(option);
        });
        const preferred = entries.find(e => e.id === previous && e.available) || entries.filter(e => e.available)[index];
        select.value = preferred?.id || '';
      });
      renderTraining(); notes();
      if (!$('reports').hidden) loadHistory();
      modelSignature = signature;
    }
    updateControls();
  } catch (error) {
    apiAvailable = false;
    $('connection').textContent = 'Không kết nối được API';
    $('training-status').textContent = error.message;
    updateControls();
  }
}
$('refresh-models').onclick = refreshModels;
setInterval(() => { if (!document.hidden) refreshModels(); }, 15000);
refreshModels();
