const $ = id => document.getElementById(id);
let entries = [], generation = 0, polling = null, reportStamp = null, modelSignature = null;
let requestPending = false, apiAvailable = false;
let resizeTruth = false;
function el(tag, text, cls) { const node = document.createElement(tag); if(text != null) node.textContent = text; if(cls) node.className = cls; return node; }
async function api(url, options) {
  const r = await fetch(url, options), data = await r.json();
  if (!r.ok) {
    if (r.status === 404 && url === '/api/evaluate-image') throw new Error('Backend đang chạy phiên bản cũ, chưa có Evaluation Mode. Hãy khởi động lại backend rồi tải lại trang.');
    throw new Error(typeof data.detail === 'string' ? data.detail : `Yêu cầu không hợp lệ (${r.status})`);
  }
  if (url === '/api/evaluation' && !options) $('stale-warning').hidden = !data.report?.stale;
  return data;
}
const imageSizes = new WeakMap();
function imageSize(file, label) {
  if (!imageSizes.has(file)) imageSizes.set(file, (async () => {
    if (file.size > 10 * 1024 * 1024) throw new Error(`${label}: mỗi file tối đa 10 MB.`);
    if (file.type && !['image/jpeg', 'image/png'].includes(file.type)) throw new Error(`${label}: chỉ hỗ trợ JPEG và PNG.`);
    let bitmap;
    try { bitmap = await createImageBitmap(file, {imageOrientation: 'none'}); }
    catch { throw new Error(`${label}: file ảnh rỗng, hỏng hoặc không hợp lệ.`); }
    try {
      if (bitmap.width * bitmap.height > 16_000_000) throw new Error(`${label}: ảnh tối đa 16 triệu pixel.`);
      return {width: bitmap.width, height: bitmap.height};
    } finally { bitmap.close(); }
  })());
  return imageSizes.get(file);
}
async function validateUploads(file, truth, allowResize = resizeTruth) {
  const image = await imageSize(file, 'Ảnh nội soi');
  const gold = truth ? await imageSize(truth, 'Ground Truth') : null;
  const description = `Ảnh nội soi: ${image.width} × ${image.height} px` + (gold ? ` · Ground Truth: ${gold.width} × ${gold.height} px` : '');
  if (gold && (gold.width !== image.width || gold.height !== image.height)) {
    if (allowResize) return `${description}. Sẽ căn chỉnh mask về ${image.width} × ${image.height} px bằng nearest-neighbor trước khi tính Dice/IoU.`;
    const error = new Error(`${description}. Khác độ phân giải. Nhấn “Căn chỉnh mask theo ảnh” nếu mask thuộc đúng ảnh, hoặc chọn lại mask.`);
    error.dimensionMismatch = true;
    throw error;
  }
  return description + (gold ? ' · Kích thước khớp.' : '.');
}
async function showUploadStatus() {
  const token = generation, file = $('image-file').files[0], truth = $('truth-file').files[0];
  $('resize-tools').hidden = true;
  $('resize-truth').hidden = resizeTruth;
  $('reset-resize').hidden = !resizeTruth;
  if (!file) { $('upload-status').textContent = ''; return; }
  $('upload-status').className = 'note';
  $('upload-status').textContent = 'Đang kiểm tra ảnh và kích thước…';
  try {
    const description = await validateUploads(file, truth);
    if (token === generation) {
      $('upload-status').textContent = description;
      $('resize-tools').hidden = !resizeTruth;
    }
  } catch (error) {
    if (token === generation) {
      $('upload-status').textContent = error.message;
      $('upload-status').className = error.dimensionMismatch ? 'note' : 'note error';
      $('resize-tools').hidden = !error.dimensionMismatch;
    }
  }
}
const fmt = n => n == null ? '—' : Number(n).toFixed(4);
function table(headers, rows) { const wrap=el('div',null,'table-wrap'), t=el('table'), head=el('tr'); headers.forEach(x=>head.append(el('th',x))); const thead=el('thead');thead.append(head);t.append(thead);const body=el('tbody');rows.forEach(row=>{const tr=el('tr');row.forEach(value=>{const td=el('td');td.append(value instanceof Node ? value : document.createTextNode(String(value)));tr.append(td)});body.append(tr)});t.append(body);wrap.append(t);return wrap; }
const integer = n => n == null ? '—' : Number(n).toLocaleString('vi-VN');
function architecture(r) {
  const name = {unet: 'U-Net', unetplusplus: 'U-Net++'}[r.architecture] || r.architecture || '—';
  return name + (r.encoder ? ' + ' + (r.encoder === 'resnet34' ? 'ResNet34' : r.encoder) : '');
}
function zoomImage(title, src) {
  $('zoom-title').textContent = title;
  $('zoom-image').src = src;
  $('zoom-image').alt = title;
  $('zoom-level').value = '100';
  $('zoom-image').style.width = '100%';
  $('image-dialog').showModal();
}
$('close-zoom').onclick = () => $('image-dialog').close();
$('zoom-level').oninput = () => { $('zoom-image').style.width = $('zoom-level').value + '%'; };
function card(title, data) {
  const f = el('figure', null, 'image-card'), img = el('img'), cap = el('figcaption'), link = el('a', 'Tải PNG ↓');
  const view = el('button', null, 'image-view');
  img.src = 'data:image/png;base64,' + data;
  img.alt = title;
  view.type = 'button';
  view.setAttribute('aria-label', 'Phóng to ' + title);
  view.onclick = () => zoomImage(title, img.src);
  view.append(img, el('span', '⤢ Phóng to', 'zoom-hint'));
  link.href = img.src; link.download = title + '.png';
  cap.append(el('span', title), link);
  f.append(view, cap);
  return f;
}
function beforeAfter(original, r) {
  const box = el('figure', null, 'before-after image-card');
  const stage = el('div', null, 'before-after-stage'), base = el('img'), overlay = el('img', null, 'overlay-layer');
  base.src = 'data:image/png;base64,' + original; base.alt = 'Original — ' + r.name;
  overlay.src = 'data:image/png;base64,' + r.overlay; overlay.alt = 'Overlay — ' + r.name;
  overlay.style.clipPath = 'inset(0 0 0 50%)';
  const divider = el('span', null, 'comparison-divider'); divider.style.left = '50%';
  const leftLabel = el('span', 'Original', 'before-label'), rightLabel = el('span', 'Overlay', 'after-label');
  stage.append(base, overlay, divider, leftLabel, rightLabel);
  const controls = el('figcaption', null, 'viewer-controls'), sliderLabel = el('label', 'Before / After'), slider = el('input');
  slider.type = 'range'; slider.min = '0'; slider.max = '100'; slider.value = '50';
  slider.setAttribute('aria-label', 'Vị trí Before / After — ' + r.name);
  slider.oninput = () => {
    overlay.style.clipPath = `inset(0 0 0 ${slider.value}%)`;
    divider.style.left = slider.value + '%';
  };
  sliderLabel.append(slider);
  const toggleLabel = el('label', null, 'overlay-toggle'), toggle = el('input');
  toggle.type = 'checkbox'; toggle.checked = true;
  toggle.onchange = () => { overlay.hidden = !toggle.checked; divider.hidden = !toggle.checked; rightLabel.hidden = !toggle.checked; slider.disabled = !toggle.checked; };
  toggleLabel.append(toggle, document.createTextNode('Bật overlay'));
  const zoom = el('button', '⤢ Phóng to overlay'); zoom.type = 'button';
  zoom.onclick = () => zoomImage('Overlay — ' + r.name, toggle.checked ? overlay.src : base.src);
  controls.append(sliderLabel, toggleLabel, zoom);
  box.append(stage, controls);
  return box;
}
function stat(label, value) {
  const box = el('div', null, 'stat'); box.append(el('span', label), el('strong', value)); return box;
}
function renderResults(target, data) {
  target.replaceChildren();
  target.className = '';
  const source = el('div', null, 'image-grid source-grid');
  source.append(card('Original · Ảnh gốc', data.original));
  if (data.ground_truth) source.append(card(data.ground_truth_alignment?.resized ? 'Ground Truth · Đã căn chỉnh kích thước' : 'Ground Truth · Mask chuẩn', data.ground_truth));
  target.append(source);
  if (data.ground_truth_alignment?.resized) {
    const a = data.ground_truth_alignment;
    target.append(el('p', `Ground Truth đã đổi từ ${a.original_width} × ${a.original_height} sang ${a.target_width} × ${a.target_height} px bằng nearest-neighbor. Dice/IoU bên dưới được tính với mask đã căn chỉnh.`, 'notice'));
  }
  if (data.image_width) target.append(el('p', `Kích thước gốc: ${data.image_width} × ${data.image_height} px · Thiết bị: ${data.device?.toUpperCase() || '—'}`, 'note'));
  const comparison = el('div', null, data.results.length > 1 ? 'model-comparison' : 'single-result');
  data.results.forEach(r => {
    const section = el('article', null, 'model-result'), head = el('div', null, 'model-heading');
    head.append(el('span', r.experiment || 'MODEL', 'badge'), el('h3', r.name), el('p', architecture(r), 'model-architecture'));
    const checkpoint = el('p', null, 'checkpoint'); checkpoint.append(document.createTextNode('Checkpoint / '), el('code', r.checkpoint || '—'));
    const stats = el('div', null, 'model-stats');
    stats.append(stat('Inference time', `${r.inference_ms} ms`), stat('Số tham số', integer(r.parameter_count)),
      stat('Diện tích dự đoán', `${integer(r.foreground_pixels)} px · ${r.foreground_percent ?? '—'}%`));
    if (r.metrics) stats.append(stat('Dice ↑', fmt(r.metrics.dice)), stat('IoU ↑', fmt(r.metrics.iou)));
    section.append(head, checkpoint, stats);
    const grid = el('div', null, 'image-grid prediction-grid');
    grid.append(card('Prediction Mask — ' + r.name, r.mask), card('Overlay — ' + r.name, r.overlay));
    section.append(grid, beforeAfter(data.original, r));
    section.append(el('p', `Xử lý ${r.processing_ms} ms · Ngưỡng ${r.threshold ?? 0.5}`, 'note'));
    if (r.empty_mask) section.append(el('p', 'Mask rỗng: không có pixel đạt ngưỡng 0,5. Cần mask chuẩn để xác định có bỏ sót hay không.', 'notice'));
    if (r.errors) {
      const detail = el('details', null, 'error-detail');
      detail.append(el('summary', 'Xem bản đồ sai lệch'), card('Bản đồ sai lệch — ' + r.name, r.errors));
      const legend = el('p', null, 'error-legend');
      legend.append(el('b', '● Đúng (TP)', 'tp'), el('b', '● Dương tính giả (FP)', 'fp'), el('b', '● Bỏ sót (FN)', 'fn'));
      detail.append(legend); section.append(detail);
    }
    comparison.append(section);
  });
  target.append(comparison);
}
function metrics(results, alignment = null) {
  const value = (r, key) => r.metrics ? fmt(r.metrics[key]) : 'Cần Ground Truth';
  const rows = [
    ['Kiến trúc', ...results.map(architecture)],
    ['Checkpoint', ...results.map(r => r.checkpoint || '—')],
    ['Ground Truth', ...results.map(r => !r.metrics ? 'Chưa tải' : alignment?.resized ? 'Đã căn chỉnh kích thước · nearest-neighbor' : 'Giữ kích thước gốc')],
    ['Prediction Mask', ...results.map(() => 'Hiển thị phía trên')],
    ['Overlay', ...results.map(() => 'Hiển thị · bật/tắt · Before/After')],
    ['Dice của ảnh ↑', ...results.map(r => value(r, 'dice'))],
    ['IoU của ảnh ↑', ...results.map(r => value(r, 'iou'))],
    ['Diện tích dự đoán (pixel)', ...results.map(r => integer(r.foreground_pixels))],
    ['Diện tích dự đoán (%)', ...results.map(r => r.foreground_percent == null ? '—' : r.foreground_percent + '%')],
    ['Inference time (ms)', ...results.map(r => r.inference_ms)],
    ['Số tham số', ...results.map(r => integer(r.parameter_count))],
    ['Dương tính giả (pixel)', ...results.map(r => integer(r.metrics?.false_positive_pixels))],
    ['Bỏ sót (pixel)', ...results.map(r => integer(r.metrics?.false_negative_pixels))],
  ];
  const box = el('section', null, 'comparison-metrics');
  box.append(el('h3', results.length > 1 ? 'So sánh trực tiếp trên cùng ảnh' : 'Chỉ số của ảnh'), table(['Tiêu chí', ...results.map(r => r.name)], rows));
  return box;
}
function activateTab(tab, reset = true) {
  $('analysis-workspace').hidden = tab === 'research';
  $('reports').hidden = tab !== 'evaluation';
  $('research').hidden = tab !== 'research';
  document.querySelectorAll('[data-tab]').forEach(button => {
    button.classList.toggle('active', button.dataset.tab === tab);
    button.setAttribute('aria-pressed', String(button.dataset.tab === tab));
  });
  if (tab === 'research') { loadHistory(); loadStudy(); return; }
  const modes = {segment: 'single', compare: 'compare', evaluation: 'evaluation'};
  const previous = $('prediction-mode').value;
  $('prediction-mode').value = modes[tab];
  if (isComparison() && $('model-a').value === $('model-b').value) {
    $('model-b').value = entries.find(e => e.available && e.id !== $('model-a').value)?.id || '';
  }
  if (reset && previous !== modes[tab]) invalidate();
  updateControls();
  if (tab === 'evaluation') poll();
}
document.querySelectorAll('[data-tab]').forEach(button => button.onclick = () => activateTab(button.dataset.tab));
function notes() {
  ['a', 'b'].forEach(side => {
    $('note-' + side).textContent = entries.find(e => e.id === $('model-' + side).value)?.note || '';
  });
}
function isEvaluation() { return $('prediction-mode').value === 'evaluation'; }
function isComparison() { return ['compare', 'evaluation'].includes($('prediction-mode').value); }
function updateControls() {
  const comparing = isComparison(), evaluating = isEvaluation();
  $('truth-file').required = evaluating;
  if (evaluating) $('truth-input').open = true;
  $('truth-summary').textContent = evaluating ? 'Ground Truth Mask · bắt buộc' : 'Thêm Ground Truth Mask (tùy chọn)';
  $('mode-label').textContent = evaluating ? 'EVALUATION MODE' : comparing ? 'MODEL COMPARISON' : 'SEGMENTATION MODE';
  $('workspace-title').textContent = evaluating ? 'Đánh giá với Ground Truth' : comparing ? 'U-Net E3 & U-Net++ E4' : 'Phân đoạn ảnh nội soi';
  $('mode-description').textContent = evaluating ? 'Tải cùng lúc ảnh nội soi và mask chuẩn để tính Dice/IoU của từng mô hình.' : comparing ? 'Hai checkpoint chạy trên cùng ảnh, đối chiếu kết quả và chỉ số cạnh nhau.' : 'Chọn một mô hình để xem ảnh gốc, mask dự đoán và overlay.';
  $('input-hint').textContent = evaluating ? 'Tải ảnh và mask tương ứng. Nếu chỉ khác độ phân giải, có thể căn chỉnh mask theo ảnh trước khi đánh giá cả hai mô hình.' : 'Có thể thêm mask chuẩn để tính Dice/IoU cho ảnh này.';
  $('second-model').hidden = !comparing;
  $('model-b').required = comparing;
  const a = entries.find(e => e.id === $('model-a').value);
  const b = entries.find(e => e.id === $('model-b').value);
  const ready = a?.available && (!comparing || (b?.available && a.id !== b.id));
  $('predict-button').disabled = requestPending || !apiAvailable || !ready || (evaluating && !$('truth-file').files[0]);
  $('predict-button').textContent = requestPending ? 'Đang xử lý…' : evaluating ? 'Đánh giá hai mô hình ↗' : comparing ? 'Phân đoạn và so sánh ↗' : 'Phân đoạn polyp ↗';
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
  showUploadStatus();
}
$('clear-truth').onclick = () => { $('truth-file').value = ''; resizeTruth = false; invalidate(); };
$('resize-truth').onclick = () => { resizeTruth = true; invalidate(); };
$('reset-resize').onclick = () => { resizeTruth = false; invalidate(); };
['image-file', 'truth-file', 'model-a', 'model-b', 'prediction-mode'].forEach(id => {
  $(id).addEventListener('change', () => {
    if (id === 'image-file' || id === 'truth-file') resizeTruth = false;
    if (id === 'prediction-mode' && isComparison() && $('model-a').value === $('model-b').value) {
      $('model-b').value = entries.find(e => e.available && e.id !== $('model-a').value)?.id || '';
    }
    if (id === 'prediction-mode') activateTab(isEvaluation() ? 'evaluation' : isComparison() ? 'compare' : 'segment', false);
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
  if (isEvaluation() && !truth) {
    msg.textContent = 'Evaluation Mode cần Ground Truth Mask cùng kích thước ảnh nội soi.'; msg.className = 'error'; return;
  }
  if ([file, truth].some(f => f && f.size > 10 * 1024 * 1024)) {
    msg.textContent = 'Mỗi file tối đa 10 MB'; msg.className = 'error'; return;
  }
  if (comparing && $('model-a').value === $('model-b').value) {
    msg.textContent = 'Hãy chọn hai mô hình khác nhau.'; msg.className = 'error'; return;
  }
  const form = new FormData();
  form.append('file', file);
  if (truth) form.append('ground_truth', truth);
  if (truth && resizeTruth) form.append('resize_ground_truth', 'true');
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
    const uploadDescription = await validateUploads(file, truth);
    if (token !== generation) return;
    $('upload-status').textContent = uploadDescription;
    $('upload-status').className = 'note';
    const data = await api(isEvaluation() ? '/api/evaluate-image' : comparing ? '/api/compare' : '/api/predict', {method: 'POST', body: form});
    if (token !== generation) return;
    renderResults($('results'), data);
    $('metrics').replaceChildren(metrics(data.results, data.ground_truth_alignment));
    $('timing').textContent = `TỔNG ${data.total_ms} ms`;
    msg.textContent = data.ground_truth_alignment?.resized ? 'Đã căn chỉnh Ground Truth về kích thước ảnh và tính Dice/IoU.' : truth ? 'Đã tính Dice/IoU trên kích thước ảnh gốc.' : 'Chưa có mask chuẩn — không tính Dice/IoU.';
  } catch (error) {
    if (token === generation) {
      if (error.dimensionMismatch) {
        $('upload-status').textContent = error.message;
        $('upload-status').className = 'note';
        $('resize-tools').hidden = false;
      }
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
function report(data){if(!data){$('test-results').textContent='Chưa có báo cáo. Nhấn đánh giá để chạy toàn bộ test.csv.';return;}if(reportStamp===data.created_at)return;reportStamp=data.created_at;$('test-results').replaceChildren(el('p','Báo cáo: '+new Date(data.created_at).toLocaleString('vi-VN')),el('p',data.research_warning || 'Tính độc lập dữ liệu và quy trình huấn luyện cần được xác minh.', 'notice'),table(['Mô hình','Số ảnh','Mean Dice ↑','Mean IoU ↑','Suy luận TB (ms)'],data.models.map(m=>[m.name,m.count,fmt(m.dice),fmt(m.iou),m.inference_ms.toFixed(2)])),el('p','Hai mask cùng rỗng: Dice = IoU = 1. Trung bình theo ảnh (macro), không gộp toàn bộ pixel. Báo cáo lưu theo checkpoint tại thời điểm chạy; chạy lại khi đổi weights hoặc dữ liệu.'));$('failures').replaceChildren();data.models.forEach(m=>{const box=el('div');box.append(el('h3',m.name));box.append(table(['Ảnh test','Dice','IoU','FP','FN',''],m.samples.slice(0,5).map(s=>{const b=el('button','Xem sai lệch ↗');b.onclick=()=>showCase(m.id,s.index);return[s.image.split(/[\\/]/).at(-1),fmt(s.dice),fmt(s.iou),s.false_positive_pixels,s.false_negative_pixels,b]})));$('failures').append(box)})}
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
      if (!$('research').hidden) loadHistory();
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

async function loadStudy() {
  try {
    const data = await api('/api/study'), state = data.status;
    const labels = {not_started: 'Chưa khởi chạy', training: 'Đang huấn luyện', evaluating: 'Đang đánh giá tập test', complete: 'Đã hoàn tất', error: 'Thí nghiệm gặp lỗi'};
    $('study-status').textContent = `${labels[state.phase] || state.phase} · ${state.completed_runs || 0}/${state.total_runs || 6} lượt` + (state.run ? ` · ${state.run}` : '') + (state.error ? ` · ${state.error}` : '');
    $('study-results').replaceChildren();
    if (data.summary) {
      const fixed = data.summary.models.fixed_0_5;
      $('study-results').append(table(['Mô hình', 'Dice mean ± SD giữa seed', 'IoU mean ± SD giữa seed'], Object.entries(fixed).map(([name, metrics]) => [name, `${fmt(metrics.dice.mean)} ± ${fmt(metrics.dice.std_across_seeds)}`, `${fmt(metrics.iou.mean)} ± ${fmt(metrics.iou.std_across_seeds)}`])));
      $('study-results').append(el('p', 'Cùng ngưỡng 0,5 và độ phân giải gốc. Với ít seed, kiểm định thống kê có lực thấp; metadata bệnh nhân/video chưa được xác minh.', 'note'));
    }
  } catch (error) { $('study-status').textContent = error.message; }
}
$('refresh-study').onclick = loadStudy;
