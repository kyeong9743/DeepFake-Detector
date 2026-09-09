const CAKE = (() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const fmtPct = v => (v == null || isNaN(v)) ? "—" : `${Math.round(v * 100)}%`;
  const fmtDate = s => s ? new Date(s).toLocaleString("ko-KR", { hour12: false }) : "—";
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const RISK = { "안전": "#2e9e5b", "낮음": "#7bb661", "보통": "#e0a800", "위험": "#e8702a", "매우위험": "#d63333" };

  async function api(url, opt) {
    const r = await fetch(url, opt);
    let body = null;
    try { body = await r.json(); } catch { /* no body */ }
    if (!r.ok) throw new Error(body?.detail || `${r.status} ${r.statusText}`);
    return body;
  }

  // ---------- 서버 상태 ----------
  async function health() {
    const dot = $("#health-dot"), txt = $("#health-txt");
    if (!dot) return;
    try {
      const h = await api("/api/health");
      const d = h.detector || {};
      if (d.models_loaded) { dot.className = "dot ok"; txt.textContent = `분석 서버 정상 (${d.mode}, ${d.device}${d.queue_size ? `, 대기 ${d.queue_size}` : ""})`; }
      else { dot.className = "dot bad"; txt.textContent = d.status === "unreachable" ? "분석 서버 연결 불가" : "분석 서버: 학습된 모델 없음"; }
    } catch { dot.className = "dot bad"; txt.textContent = "서버 상태 확인 실패"; }
  }

  // ---------- 업로드 페이지 ----------
  function initUpload() {
    $$(".tab").forEach(t => t.addEventListener("click", () => {
      $$(".tab").forEach(x => x.classList.toggle("on", x === t));
      $$(".tabpane").forEach(p => p.hidden = p.dataset.pane !== t.dataset.tab);
    }));
    const drop = $("#drop"), input = $("#file-input"), name = $("#file-name"), btn = $("#btn-upload"), err = $("#up-err");
    const maxMB = parseFloat(drop.closest("section").querySelector(".muted").textContent.match(/최대 (\d+)MB/)?.[1] || "300");
    const accept = (input.getAttribute("accept") || "").split(",").filter(Boolean);
    const pick = f => {
      err.textContent = "";
      if (!f) return;
      const ext = "." + f.name.split(".").pop().toLowerCase();
      if (accept.length && !accept.includes(ext)) { err.textContent = `지원하지 않는 형식입니다 (${ext}).`; btn.disabled = true; return; }
      if (f.size > maxMB * 1024 * 1024) { err.textContent = `파일이 ${maxMB}MB 를 초과합니다 (${(f.size / 1048576).toFixed(1)}MB).`; btn.disabled = true; return; }
      name.textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
      btn.disabled = false;
    };
    input.addEventListener("change", () => pick(input.files[0]));
    ["dragenter", "dragover"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", ev => { const f = ev.dataTransfer.files[0]; if (f) { input.files = ev.dataTransfer.files; pick(f); } });

    $("#upload-form").addEventListener("submit", ev => {
      ev.preventDefault();
      const f = input.files[0]; if (!f) return;
      const fd = new FormData(); fd.append("file", f); fd.append("mode", $("#upload-form input[name=mode]:checked").value);
      const prog = $("#up-progress"), bar = $(".bar", prog), ptxt = $(".ptxt", prog);
      prog.hidden = false; btn.disabled = true; err.textContent = "";
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/analyses");
      xhr.upload.onprogress = e => { if (e.lengthComputable) { const p = Math.round(e.loaded / e.total * 100); bar.style.width = p + "%"; ptxt.textContent = `업로드 중 ${p}%`; } };
      xhr.onload = () => {
        try {
          const res = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) { ptxt.textContent = "업로드 완료, 분석 대기열로 이동"; location.href = res.url; }
          else { err.textContent = res.detail || `오류 ${xhr.status}`; btn.disabled = false; prog.hidden = true; }
        } catch { err.textContent = `서버 응답 오류 (${xhr.status})`; btn.disabled = false; prog.hidden = true; }
      };
      xhr.onerror = () => { err.textContent = "네트워크 오류로 업로드에 실패했습니다."; btn.disabled = false; prog.hidden = true; };
      xhr.send(fd);
    });

    $("#url-form").addEventListener("submit", async ev => {
      ev.preventDefault();
      const e2 = $("#url-err"); e2.textContent = "";
      const fd = new FormData(); fd.append("url", $("#url-input").value.trim()); fd.append("mode", $("#url-form input[name=mode]:checked").value);
      try { const res = await api("/api/analyses", { method: "POST", body: fd }); location.href = res.url; }
      catch (x) { e2.textContent = x.message; }
    });

    api("/api/stats").then(s => {
      $("#st-total").textContent = s.total; $("#st-fake").textContent = s.fake;
      $("#st-time").textContent = s.avg_processing_time ? `${s.avg_processing_time}s` : "—";
    }).catch(() => {});
    health();
  }

  // ---------- 상세 분석 ----------
  function drawChart(canvas, fs, thr) {
    const dpr = devicePixelRatio || 1, W = canvas.clientWidth, H = 220;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    const L = 36, R = 10, T = 10, B = 26, w = W - L - R, h = H - T - B;
    const t = fs.time, tmax = Math.max(...t, 1);
    const x = v => L + v / tmax * w, y = v => T + (1 - v) * h;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#e6e8f0"; ctx.lineWidth = 1;
    [0, .25, .5, .75, 1].forEach(g => { ctx.beginPath(); ctx.moveTo(L, y(g)); ctx.lineTo(W - R, y(g)); ctx.stroke(); ctx.fillStyle = "#888"; ctx.font = "11px sans-serif"; ctx.fillText(g.toFixed(2), 4, y(g) + 4); });
    for (let s = 0; s <= tmax; s += Math.max(1, Math.round(tmax / 8))) { ctx.fillStyle = "#888"; ctx.fillText(`${s}s`, x(s) - 6, H - 8); }
    ctx.setLineDash([5, 4]); ctx.strokeStyle = "#999"; ctx.beginPath(); ctx.moveTo(L, y(thr)); ctx.lineTo(W - R, y(thr)); ctx.stroke(); ctx.setLineDash([]);
    const line = (arr, color, lw) => { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath(); let started = false;
      arr.forEach((v, i) => { if (v == null) { started = false; return; } if (!started) { ctx.moveTo(x(t[i]), y(v)); started = true; } else ctx.lineTo(x(t[i]), y(v)); }); ctx.stroke(); };
    line(fs.cnn, "#3b6cf6", 1.2); line(fs.frequency, "#f39a2a", 1.2); line(fs.lstm, "#2aa876", 1.2); line(fs.ensemble, "#d63333", 2.2);
  }

  function renderResult(a) {
    const r = a.result || {}; if (!r.scores) return;
    $("#an-result").hidden = false; $("#an-progress").hidden = true;
    const risk = $("#an-risk"); risk.textContent = r.risk_level; risk.style.background = RISK[r.risk_level] || "#888";
    $("#an-meta").textContent = `${r.frame_count} 프레임${r.frames_no_face ? ` (얼굴 없는 ${r.frames_no_face} 제외)` : ""} · ${r.video?.duration ?? "?"}s · 처리 ${r.processing_time}s · 얼굴 검출 ${fmtPct(r.face_ratio)}${r.face_px ? ` · 얼굴 ${r.face_px}px` : ""}`;
    $("#r-score").textContent = Math.round(r.deepfake_score * 100);
    $("#r-conf").textContent = fmtPct(r.confidence);
    $("#r-vote").textContent = `${r.vote.fake} / ${r.vote.total}`;
    const names = { cnn: ["CNN (ResNet)", "#3b6cf6"], lstm: ["LSTM (시간)", "#2aa876"], frequency: ["Frequency (FFT)", "#f39a2a"] };
    $("#r-bars").innerHTML = Object.entries(names).map(([k, [n, c]]) =>
      `<div class="b"><span>${n}</span><div class="track"><div class="fill" style="width:${r.scores[k] * 100}%;background:${c}"></div></div><span>${fmtPct(r.scores[k])} <span class="w">w${(r.weights[k]).toFixed(2)}</span></span></div>`).join("");
    $("#r-ens").textContent = `앙상블: ${r.ensemble_method || "가중 평균"} · 판정 임계값 ${r.threshold} · ${r.is_deepfake ? "임계값 이상 -> 딥페이크 의심" : "임계값 미만"}`;
    const qm = r.quality_messages || (r.face_warning ? ["얼굴이 검출된 프레임 비율이 낮습니다. 얼굴이 작거나 가려진 영상은 결과 신뢰성이 떨어집니다."] : []);
    $("#r-facewarn").innerHTML = qm.map(m => `<p class="warn">⚠ ${esc(m)}</p>`).join(""); $("#r-facewarn").hidden = qm.length === 0;
    if (r.frame_scores) { const c = $("#r-chart"); drawChart(c, r.frame_scores, r.threshold); addEventListener("resize", () => drawChart(c, r.frame_scores, r.threshold)); }
    const base = `/reports/${a.id}/`;
    if (r.heatmap_video) { const v = $("#r-video"); v.src = base + r.heatmap_video; v.hidden = false; }
    const heatN = Math.min(12, r.frame_count || 0);
    $("#r-thumbs").innerHTML = r.artifacts?.heatmap_dir ? Array.from({ length: heatN }, (_, i) => `<img src="${base}heatmap/frame_${String(Math.round(i * (Math.min(r.frame_count, 96) - 1) / Math.max(1, heatN - 1))).padStart(4, "0")}.jpg" loading="lazy" onclick="window.open(this.src)">`).join("") : "";
    $("#r-sus").innerHTML = (r.suspicious_frames || []).map(n => `<img src="${base}suspicious/${esc(n)}" loading="lazy" onclick="window.open(this.src)">`).join("") || `<p class="muted small">임계값 이상 프레임이 없습니다.</p>`;
    $("#dl-html").href = base + "report.html"; $("#dl-csv").href = `/api/analyses/${a.id}/download/scores.csv`;
    $("#dl-json").href = `/api/analyses/${a.id}/download/result.json`; $("#dl-video").href = `/api/analyses/${a.id}/download/heatmap.mp4`;
    $("#dl-video").hidden = !r.heatmap_video;
  }

  function initAnalysis() {
    const el = $("#an"), id = el.dataset.id;
    let delay = 1500;
    const poll = async () => {
      try {
        const a = await api(`/api/analyses/${id}`);
        $("#an-ptxt").textContent = `${a.message || a.stage || ""} (${a.progress}%)`;
        $(".bar", $("#an-progress")).style.width = a.progress + "%";
        if (a.status === "done") { renderResult(a); return; }
        if (a.status === "failed") { $("#an-progress").hidden = true; $("#an-err").textContent = a.error || "분석 실패"; $("#an-risk").textContent = "실패"; return; }
      } catch (e) { $("#an-err").textContent = e.message; }
      setTimeout(poll, delay); delay = Math.min(4000, delay * 1.15);
    };
    if (el.dataset.status === "done") api(`/api/analyses/${id}`).then(renderResult); else poll();
    health();
  }

  // ---------- 기록 ----------
  function initHistory() {
    let page = 1;
    const load = async () => {
      const f = $("#h-filter").value;
      const d = await api(`/api/analyses?page=${page}&size=20${f ? `&status_f=${f}` : ""}`);
      const tb = $("#h-table tbody");
      tb.innerHTML = d.items.map(a => `<tr>
        <td>${fmtDate(a.created_at)}</td><td class="src" title="${esc(a.source_name)}">${esc(a.source_name)}</td><td>${a.mode.toUpperCase()}</td>
        <td><span class="badge ${a.status}">${{ queued: "대기", processing: "진행 중", done: "완료", failed: "실패" }[a.status] || a.status}</span></td>
        <td>${a.risk_level ? `<span class="risk" style="background:${RISK[a.risk_level] || "#888"}">${a.risk_level}</span>` : "—"}</td>
        <td>${a.deepfake_score != null ? Math.round(a.deepfake_score * 100) : "—"}</td><td>${fmtPct(a.confidence)}</td>
        <td><a class="btn" href="/analyses/${a.id}">보기</a></td></tr>`).join("");
      $("#h-empty").hidden = d.items.length > 0;
      $("#h-page").textContent = `${page} / ${Math.max(1, Math.ceil(d.total / d.size))}`;
      $("#h-prev").disabled = page <= 1; $("#h-next").disabled = page * d.size >= d.total;
    };
    $("#h-prev").onclick = () => { page--; load(); }; $("#h-next").onclick = () => { page++; load(); };
    $("#h-filter").onchange = () => { page = 1; load(); };
    load(); health();
    setInterval(load, 8000);
  }

  // ---------- 실시간 분석 (웹UI에서는 비활성화, 개발자 확인용 1프레임 전송만 허용) ----------
  function initRealtime() {
    health();
    $("#rt-test-send").onclick = () => {
      const f = $("#rt-test-file").files[0], out = $("#rt-test-out");
      if (!f) { out.textContent = "이미지를 선택하십시오."; return; }
      const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/realtime`);
      out.textContent = "연결 중…";
      ws.onopen = async () => { ws.send(JSON.stringify({ analysis_id: "dev-test", frame_number: 1 })); ws.send(await f.arrayBuffer()); };
      ws.onmessage = m => { out.textContent = JSON.stringify(JSON.parse(m.data), null, 1); ws.close(); };
      ws.onerror = () => { out.textContent = "WebSocket 오류"; };
    };
  }

  // ---------- 시스템 모니터링 ----------
  function initMonitor() {
    const MAX = 60; // 2초 × 60 = 2분
    const hist = { gpu: [], gmem: [], cpu: [], mem: [] };
    const canvas = $("#m-chart");
    const push = (k, v) => { hist[k].push(v == null ? null : v); if (hist[k].length > MAX) hist[k].shift(); };
    const draw = () => {
      const dpr = devicePixelRatio || 1, W = canvas.clientWidth, H = 240;
      canvas.width = W * dpr; canvas.height = H * dpr;
      const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
      const L = 36, R = 10, T = 10, B = 24, w = W - L - R, h = H - T - B;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "#e6e8f0"; ctx.font = "11px sans-serif";
      [0, 25, 50, 75, 100].forEach(g => { const y = T + (1 - g / 100) * h; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke(); ctx.fillStyle = "#888"; ctx.fillText(g + "%", 4, y + 4); });
      ctx.fillStyle = "#888"; ctx.fillText("-2분", L, H - 6); ctx.fillText("지금", W - R - 26, H - 6);
      const line = (arr, color) => { ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath(); let s = false;
        arr.forEach((v, i) => { if (v == null) { s = false; return; } const x = L + (i / (MAX - 1)) * w, y = T + (1 - v / 100) * h; if (!s) { ctx.moveTo(x, y); s = true; } else ctx.lineTo(x, y); }); ctx.stroke(); };
      const pad = a => Array(MAX - a.length).fill(null).concat(a);
      line(pad(hist.gpu), "#3b6cf6"); line(pad(hist.gmem), "#d63333"); line(pad(hist.cpu), "#2aa876"); line(pad(hist.mem), "#f39a2a");
    };
    const tick = async () => {
      try {
        const m = await api("/api/metrics");
        const d = m.detector || {}, g = d.gpu || {};
        if (d.error) { $("#m-state").textContent = "분석 서버 연결 불가"; $("#m-state").className = "badge failed"; }
        else { $("#m-state").textContent = d.models_loaded ? `정상 · ${d.mode}` : "모델 미로드"; $("#m-state").className = "badge " + (d.models_loaded ? "done" : "processing"); }
        $("#m-gpu").textContent = g.util_percent != null ? `${Math.round(g.util_percent)}%` : (g.available ? "n/a" : "없음");
        $("#m-gpu-name").textContent = g.name || "";
        $("#m-gmem").textContent = g.mem_percent != null ? `${g.mem_percent}%` : "—";
        $("#m-gmem-d").textContent = g.mem_total_mb ? `${g.mem_used_mb} / ${g.mem_total_mb} MB · torch ${g.torch_allocated_mb} MB` : "";
        $("#m-cpu").textContent = d.cpu_percent != null ? `${Math.round(d.cpu_percent)}%` : "—";
        $("#m-cpu-d").textContent = d.cpu_count ? `${d.cpu_count} 코어` : "";
        $("#m-mem").textContent = d.mem_percent != null ? `${d.mem_percent}%` : "—";
        $("#m-mem-d").textContent = d.mem_total_mb ? `${d.mem_used_mb} / ${d.mem_total_mb} MB` : "";
        $("#m-queue").textContent = d.queue_size ?? "—";
        const j = d.jobs || {}; $("#m-jobs").textContent = `처리 중 ${j.processing || 0} · 완료 ${j.done || 0} · 실패 ${j.failed || 0}`;
        $("#m-temp").textContent = g.temp_c != null ? `${g.temp_c}°C` : "—";
        $("#m-power").textContent = g.power_w != null ? `${g.power_w} W` : "";
        if (d.current) { $("#m-current").textContent = `${d.current.analysis_id} — ${d.current.stage} · ${d.current.message}`; $("#m-prog").hidden = false; $("#m-prog .bar").style.width = d.current.progress + "%"; $("#m-prog .ptxt").textContent = d.current.progress + "%"; }
        else { $("#m-current").textContent = "처리 중인 분석이 없습니다."; $("#m-prog").hidden = true; }
        const b = m.backend || {}; $("#b-cpu").textContent = `${Math.round(b.cpu_percent ?? 0)}%`; $("#b-mem").textContent = `${b.mem_percent ?? "—"}%`;
        $("#b-mode").textContent = d.models_loaded ? `${d.mode} · 로드됨` : "모델 없음";
        push("gpu", g.util_percent); push("gmem", g.mem_percent); push("cpu", d.cpu_percent); push("mem", d.mem_percent);
        draw();
      } catch (e) { $("#m-state").textContent = "오류: " + e.message; $("#m-state").className = "badge failed"; }
    };
    tick(); setInterval(tick, 2000); addEventListener("resize", draw); health();
  }

  return { initUpload, initAnalysis, initHistory, initRealtime, initMonitor };
})();
