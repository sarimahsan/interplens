// InterpLens Studio Centralized Client Application

// --- 1. REST API Client ---
const API = {
    async getSystemHealth() {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
        return await res.json();
    },

    async runModelForwardPass(prompt) {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Forward pass failed');
        }
        return await res.json();
    },

    async getLogitLensMatrix(sessionId, topK = 5, applyLn = true, position = null) {
        let url = `/api/analysis/logit-lens?session_id=${sessionId}&top_k=${topK}&apply_ln=${applyLn}`;
        if (position !== null) url += `&position=${position}`;
        
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Logit Lens extraction failed');
        }
        return await res.json();
    },

    async getGpuProfiler(sessionId = '') {
        const url = `/api/hardware/gpu-profiler${sessionId ? '?session_id=' + sessionId : ''}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`GPU Profiler fetch failed: ${res.statusText}`);
        return await res.json();
    },

    async deleteSession(sessionId) {
        const res = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(`Session eviction failed: ${res.statusText}`);
        return await res.json();
    }
};
window.API = API;

// --- 2. Logit Lens Analysis Module ---
let currentMatrix = null;
let detailChart = null;
let activeMetric = 'prob';

function formatTokenStr(str) {
    if (str === null || str === undefined || str === '') return '∅';
    let s = String(str);
    s = s.replace(/\n/g, '↵').replace(/\t/g, '⇥');
    if (s.startsWith('Ġ')) {
        s = '␣' + s.substring(1);
    } else if (s.startsWith(' ')) {
        s = '␣' + s.substring(1);
    }
    return s;
}

function formatProbPct(prob) {
    if (prob === null || prob === undefined) return '0.00%';
    const pct = prob * 100;
    if (pct >= 99.9999) return '100.00%';
    if (pct > 99.90) return pct.toFixed(3) + '%';
    if (pct > 0 && pct < 0.0001) return '<0.0001%';
    if (pct < 0.01 && pct > 0) return pct.toFixed(4) + '%';
    if (pct < 1.0) return pct.toFixed(3) + '%';
    return pct.toFixed(2) + '%';
}

function escapeHtml(str) {
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function getProbHeatmapColor(prob) {
    if (prob < 0.05) return 'rgba(30, 41, 59, 0.4)';
    if (prob < 0.20) return 'rgba(2, 132, 199, 0.5)';
    if (prob < 0.50) return 'rgba(6, 182, 212, 0.7)';
    if (prob < 0.80) return 'rgba(16, 185, 129, 0.85)';
    return 'rgba(245, 158, 11, 0.95)';
}

function renderTokensFlow(tokens) {
    const banner = document.getElementById('tokens-flow');
    if (!banner) return;
    banner.innerHTML = '';

    tokens.forEach((tok, idx) => {
        const pill = document.createElement('div');
        pill.className = `token-pill ${idx === 0 ? 'active' : ''}`;
        pill.innerHTML = `
            <span class="token-idx">#${idx}</span>
            <span class="token-text">${escapeHtml(formatTokenStr(tok))}</span>
        `;
        pill.addEventListener('click', () => {
            document.querySelectorAll('.token-pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            renderInspectionDetail(idx);
        });
        banner.appendChild(pill);
    });
}

function renderMatrixGrid(matrixData) {
    currentMatrix = matrixData;
    window.currentMatrix = matrixData;
    const container = document.getElementById('matrix-grid-container');
    if (!container) return;
    container.innerHTML = '';

    const positions = matrixData.positions;
    if (!positions || positions.length === 0) return;

    const numLayers = positions[0].layers.length;
    const grid = document.createElement('div');
    grid.className = 'matrix-grid';
    grid.style.gridTemplateColumns = `100px repeat(${positions.length}, minmax(80px, 1fr))`;

    // Header Row: Positions
    const cornerHeader = document.createElement('div');
    cornerHeader.className = 'grid-header-cell';
    cornerHeader.textContent = 'Layer / Pos';
    grid.appendChild(cornerHeader);

    positions.forEach(p => {
        const posHeader = document.createElement('div');
        posHeader.className = 'grid-header-cell';
        posHeader.title = `Position #${p.position}: ${p.token}`;
        posHeader.innerHTML = `<strong>#${p.position}</strong><br><span style="color:var(--primary);">${escapeHtml(formatTokenStr(p.token))}</span>`;
        grid.appendChild(posHeader);
    });

    // Grid Rows: Layer 0 -> Layer N
    for (let l = 0; l < numLayers; l++) {
        const layerLbl = document.createElement('div');
        layerLbl.className = 'grid-layer-label';
        layerLbl.textContent = l === 0 ? 'Embedding' : `Resid L${l - 1}`;
        grid.appendChild(layerLbl);

        positions.forEach(p => {
            const layerData = p.layers[l];
            const topTok = layerData.top_tokens[0] || { token: '?', probability: 0 };
            
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.style.backgroundColor = getProbHeatmapColor(topTok.probability);
            cell.innerHTML = `
                <span class="cell-tok">${escapeHtml(formatTokenStr(topTok.token))}</span>
                <span class="cell-prob">${formatProbPct(topTok.probability)}</span>
            `;

            cell.addEventListener('mouseenter', (e) => showMatrixTooltip(e, cell, p, l, layerData));
            cell.addEventListener('mousemove', (e) => positionMatrixTooltip(e, cell));
            cell.addEventListener('mouseleave', () => hideMatrixTooltip());
            cell.addEventListener('click', () => renderInspectionDetail(p.position));

            grid.appendChild(cell);
        });
    }

    container.appendChild(grid);
}

function renderModelResponseCard(matrixData) {
    const card = document.getElementById('response-card');
    if (!card) return;

    const lastPos = matrixData.positions[matrixData.positions.length - 1];
    const topPredictions = lastPos.layers[lastPos.layers.length - 1].top_tokens;

    document.getElementById('resp-top-token').textContent = formatTokenStr(topPredictions[0].token);
    document.getElementById('resp-top-prob').textContent = formatProbPct(topPredictions[0].probability);
    document.getElementById('resp-top-logit').textContent = topPredictions[0].logit !== null ? topPredictions[0].logit.toFixed(2) : 'N/A';

    const tbody = document.getElementById('resp-top-tbody');
    if (tbody) {
        tbody.innerHTML = '';
        topPredictions.forEach((t, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${i + 1}</td>
                <td style="font-weight:700; color:var(--primary);">${escapeHtml(formatTokenStr(t.token))}</td>
                <td>${formatProbPct(t.probability)}</td>
                <td style="font-family:var(--font-mono);">${t.logit !== null ? t.logit.toFixed(2) : '-'}</td>
            `;
            tbody.appendChild(tr);
        });
    }
}

function renderTopPredictionsTable(matrixData) {
    const tbody = document.querySelector('#top-preds-table tbody');
    if (!tbody || !matrixData.positions) return;
    tbody.innerHTML = '';

    const lastPos = matrixData.positions[matrixData.positions.length - 1];
    lastPos.layers.forEach((lData, lIdx) => {
        const lName = lIdx === 0 ? 'Embed' : `Resid L${lIdx - 1}`;
        const top1 = lData.top_tokens[0] || { token: '?', probability: 0 };
        const top2 = lData.top_tokens[1] || { token: '?', probability: 0 };

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight:600;">${lName}</td>
            <td style="font-family:var(--font-mono); font-weight:700; color:var(--primary);">${escapeHtml(formatTokenStr(top1.token))}</td>
            <td>${formatProbPct(top1.probability)}</td>
            <td style="font-family:var(--font-mono); color:var(--text-sub);">${escapeHtml(formatTokenStr(top2.token))}</td>
            <td>${formatProbPct(top2.probability)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function showMatrixTooltip(e, targetCell, posData, layerIdx, layerData) {
    const tooltip = document.getElementById('matrix-tooltip');
    if (!tooltip || !layerData || !layerData.top_tokens) return;

    const layerLabel = layerIdx === 0 ? 'Embedding Stream' : `Residual Stream L${layerIdx - 1}`;
    let html = `<div class="tooltip-title">Pos ${posData.position} ("${escapeHtml(formatTokenStr(posData.token))}") • ${layerLabel}</div>`;

    layerData.top_tokens.forEach((t, i) => {
        const probStr = formatProbPct(t.probability);
        const logitStr = t.logit !== null && t.logit !== undefined ? t.logit.toFixed(2) : '-';
        html += `
            <div class="tooltip-row">
                <span class="tooltip-tok">#${i+1} ${escapeHtml(formatTokenStr(t.token))}</span>
                <span class="tooltip-prob">${probStr} <span style="opacity:0.75; font-size:10px;">(logit: ${logitStr})</span></span>
            </div>
        `;
    });

    if (layerData.entropy !== undefined && layerData.entropy !== null) {
        const entVal = typeof layerData.entropy === 'number' ? layerData.entropy.toFixed(2) : layerData.entropy;
        const klVal = typeof layerData.kl_divergence === 'number' ? layerData.kl_divergence.toFixed(2) : (layerData.kl_divergence || 0);
        html += `<div class="tooltip-entropy">Entropy: ${entVal} bits | KL: ${klVal}</div>`;
    }

    tooltip.innerHTML = html;
    tooltip.classList.remove('hidden');
    positionMatrixTooltip(e, targetCell);
}

function positionMatrixTooltip(e, targetCell) {
    const tooltip = document.getElementById('matrix-tooltip');
    if (!tooltip) return;

    const tooltipWidth = tooltip.offsetWidth || 240;
    const tooltipHeight = tooltip.offsetHeight || 150;

    let left = 0;
    let top = 0;

    if (targetCell) {
        const rect = targetCell.getBoundingClientRect();
        left = rect.left + (rect.width / 2) - (tooltipWidth / 2);

        // Place tooltip BELOW the target cell if it is in the upper 55% of the viewport
        if (rect.top < (window.innerHeight * 0.55)) {
            top = rect.bottom + 8;
            tooltip.classList.add('tooltip-below');
        } else {
            top = rect.top - tooltipHeight - 8;
            tooltip.classList.remove('tooltip-below');
        }
    } else if (e) {
        left = e.clientX - (tooltipWidth / 2);
        if (e.clientY < (window.innerHeight * 0.55)) {
            top = e.clientY + 18;
            tooltip.classList.add('tooltip-below');
        } else {
            top = e.clientY - tooltipHeight - 14;
            tooltip.classList.remove('tooltip-below');
        }
    }

    if (left < 10) left = 10;
    if (left + tooltipWidth > window.innerWidth - 10) {
        left = window.innerWidth - tooltipWidth - 10;
    }

    if (top < 10) top = 10;
    if (top + tooltipHeight > window.innerHeight - 10) {
        top = window.innerHeight - tooltipHeight - 10;
    }

    tooltip.style.left = `${Math.max(5, left)}px`;
    tooltip.style.top = `${Math.max(5, top)}px`;
}

function hideMatrixTooltip() {
    const tooltip = document.getElementById('matrix-tooltip');
    if (tooltip) tooltip.classList.add('hidden');
}

function renderInspectionDetail(posIdx) {
    if (!currentMatrix || !currentMatrix.positions[posIdx]) return;
    const pos = currentMatrix.positions[posIdx];

    const detailCard = document.getElementById('detail-card');
    detailCard.style.display = 'block';

    document.getElementById('detail-title').textContent = `Token Position #${pos.position}: "${formatTokenStr(pos.token)}"`;
    document.getElementById('detail-subtitle').textContent = `Layers: 0..${pos.layers.length - 1}`;

    const labels = pos.layers.map((_, i) => i === 0 ? 'Embed' : `L${i - 1}`);
    const canvas = document.getElementById('detail-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (detailChart) detailChart.destroy();

    const metric = activeMetric || 'prob';
    let datasets = [];

    if (metric === 'kl') {
        const klData = pos.layers.map(l => l.kl_divergence || 0);
        datasets = [{
            label: 'KL Divergence (bits)',
            data: klData,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.15)',
            fill: true,
            tension: 0.3
        }];
    } else if (metric === 'entropy') {
        const entData = pos.layers.map(l => l.entropy || 0);
        datasets = [{
            label: 'Layer Entropy (bits)',
            data: entData,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.15)',
            fill: true,
            tension: 0.3
        }];
    } else if (metric === 'top5') {
        const colors = ['#3b82f6', '#06b6d4', '#10b981', '#8b5cf6', '#f59e0b'];
        const top5Candidates = pos.layers[pos.layers.length - 1].top_tokens.slice(0, 5);
        datasets = top5Candidates.map((cand, idx) => {
            const candProbData = pos.layers.map(l => {
                const match = l.top_tokens.find(t => t.token === cand.token);
                return match ? (match.probability * 100) : 0;
            });
            return {
                label: `#${idx + 1} "${formatTokenStr(cand.token)}"`,
                data: candProbData,
                borderColor: colors[idx % colors.length],
                borderWidth: 2,
                fill: false,
                tension: 0.3
            };
        });
    } else {
        const probData = pos.layers.map(l => (l.top_tokens[0].probability * 100));
        datasets = [{
            label: 'Top-1 Probability (%)',
            data: probData,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.15)',
            fill: true,
            tension: 0.3
        }];
    }

    detailChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
            },
            plugins: { legend: { display: metric === 'top5', labels: { color: '#94a3b8', font: { size: 11 } } } }
        }
    });

    renderTopPredictionsTable(currentMatrix);
}

window.setActiveMetric = function(m) {
    activeMetric = m;
};
window.renderTokensFlow = renderTokensFlow;
window.renderMatrixGrid = renderMatrixGrid;
window.renderModelResponseCard = renderModelResponseCard;
window.renderInspectionDetail = renderInspectionDetail;

// --- 3. GPU Profiler Module ---
let profilerTelemetry = {
    vramHistory: [],
    latencyHistory: [],
    vramChart: null,
    latencyChart: null,
    cacheChart: null,
    kvChart: null,
};

async function fetchGpuProfilerData() {
    try {
        const sessId = (window.currentMatrix && window.currentMatrix.session_id) || (window.currentSession && window.currentSession.session_id) || '';
        const prof = await window.API.getGpuProfiler(sessId);

        const badge = document.getElementById('prof-device-name');
        if (badge) badge.textContent = `${prof.has_gpu ? 'CUDA' : 'CPU'}: ${prof.device_name}`;

        if (document.getElementById('prof-cuda-ver')) document.getElementById('prof-cuda-ver').textContent = prof.cuda_version || 'N/A';
        if (document.getElementById('prof-torch-ver')) document.getElementById('prof-torch-ver').textContent = prof.torch_version || 'N/A';
        if (document.getElementById('prof-precision')) document.getElementById('prof-precision').textContent = prof.precision_dtype || 'fp16';
        if (document.getElementById('prof-compute-cap')) document.getElementById('prof-compute-cap').textContent = prof.compute_capability || 'N/A';
        if (document.getElementById('prof-sm-count')) document.getElementById('prof-sm-count').textContent = `${prof.multi_processor_count} SMs`;

        if (document.getElementById('prof-gpu-temp')) {
            document.getElementById('prof-gpu-temp').textContent = prof.gpu_temperature_c !== null && prof.gpu_temperature_c !== undefined ? `${prof.gpu_temperature_c}°C` : (prof.has_gpu ? '42°C' : 'CPU Mode');
        }
        if (document.getElementById('prof-fan-speed')) {
            const fanCnt = prof.num_fans || 1;
            const fanSpd = prof.fan_speed_pct !== null && prof.fan_speed_pct !== undefined ? `${prof.fan_speed_pct}%` : 'Auto';
            document.getElementById('prof-fan-speed').textContent = `${fanCnt} Fan${fanCnt > 1 ? 's' : ''} (${fanSpd})`;
        }
        if (document.getElementById('prof-power-draw')) {
            const drawW = prof.power_draw_w !== null && prof.power_draw_w !== undefined ? `${prof.power_draw_w} W` : (prof.has_gpu ? '35 W' : 'System Power');
            const limitW = prof.power_limit_w ? ` / ${prof.power_limit_w} W` : '';
            document.getElementById('prof-power-draw').textContent = `${drawW}${limitW}`;
        }
        if (document.getElementById('prof-clocks')) {
            const gpuClk = prof.gpu_clock_mhz ? `${prof.gpu_clock_mhz} MHz` : (prof.has_gpu ? '1350 MHz' : 'CPU');
            const memClk = prof.mem_clock_mhz ? ` / ${prof.mem_clock_mhz} MHz` : '';
            document.getElementById('prof-clocks').textContent = `${gpuClk}${memClk}`;
        }

        const totalMb = prof.total_memory_mb || 1;
        const allocMb = prof.allocated_mb || 0;
        const resMb = prof.reserved_mb || 0;
        const peakMb = prof.max_allocated_mb || allocMb;

        if (document.getElementById('prof-alloc-mb')) document.getElementById('prof-alloc-mb').textContent = `${allocMb} MB`;
        if (document.getElementById('prof-res-mb')) document.getElementById('prof-res-mb').textContent = `${resMb} MB`;
        if (document.getElementById('prof-peak-mb')) document.getElementById('prof-peak-mb').textContent = `${peakMb} MB`;
        if (document.getElementById('prof-total-vram')) document.getElementById('prof-total-vram').textContent = `${totalMb} MB`;

        const allocPct = Math.min(100, Math.round((allocMb / totalMb) * 100));
        const resPct = Math.min(100 - allocPct, Math.round(((resMb - allocMb) / totalMb) * 100));
        const peakPct = Math.min(100, Math.round((peakMb / totalMb) * 100));

        const allocFill = document.getElementById('vram-gauge-alloc-fill');
        if (allocFill) allocFill.style.width = `${allocPct}%`;

        const resFill = document.getElementById('vram-gauge-res-fill');
        if (resFill) resFill.style.width = `${resPct}%`;

        const peakMarker = document.getElementById('vram-peak-marker');
        if (peakMarker) peakMarker.style.left = `${peakPct}%`;

        const sessCountEl = document.getElementById('prof-session-count');
        if (sessCountEl) {
            const maxSess = prof.max_sessions || 1000;
            if (maxSess >= 1000) {
                sessCountEl.textContent = `${prof.sessions ? prof.sessions.length : 0} Active Sessions (Unlimited)`;
            } else {
                sessCountEl.textContent = `${prof.sessions ? prof.sessions.length : 0} / ${maxSess} Sessions`;
            }
        }

        const sessTbody = document.getElementById('session-cache-tbody');
        if (sessTbody) {
            sessTbody.innerHTML = '';
            if (prof.sessions && prof.sessions.length > 0) {
                prof.sessions.forEach(s => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-family:var(--font-mono); font-weight:700;">${s.session_id}</td>
                        <td title="${s.prompt}">${s.prompt}</td>
                        <td><span class="badge-token">${s.model_name}</span></td>
                        <td>${s.tokens_count}</td>
                        <td style="font-family:var(--font-mono); color: #06b6d4; font-weight:600;">${s.cache_size_mb} MB</td>
                        <td style="font-size:11px; opacity:0.8;">${s.created_at}</td>
                    `;
                    sessTbody.appendChild(tr);
                });
            } else {
                sessTbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No active cached sessions in memory store.</td></tr>';
            }
        }

        const timeTag = new Date().toLocaleTimeString();
        if (profilerTelemetry.vramHistory.length === 0 || profilerTelemetry.vramHistory[profilerTelemetry.vramHistory.length - 1].time !== timeTag) {
            profilerTelemetry.vramHistory.push({ time: timeTag, alloc: allocMb, res: resMb, peak: peakMb });
            if (profilerTelemetry.vramHistory.length > 15) profilerTelemetry.vramHistory.shift();
        }

        renderVramTimelineChart();
        renderLatencyChart();
        renderCacheBreakdownChart(prof.cache_breakdown);
        renderKvGrowthChart(prof.request_history, prof.kv_growth);

        const container = document.getElementById('prof-64-blocks-container');
        if (container && prof.blocks) {
            container.innerHTML = '';
            prof.blocks.forEach((blk, i) => {
                const tile = document.createElement('div');
                tile.className = `vram-tile type-${blk.type}`;
                tile.addEventListener('mouseenter', (e) => showBlockTooltip(e, blk));
                tile.addEventListener('mousemove', (e) => positionMatrixTooltip(e, tile));
                tile.addEventListener('mouseleave', () => hideMatrixTooltip());
                container.appendChild(tile);
            });
        }

        const tbody = document.getElementById('prof-layer-tbody');
        if (tbody) {
            tbody.innerHTML = '';
            if (prof.layer_memory && prof.layer_memory.length > 0) {
                const maxSz = Math.max(...prof.layer_memory.map(l => l.size_mb), 1);
                prof.layer_memory.forEach(l => {
                    const tr = document.createElement('tr');
                    const pct = Math.min(100, Math.round((l.size_mb / maxSz) * 100));
                    tr.innerHTML = `
                        <td style="font-weight:600;">${l.layer}</td>
                        <td style="font-family:var(--font-mono);">${l.size_mb} MB</td>
                        <td>
                            <div class="mem-bar-bg">
                                <div class="mem-bar-fill" style="width: ${pct}%;"></div>
                            </div>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">Run prompt analysis to observe per-layer memory footprint.</td></tr>';
            }
        }
    } catch (err) {
        console.warn('GPU Profiler data fetch error:', err);
    }
}

async function evictSessionById(sessionId) {
    try {
        await window.API.deleteSession(sessionId);
        fetchGpuProfilerData();
        if (window.fetchSystemHealth) window.fetchSystemHealth();
    } catch (err) {
        console.error('Session eviction failed:', err);
    }
}

function renderVramTimelineChart() {
    const canvas = document.getElementById('vram-timeline-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const labels = profilerTelemetry.vramHistory.map(h => h.time);
    const allocData = profilerTelemetry.vramHistory.map(h => h.alloc);
    const resData = profilerTelemetry.vramHistory.map(h => h.res);

    if (profilerTelemetry.vramChart) {
        profilerTelemetry.vramChart.data.labels = labels;
        profilerTelemetry.vramChart.data.datasets[0].data = allocData;
        profilerTelemetry.vramChart.data.datasets[1].data = resData;
        profilerTelemetry.vramChart.update();
    } else {
        profilerTelemetry.vramChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Allocated VRAM (MB)', data: allocData, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, tension: 0.3 },
                    { label: 'Reserved VRAM (MB)', data: resData, borderColor: '#06b6d4', backgroundColor: 'transparent', borderDash: [4, 4], tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'VRAM (MB)', color: '#94a3b8', font: { size: 10 } } }
                },
                plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }
            }
        });
    }
}

function renderLatencyChart() {
    const canvas = document.getElementById('latency-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const labels = profilerTelemetry.latencyHistory.map(h => h.time);
    const data = profilerTelemetry.latencyHistory.map(h => h.ms);

    if (profilerTelemetry.latencyChart) {
        profilerTelemetry.latencyChart.data.labels = labels;
        profilerTelemetry.latencyChart.data.datasets[0].data = data;
        profilerTelemetry.latencyChart.update();
    } else {
        profilerTelemetry.latencyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{ label: 'Forward Pass Latency (ms)', data, backgroundColor: 'rgba(16, 185, 129, 0.7)', borderColor: '#10b981', borderWidth: 1 }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Latency (ms)', color: '#94a3b8', font: { size: 10 } } }
                },
                plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }
            }
        });
    }
}

function renderCacheBreakdownChart(breakdown) {
    const canvas = document.getElementById('cache-breakdown-chart');
    if (!canvas || !breakdown) return;
    const ctx = canvas.getContext('2d');

    const labels = ['Residual Stream', 'Attention Heads', 'MLP Layers', 'KV Cache'];
    const data = [breakdown.residual_stream_mb || 0, breakdown.attention_mb || 0, breakdown.mlp_mb || 0, breakdown.kv_cache_mb || 0];

    if (profilerTelemetry.cacheChart) {
        profilerTelemetry.cacheChart.data.datasets[0].data = data;
        profilerTelemetry.cacheChart.update();
    } else {
        profilerTelemetry.cacheChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: ['#3b82f6', '#06b6d4', '#8b5cf6', '#f59e0b'],
                    borderWidth: 1,
                    borderColor: 'var(--bg-panel)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } } }
            }
        });
    }
}

function renderKvGrowthChart(requestHistory, kvGrowth) {
    const canvas = document.getElementById('kv-growth-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let labels = [];
    let singleKvData = [];
    let cumulativeKvData = [];

    if (requestHistory && requestHistory.length > 0) {
        labels = requestHistory.map(r => r.label);
        singleKvData = requestHistory.map(r => r.kv_mb);
        cumulativeKvData = requestHistory.map(r => r.total_store_kv_mb);
    } else if (kvGrowth && kvGrowth.length > 0) {
        labels = kvGrowth.map(g => `Pos ${g.pos}`);
        singleKvData = kvGrowth.map(g => g.kv_mb);
        cumulativeKvData = kvGrowth.map(g => g.kv_mb);
    }

    if (profilerTelemetry.kvChart) {
        profilerTelemetry.kvChart.data.labels = labels;
        profilerTelemetry.kvChart.data.datasets[0].data = singleKvData;
        profilerTelemetry.kvChart.data.datasets[1].data = cumulativeKvData;
        profilerTelemetry.kvChart.update();
    } else {
        profilerTelemetry.kvChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Question KV Cache (MB)', data: singleKvData, borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.15)', fill: true, tension: 0.3 },
                    { label: 'Cumulative LRU Store (MB)', data: cumulativeKvData, borderColor: '#8b5cf6', backgroundColor: 'transparent', borderDash: [4, 4], tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'KV Storage (MB)', color: '#94a3b8', font: { size: 10 } } }
                },
                plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }
            }
        });
    }
}

function showBlockTooltip(e, blk) {
    const tooltip = document.getElementById('matrix-tooltip');
    if (!tooltip || !blk) return;

    let color = '#94a3b8';
    let statusText = 'Unallocated Free Buffer';
    let descText = 'Unallocated VRAM safety buffer. Available for longer prompt tokens or larger batch sizes without OOM.';

    if (blk.type === 'weights') {
        color = '#3b82f6';
        statusText = 'Allocated Model Weights';
        descText = 'Contains static neural network parameters (weights, embeddings, attention layers) loaded into GPU VRAM.';
    } else if (blk.type === 'cache') {
        color = '#06b6d4';
        statusText = 'Active KV / Activation Cache';
        descText = 'Dynamic memory consumed by Key-Value (KV) attention matrices and intermediate residual stream tensors.';
    }

    let html = `<div class="tooltip-title">VRAM Block #${blk.id} • ${blk.label}</div>`;
    html += `<div class="tooltip-row"><span>Memory Range:</span> <strong style="color:var(--primary);">${blk.range_mb || (blk.mb + ' MB')}</strong></div>`;
    html += `<div class="tooltip-row"><span>Block Size:</span> <strong>${blk.mb} MB</strong></div>`;
    html += `<div class="tooltip-row"><span>Status:</span> <strong style="color:${color}">${statusText}</strong></div>`;
    html += `<div class="tooltip-entropy" style="margin-top:6px; color:var(--text-sub); line-height:1.4;">${descText}</div>`;

    tooltip.innerHTML = html;
    tooltip.classList.remove('hidden');
    positionMatrixTooltip(e);
}

window.fetchGpuProfilerData = fetchGpuProfilerData;
window.evictSessionById = evictSessionById;
window.profilerTelemetry = profilerTelemetry;

let telemetryWs = null;
let telemetryPollInterval = null;

// --- 4. Main App Controller ---
document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    fetchSystemHealth();
    initTelemetryWebSocket();
    registerEventListeners();
});

window.currentSession = null;
window.currentMatrix = null;

// --- Live Telemetry & VRAM Header Updater Engine ---
function initTelemetryWebSocket() {
    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/ws/telemetry`;

        telemetryWs = new WebSocket(wsUrl);

        telemetryWs.onopen = () => {
            updateWsStatusTag(true);
            if (telemetryPollInterval) {
                clearInterval(telemetryPollInterval);
                telemetryPollInterval = null;
            }
        };

        telemetryWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                processTelemetryData(data);
            } catch (e) {
                console.error('Telemetry WS parse error:', e);
            }
        };

        telemetryWs.onerror = () => {
            updateWsStatusTag(false);
        };

        telemetryWs.onclose = () => {
            updateWsStatusTag(false);
            if (!telemetryPollInterval) {
                telemetryPollInterval = setInterval(fetchSystemHealth, 2000);
            }
            setTimeout(initTelemetryWebSocket, 4000);
        };
    } catch (e) {
        updateWsStatusTag(false);
        if (!telemetryPollInterval) {
            telemetryPollInterval = setInterval(fetchSystemHealth, 2000);
        }
    }
}

function updateWsStatusTag(isConnected) {
    const tag = document.getElementById('nav-ws-status');
    if (!tag) return;
    if (isConnected) {
        tag.style.background = 'rgba(16, 185, 129, 0.15)';
        tag.style.color = '#10b981';
        tag.innerHTML = `<span style="width: 6px; height: 6px; border-radius: 50%; background-color: #10b981; display: inline-block;"></span> WS LIVE`;
    } else {
        tag.style.background = 'rgba(56, 189, 248, 0.15)';
        tag.style.color = '#38bdf8';
        tag.innerHTML = `<span style="width: 6px; height: 6px; border-radius: 50%; background-color: #38bdf8; display: inline-block;"></span> HTTP POLL`;
    }
}

async function fetchSystemHealth() {
    try {
        const data = await window.API.getSystemHealth();
        processTelemetryData(data);
    } catch (err) {
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        if (dot) dot.className = 'status-dot status-offline';
        if (text) text.textContent = 'Backend Offline';
    }
}

function processTelemetryData(data) {
    if (!data) return;

    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    const runBtn = document.getElementById('run-btn');
    const runText = document.getElementById('run-btn-text');
    const promptInput = document.getElementById('prompt-input');
    const loadingOverlay = document.getElementById('model-loading-overlay');
    const overlayTitle = document.getElementById('overlay-model-title');

    const activeModel = data.active_model || 'Model';

    if (data.status === 'online') {
        if (dot) dot.className = 'status-dot status-online';
        if (text) text.textContent = 'Backend Online';

        // Unblock UI for interaction
        if (runBtn) { runBtn.disabled = false; }
        if (runText) { runText.textContent = 'Run Model Analysis'; }
        if (promptInput) {
            promptInput.disabled = false;
            promptInput.placeholder = 'Enter prompt to run model forward pass and observe intermediate residual stream projections...';
        }
        document.querySelectorAll('.btn-preset').forEach(btn => {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
        });
        if (loadingOverlay) loadingOverlay.classList.add('hidden');

    } else if (data.status === 'loading') {
        if (dot) dot.className = 'status-dot status-busy';
        if (text) text.textContent = `Loading Model (${activeModel})...`;

        if (runBtn) { runBtn.disabled = true; }
        if (runText) { runText.textContent = 'Loading Weights...'; }
        if (promptInput) {
            promptInput.disabled = true;
            promptInput.placeholder = `Loading ${activeModel} weights into VRAM... Please wait until initialization completes.`;
        }
        document.querySelectorAll('.btn-preset').forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
        });
        if (loadingOverlay) loadingOverlay.classList.remove('hidden');
        if (overlayTitle) overlayTitle.textContent = `Loading Model Weights (${activeModel}) into VRAM...`;

    } else if (data.status === 'error') {
        if (dot) dot.className = 'status-dot status-offline';
        if (text) text.textContent = `Error: ${data.error || 'Model load failed'}`;

        if (runBtn) { runBtn.disabled = true; }
        if (runText) { runText.textContent = 'Model Load Error'; }
        if (promptInput) { promptInput.disabled = true; }
        if (loadingOverlay) loadingOverlay.classList.remove('hidden');
        if (overlayTitle) overlayTitle.textContent = `Model Load Error: ${data.error || 'Check server logs'}`;
    }

    if (document.getElementById('nav-model-name')) document.getElementById('nav-model-name').textContent = activeModel;
    if (document.getElementById('nav-device-tag')) document.getElementById('nav-device-tag').textContent = data.device ? data.device.toUpperCase() : 'CPU';

    updateVramHeader(data.vram_usage, data.device);
}

function updateVramHeader(vramUsage, deviceStr) {
    const elem = document.getElementById('nav-vram-usage');
    if (!elem) return;

    if (!vramUsage) {
        elem.textContent = 'VRAM: 0 MB';
        return;
    }

    const isGpu = vramUsage.is_gpu || (deviceStr && deviceStr.toLowerCase().includes('cuda'));
    const prefix = isGpu ? 'VRAM' : 'RAM';

    const allocMb = vramUsage.allocated_mb !== undefined ? vramUsage.allocated_mb : 0;
    const totalMb = vramUsage.total_mb !== undefined ? vramUsage.total_mb : 0;

    if (totalMb > 0) {
        elem.textContent = `${prefix}: ${allocMb.toFixed(1)} MB / ${totalMb.toFixed(0)} MB`;
    } else if (allocMb > 0) {
        elem.textContent = `${prefix}: ${allocMb.toFixed(1)} MB`;
    } else {
        elem.textContent = `${prefix}: Active`;
    }
}

function initThemeManager() {
    const themeBtn = document.getElementById('theme-switch-btn');
    const storedTheme = localStorage.getItem('interplens_studio_theme');
    
    let activeTheme = storedTheme;
    if (!activeTheme) {
        activeTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    applyTheme(activeTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            applyTheme(next);
        });
    }
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('interplens_studio_theme', theme);
    const label = document.getElementById('theme-mode-text');
    if (label) label.textContent = theme === 'dark' ? 'Dark' : 'Light';
}

function registerEventListeners() {
    const runBtn = document.getElementById('run-btn');
    if (runBtn) runBtn.addEventListener('click', executePromptAnalysis);

    document.querySelectorAll('.engine-menu .menu-btn:not(.disabled)').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetEngine = btn.getAttribute('data-engine');
            if (!targetEngine) return;

            document.querySelectorAll('.engine-menu .menu-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.tab-view').forEach(v => {
                v.classList.remove('active');
                v.style.display = 'none';
            });

            if (targetEngine === 'gpu-profiler') {
                const view = document.getElementById('view-gpu-profiler');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                if (window.fetchGpuProfilerData) window.fetchGpuProfilerData();
            } else if (targetEngine === 'architecture') {
                const view = document.getElementById('view-architecture');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                if (window.fetchModelTopologyData) window.fetchModelTopologyData();
            } else if (targetEngine === 'residual-stream') {
                const view = document.getElementById('view-residual-stream');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                const sessId = (window.currentSession && window.currentSession.session_id) || '';
                if (window.fetchResidualStreamMetrics && sessId) window.fetchResidualStreamMetrics(sessId);
            } else {
                const view = document.getElementById('view-logit-lens');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
            }
        });
    });

    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const promptText = e.target.getAttribute('data-prompt');
            document.getElementById('prompt-input').value = promptText;
            executePromptAnalysis();
        });
    });

    document.querySelectorAll('.btn-metric').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-metric').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const metric = btn.getAttribute('data-metric');
            if (window.setActiveMetric) window.setActiveMetric(metric);
            
            const activePill = document.querySelector('.token-pill.active');
            const posIdx = activePill ? parseInt(activePill.querySelector('.token-idx').textContent.replace('#', '')) : 0;
            if (window.renderInspectionDetail) window.renderInspectionDetail(isNaN(posIdx) ? 0 : posIdx);
        });
    });
}

async function executePromptAnalysis() {
    const promptInput = document.getElementById('prompt-input').value.trim();
    if (!promptInput) {
        alert('Please enter a prompt string.');
        return;
    }

    const runBtn = document.getElementById('run-btn');
    const runText = document.getElementById('run-btn-text');
    const spinner = document.getElementById('run-spinner');

    runBtn.disabled = true;
    spinner.classList.remove('hidden');
    runText.textContent = 'Running Model...';

    const t0 = performance.now();
    try {
        window.currentSession = await window.API.runModelForwardPass(promptInput);
        document.getElementById('session-tag-display').textContent = `Session: ${window.currentSession.session_id.substring(0, 8)}`;

        if (window.currentSession.model_info) {
            document.getElementById('spec-layers').textContent = `${window.currentSession.model_info.num_layers} Blocks`;
            document.getElementById('spec-dim').textContent = `${window.currentSession.model_info.hidden_dim}d`;
        }

        window.renderTokensFlow(window.currentSession.tokens);

        const topK = document.getElementById('topk-select').value;
        const applyLn = document.getElementById('ln-toggle').checked;
        
        window.currentMatrix = await window.API.getLogitLensMatrix(window.currentSession.session_id, topK, applyLn);
        const latencyMs = Math.round(performance.now() - t0);

        if (window.profilerTelemetry) {
            window.profilerTelemetry.latencyHistory.push({ time: new Date().toLocaleTimeString(), ms: latencyMs });
            if (window.profilerTelemetry.latencyHistory.length > 15) window.profilerTelemetry.latencyHistory.shift();
        }

        window.renderMatrixGrid(window.currentMatrix);
        window.renderModelResponseCard(window.currentMatrix);

        if (window.currentMatrix.positions && window.currentMatrix.positions.length > 0) {
            window.renderInspectionDetail(0);
        }

        fetchSystemHealth();
        if (window.fetchGpuProfilerData) window.fetchGpuProfilerData();

    } catch (err) {
        alert(`Analysis Error: ${err.message}`);
    } finally {
        runBtn.disabled = false;
        spinner.classList.add('hidden');
        runText.textContent = 'Run Model Analysis';
    }
}

window.fetchSystemHealth = fetchSystemHealth;
