// InterpLens Studio Frontend Engine

document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    fetchSystemHealth();
    registerEventListeners();
});

let currentSession = null;
let currentMatrix = null;
let detailChart = null;
let activeMetric = 'prob'; // 'prob', 'rank', 'entropy'

// --- Theme Management ---
function initThemeManager() {
    const themeBtn = document.getElementById('theme-switch-btn');
    const storedTheme = localStorage.getItem('interplens_studio_theme');
    
    let activeTheme = storedTheme;
    if (!activeTheme) {
        activeTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    applyTheme(activeTheme);

    themeBtn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        applyTheme(next);
    });

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        if (!localStorage.getItem('interplens_studio_theme')) {
            applyTheme(e.matches ? 'dark' : 'light');
        }
    });
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('interplens_studio_theme', theme);
    const textEl = document.getElementById('theme-mode-text');
    if (textEl) textEl.textContent = theme === 'dark' ? 'Dark' : 'Light';
    
    if (detailChart) {
        updateChartStyles(theme);
    }
}

// --- Health & GPU Hardware Check ---
async function fetchSystemHealth() {
    try {
        const res = await fetch('/api/health');
        if (!res.ok) return;
        const data = await res.json();
        
        const vramText = document.getElementById('vram-display');
        if (vramText && data.vram_usage) {
            const alloc = data.vram_usage.allocated_mb || 0;
            const total = data.vram_usage.total_mb || 0;
            vramText.textContent = total > 0 ? `VRAM: ${alloc.toFixed(0)} / ${total.toFixed(0)} MB` : `Device: ${data.device.toUpperCase()}`;
        }

        const modelEl = document.getElementById('header-model-name');
        const badgeEl = document.getElementById('loaded-model-badge');
        
        if (data.status === 'loading') {
            if (modelEl) modelEl.textContent = `Loading ${data.active_model}...`;
            if (badgeEl) badgeEl.textContent = `Loading ${data.active_model}...`;
            setTimeout(fetchSystemHealth, 3000);
        } else if (data.status === 'error') {
            if (modelEl) modelEl.textContent = `Error Loading Model`;
            if (badgeEl) badgeEl.textContent = `Error: ${data.error || 'Failed to load'}`;
        } else if (data.active_model) {
            if (modelEl) modelEl.textContent = data.active_model;
            if (badgeEl) badgeEl.textContent = data.active_model;
        }

        const deviceEl = document.getElementById('header-device-info');
        if (deviceEl && data.device) {
            deviceEl.textContent = data.device.toUpperCase();
        }

        fetchGpuHardwareStatus();
    } catch (err) {
        console.warn('System status update error:', err);
    }
}

async function fetchGpuHardwareStatus() {
    try {
        const res = await fetch('/api/hardware/gpu-status');
        if (!res.ok) return;
        const gpu = await res.json();

        const badge = document.getElementById('gpu-device-name-badge');
        if (badge) badge.textContent = `${gpu.has_gpu ? 'CUDA' : 'CPU'}: ${gpu.device_name}`;

        document.getElementById('gpu-alloc-val').textContent = `${gpu.allocated_mb} MB`;
        document.getElementById('gpu-res-val').textContent = `${gpu.reserved_mb} MB`;
        document.getElementById('gpu-peak-val').textContent = `${gpu.max_allocated_mb} MB`;
        document.getElementById('gpu-util-val').textContent = `${gpu.utilization_pct}%`;

        // Render 32 VRAM Block Tiles
        const container = document.getElementById('vram-blocks-container');
        if (container && gpu.blocks) {
            container.innerHTML = '';
            const mbPerBlock = gpu.total_mb > 0 ? (gpu.total_mb / 32).toFixed(0) : 0;

            gpu.blocks.forEach((blk, i) => {
                const tile = document.createElement('div');
                tile.className = `vram-tile type-${blk.type}`;
                tile.title = `VRAM Block #${i + 1}: ${blk.label} (~${mbPerBlock} MB)`;
                container.appendChild(tile);
            });
        }
    } catch (err) {
        console.warn('GPU hardware status fetch error:', err);
    }
}

async function fetchGpuProfilerData() {
    try {
        const sessId = (currentAnalysisData && currentAnalysisData.session_id) || (currentSession && currentSession.session_id) || '';
        const res = await fetch(`/api/hardware/gpu-profiler${sessId ? '?session_id=' + sessId : ''}`);
        if (!res.ok) return;
        const prof = await res.json();

        const badge = document.getElementById('prof-device-name');
        if (badge) badge.textContent = `${prof.has_gpu ? 'CUDA' : 'CPU'}: ${prof.device_name}`;

        document.getElementById('prof-compute-cap').textContent = prof.compute_capability || 'N/A';
        document.getElementById('prof-sm-count').textContent = `${prof.multi_processor_count} SMs`;
        document.getElementById('prof-total-vram').textContent = `${prof.total_memory_mb} MB`;
        document.getElementById('prof-active-tensors').textContent = `${prof.active_tensors_mb} MB`;
        document.getElementById('prof-retries').textContent = `${prof.alloc_retries}`;

        // Render 64-Block VRAM Memory Topology Map
        const container = document.getElementById('prof-64-blocks-container');
        if (container && prof.blocks) {
            container.innerHTML = '';
            prof.blocks.forEach((blk, i) => {
                const tile = document.createElement('div');
                tile.className = `vram-tile type-${blk.type}`;
                tile.title = `VRAM Topology Block #${i + 1}: ${blk.label} (${blk.mb} MB)`;
                container.appendChild(tile);
            });
        }

        // Render Layer Memory Footprint Table
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
                tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">Run prompt analysis to calculate layer memory footprints.</td></tr>';
            }
        }
    } catch (err) {
        console.warn('GPU Profiler data fetch error:', err);
    }
}

function registerEventListeners() {
    const runBtn = document.getElementById('run-btn');
    runBtn.addEventListener('click', executePromptAnalysis);

    // Sidebar Tab Switching
    document.querySelectorAll('.engine-menu .menu-btn:not(.disabled)').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetEngine = btn.getAttribute('data-engine');
            if (!targetEngine) return;

            document.querySelectorAll('.engine-menu .menu-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const viewLogit = document.getElementById('view-logit-lens');
            const viewGpu = document.getElementById('view-gpu-profiler');

            if (targetEngine === 'gpu-profiler') {
                if (viewLogit) viewLogit.style.display = 'none';
                if (viewGpu) viewGpu.style.display = 'block';
                fetchGpuProfilerData();
            } else {
                if (viewGpu) viewGpu.style.display = 'none';
                if (viewLogit) viewLogit.style.display = 'block';
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

    // Export Buttons
    const exportBtn = document.getElementById('btn-export-json');
    if (exportBtn) exportBtn.addEventListener('click', exportMatrixJSON);

    const copyCsvBtn = document.getElementById('btn-copy-csv');
    if (copyCsvBtn) copyCsvBtn.addEventListener('click', copyMatrixCSV);

    // Metric Toggles
    document.querySelectorAll('.btn-metric').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-metric').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeMetric = btn.getAttribute('data-metric');
            
            const activePill = document.querySelector('.token-pill.active');
            const posIdx = activePill ? parseInt(activePill.querySelector('.token-idx').textContent) : 0;
            renderInspectionDetail(isNaN(posIdx) ? 0 : posIdx);
        });
    });
}

// --- Execution Handler ---
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

    try {
        // Step 1: POST /api/run
        const runRes = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: promptInput })
        });

        if (!runRes.ok) {
            const err = await runRes.json();
            throw new Error(err.detail || 'Forward pass failed');
        }

        currentSession = await runRes.json();
        document.getElementById('session-tag-display').textContent = `Session: ${currentSession.session_id.substring(0, 8)}`;

        // Update Spec Cards
        if (currentSession.model_info) {
            document.getElementById('spec-layers').textContent = `${currentSession.model_info.num_layers} Blocks`;
            document.getElementById('spec-dim').textContent = `${currentSession.model_info.hidden_dim}d`;
        }

        // Render Tokens Flow
        renderTokensFlow(currentSession.tokens);

        // Step 2: GET /api/analysis/logit-lens
        const topK = document.getElementById('topk-select').value;
        const applyLn = document.getElementById('ln-toggle').checked;
        
        const lensUrl = `/api/analysis/logit-lens?session_id=${currentSession.session_id}&top_k=${topK}&apply_ln=${applyLn}`;
        const lensRes = await fetch(lensUrl);
        if (!lensRes.ok) {
            const err = await lensRes.json();
            throw new Error(err.detail || 'Logit Lens extraction failed');
        }

        currentMatrix = await lensRes.json();

        // Render Grid Matrix & Model Response Card
        renderMatrixGrid(currentMatrix);
        renderModelResponseCard(currentMatrix);

        // Select position 0 by default
        if (currentMatrix.positions && currentMatrix.positions.length > 0) {
            renderInspectionDetail(0);
        }

        fetchSystemHealth();
        fetchGpuProfilerData();

    } catch (err) {
        alert(`Analysis Error: ${err.message}`);
    } finally {
        runBtn.disabled = false;
        spinner.classList.add('hidden');
        runText.textContent = 'Execute Forward Pass';
    }
}

// --- Format Token Display ---
function formatTokenStr(tokenStr) {
    if (!tokenStr) return '""';
    let clean = tokenStr.replace(/Ġ/g, ' ').replace(/Ċ/g, '\\n');
    return clean;
}

// --- Render Tokens Flow ---
function renderTokensFlow(tokens) {
    const container = document.getElementById('tokens-strip');
    container.innerHTML = '';

    tokens.forEach((t, idx) => {
        const pill = document.createElement('div');
        pill.className = `token-pill ${idx === 0 ? 'active' : ''}`;
        pill.innerHTML = `<span class="token-idx">${idx}</span> <span class="token-str">${escapeHtml(formatTokenStr(t))}</span>`;
        pill.addEventListener('click', () => {
            document.querySelectorAll('.token-pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            renderInspectionDetail(idx);
        });
        container.appendChild(pill);
    });
}

// --- Render Model Output Response Card ---
function renderModelResponseCard(matrix) {
    const card = document.getElementById('response-card');
    if (!card || !matrix.positions || matrix.positions.length === 0) return;

    card.style.display = 'block';

    // Target the last position in prompt (which generates the model's next token output)
    const lastPos = matrix.positions[matrix.positions.length - 1];
    const finalLayer = lastPos.layers[lastPos.layers.length - 1];
    const top1 = finalLayer.top_tokens[0];

    document.getElementById('response-winner-badge').textContent = `Predicted Next Token: "${formatTokenStr(top1.token)}"`;
    document.getElementById('response-tok-val').textContent = `"${formatTokenStr(top1.token)}"`;
    document.getElementById('response-prob-val').textContent = `${(top1.probability * 100).toFixed(1)}% Probability`;
    document.getElementById('response-logit-val').textContent = `Logit: ${top1.logit !== null ? top1.logit.toFixed(2) : '-'}`;

    // Render candidate competition progress bars
    const barsContainer = document.getElementById('candidate-bars-container');
    barsContainer.innerHTML = '';

    finalLayer.top_tokens.forEach((cand, idx) => {
        const pct = (cand.probability * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'cand-bar-row';
        row.innerHTML = `
            <span class="cand-tok-name" title="${escapeHtml(cand.token)}">#${idx+1} ${escapeHtml(formatTokenStr(cand.token))}</span>
            <div class="cand-bar-outer">
                <div class="cand-bar-inner ${idx === 0 ? 'winner' : ''}" style="width: ${pct}%;"></div>
            </div>
            <span class="cand-pct-val">${pct}%</span>
        `;
        barsContainer.appendChild(row);
    });
}

// --- Render Logit Lens Matrix Grid ---
function renderMatrixGrid(matrix) {
    const container = document.getElementById('matrix-container');
    container.innerHTML = '';

    if (!matrix.positions || matrix.positions.length === 0) {
        container.innerHTML = '<div class="empty-state">No matrix data available.</div>';
        return;
    }

    const table = document.createElement('table');
    table.className = 'studio-grid';

    // Header Row
    const thead = document.createElement('thead');
    const headTr = document.createElement('tr');
    
    const cornerTh = document.createElement('th');
    cornerTh.textContent = 'Layer';
    headTr.appendChild(cornerTh);

    matrix.positions.forEach(pos => {
        const th = document.createElement('th');
        th.innerHTML = `Pos ${pos.position}<br><span style="color: var(--primary); font-family: var(--font-mono);">${escapeHtml(formatTokenStr(pos.token))}</span>`;
        headTr.appendChild(th);
    });
    thead.appendChild(headTr);
    table.appendChild(thead);

    // Body Rows
    const tbody = document.createElement('tbody');
    const totalLayers = matrix.num_layers;

    for (let l = 0; l < totalLayers; l++) {
        const tr = document.createElement('tr');
        
        const labelTd = document.createElement('td');
        labelTd.style.fontWeight = '600';
        labelTd.style.fontSize = '10px';
        labelTd.style.color = 'var(--text-muted)';
        labelTd.textContent = l === 0 ? 'Embed' : `L${l-1}`;
        tr.appendChild(labelTd);

        matrix.positions.forEach(pos => {
            const td = document.createElement('td');
            td.className = 'grid-cell';

            const layerData = pos.layers[l];
            const top1 = layerData && layerData.top_tokens ? layerData.top_tokens[0] : null;

            if (top1) {
                const prob = top1.probability;
                const pct = (prob * 100).toFixed(1);

                // Cell Intensity Styling
                if (prob > 0.6) {
                    td.style = `background-color: var(--cell-high); color: var(--cell-text-high);`;
                } else if (prob > 0.2) {
                    td.style = `background-color: var(--cell-mid); color: var(--text-main);`;
                } else {
                    td.style = `background-color: var(--cell-low); color: var(--text-sub);`;
                }

                td.innerHTML = `
                    <div class="cell-tok-name">${escapeHtml(formatTokenStr(top1.token))}</div>
                    <div class="cell-tok-pct">${pct}%</div>
                `;

                // Hover Tooltip Listener (Feature 3: Top-5 Floating Distribution)
                td.addEventListener('mouseenter', (e) => showMatrixTooltip(e, pos, l, layerData));
                td.addEventListener('mousemove', (e) => positionMatrixTooltip(e));
                td.addEventListener('mouseleave', hideMatrixTooltip);

                // Click Inspection Listener
                td.addEventListener('click', () => {
                    renderInspectionDetail(pos.position);
                });
            }
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    }

    table.appendChild(tbody);
    container.appendChild(table);
}

// --- Floating Matrix Hover Tooltip ---
function showMatrixTooltip(e, posData, layerIdx, layerData) {
    const tooltip = document.getElementById('matrix-tooltip');
    if (!tooltip || !layerData || !layerData.top_tokens) return;

    const layerLabel = layerIdx === 0 ? 'Embedding Layer' : `Layer ${layerIdx - 1}`;
    let html = `<div class="tooltip-title">Pos ${posData.position} ("${escapeHtml(formatTokenStr(posData.token))}") • ${layerLabel}</div>`;

    layerData.top_tokens.forEach((t, i) => {
        html += `
            <div class="tooltip-row">
                <span class="tooltip-tok">#${i+1} ${escapeHtml(formatTokenStr(t.token))}</span>
                <span class="tooltip-prob">${(t.probability * 100).toFixed(1)}% (logit: ${t.logit !== null ? t.logit.toFixed(2) : '-'})</span>
            </div>
        `;
    });

    if (layerData.entropy !== undefined) {
        html += `<div class="tooltip-entropy">Entropy: ${layerData.entropy} bits</div>`;
    }

    tooltip.innerHTML = html;
    tooltip.classList.remove('hidden');
    positionMatrixTooltip(e);
}

function positionMatrixTooltip(e) {
    const tooltip = document.getElementById('matrix-tooltip');
    if (!tooltip) return;
    const x = e.clientX + 15;
    const y = e.clientY + 15;
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
}

function hideMatrixTooltip() {
    const tooltip = document.getElementById('matrix-tooltip');
    if (tooltip) tooltip.classList.add('hidden');
}

// --- Render Inspection Detail Panel ---
function renderInspectionDetail(posIdx) {
    if (!currentMatrix || !currentMatrix.positions[posIdx]) return;

    const pos = currentMatrix.positions[posIdx];

    const detailCard = document.getElementById('detail-card');
    detailCard.style.display = 'block';

    document.getElementById('detail-title').textContent = `Inspection: Position ${posIdx}`;
    document.getElementById('detail-subtitle').textContent = `Target: "${formatTokenStr(pos.token)}"`;

    // Populate Table
    const tbody = document.querySelector('#drilldown-table tbody');
    tbody.innerHTML = '';

    const layerLabels = [];
    const metricData = [];

    pos.layers.forEach((lData, lIdx) => {
        const lName = lIdx === 0 ? 'Embed' : `Layer ${lIdx-1}`;
        layerLabels.push(lName);

        const tr = document.createElement('tr');
        let html = `<td><strong>${lName}</strong></td>`;
        
        const top1 = lData.top_tokens[0];
        const top2 = lData.top_tokens[1];

        if (activeMetric === 'prob') {
            metricData.push(top1 ? top1.probability * 100 : 0);
        } else if (activeMetric === 'rank') {
            const rank = pos.target_token_ranks ? pos.target_token_ranks[lIdx] : (top1 ? top1.rank : 1);
            metricData.push(rank);
        } else if (activeMetric === 'kl') {
            metricData.push(lData.kl_divergence || 0);
        } else if (activeMetric === 'entropy') {
            metricData.push(lData.entropy || 0);
        }

        if (top1) {
            html += `<td><span class="badge-token">${escapeHtml(formatTokenStr(top1.token))}</span></td>
                     <td>${(top1.probability * 100).toFixed(1)}%</td>`;
        } else {
            html += `<td>-</td><td>-</td>`;
        }

        if (top2) {
            html += `<td><span class="badge-token" style="background-color: var(--bg-subtle); color: var(--text-sub);">${escapeHtml(formatTokenStr(top2.token))}</span></td>
                     <td>${(top2.probability * 100).toFixed(1)}%</td>`;
        } else {
            html += `<td>-</td><td>-</td>`;
        }

        tr.innerHTML = html;
        tbody.appendChild(tr);
    });

    if (activeMetric === 'ribbon') {
        renderRibbonChart(layerLabels, pos.top5_trajectories || {});
    } else {
        renderLineChart(layerLabels, metricData);
    }
}

function renderRibbonChart(labels, top5Trajectories) {
    const ctx = document.getElementById('prob-chart').getContext('2d');
    if (detailChart) detailChart.destroy();

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    const textColor = isDark ? '#94a3b8' : '#475569';
    const colorPalette = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6'];

    const datasets = [];
    let colorIdx = 0;

    for (const [tokStr, probSeries] of Object.entries(top5Trajectories)) {
        const color = colorPalette[colorIdx % colorPalette.length];
        datasets.push({
            label: `Candidate "${formatTokenStr(tokStr)}"`,
            data: probSeries,
            borderColor: color,
            backgroundColor: color + '15',
            fill: false,
            tension: 0.2,
            pointRadius: 3
        });
        colorIdx++;
    }

    detailChart = new Chart(ctx, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    title: { display: true, text: 'Probability (%)', color: textColor, font: { family: 'Inter', size: 10 } },
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { family: 'Inter', size: 11 } }
                },
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { family: 'Inter', size: 11 } }
                }
            },
            plugins: {
                legend: { labels: { color: textColor, font: { family: 'Inter', size: 11, weight: '500' } } }
            }
        }
    });
}

function renderLineChart(labels, dataPts) {
    const ctx = document.getElementById('prob-chart').getContext('2d');
    if (detailChart) detailChart.destroy();

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const primaryColor = activeMetric === 'rank' ? '#eab308' : (activeMetric === 'kl' ? '#ef4444' : (activeMetric === 'entropy' ? '#06b6d4' : (isDark ? '#3b82f6' : '#2563eb')));
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    const textColor = isDark ? '#94a3b8' : '#475569';

    let yLabel = 'Top-1 Probability (%)';
    let yScaleOpts = {
        grid: { color: gridColor },
        ticks: { color: textColor, font: { family: 'Inter', size: 11 } }
    };

    if (activeMetric === 'prob') {
        yLabel = 'Top-1 Probability (%)';
        yScaleOpts.min = 0;
        yScaleOpts.max = 100;
    } else if (activeMetric === 'rank') {
        yLabel = 'Target Token Rank (1 is top)';
        yScaleOpts.reverse = true;
        const maxRank = Math.max(...dataPts, 5);
        yScaleOpts.suggestedMin = 1;
        yScaleOpts.suggestedMax = maxRank;
    } else if (activeMetric === 'kl') {
        yLabel = 'KL Divergence KL(P_L || P_L-1) (bits)';
        yScaleOpts.beginAtZero = true;
        yScaleOpts.suggestedMax = Math.max(...dataPts, 1.0);
    } else if (activeMetric === 'entropy') {
        yLabel = 'Prediction Entropy (bits)';
        yScaleOpts.beginAtZero = true;
        yScaleOpts.suggestedMax = Math.max(...dataPts, 2.0);
    }

    detailChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: yLabel,
                data: dataPts,
                borderColor: primaryColor,
                backgroundColor: primaryColor + '22',
                fill: true,
                tension: 0.25,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: yScaleOpts,
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { family: 'Inter', size: 11 } }
                }
            },
            plugins: {
                legend: { labels: { color: textColor, font: { family: 'Inter', size: 12, weight: '500' } } }
            }
        }
    });
}

// --- Data Export & Copy ---
function exportMatrixJSON() {
    if (!currentMatrix) {
        alert('No analysis matrix data available to export.');
        return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentMatrix, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute("href", dataStr);
    dlAnchor.setAttribute("download", `interplens_logit_lens_${currentMatrix.session_id}.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
}

function copyMatrixCSV() {
    if (!currentMatrix || !currentMatrix.positions) {
        alert('No analysis matrix data available to copy.');
        return;
    }

    let csv = "Layer," + currentMatrix.positions.map(p => `Pos_${p.position} ("${formatTokenStr(p.token)}")`).join(",") + "\n";

    for (let l = 0; l < currentMatrix.num_layers; l++) {
        const layerName = l === 0 ? "Embed" : `L${l-1}`;
        const row = [layerName];
        currentMatrix.positions.forEach(pos => {
            const top1 = pos.layers[l] && pos.layers[l].top_tokens ? pos.layers[l].top_tokens[0] : null;
            if (top1) {
                row.push(`"${formatTokenStr(top1.token)}" (${(top1.probability*100).toFixed(1)}%)`);
            } else {
                row.push("-");
            }
        });
        csv += row.join(",") + "\n";
    }

    navigator.clipboard.writeText(csv).then(() => {
        alert('✅ Logit Lens CSV matrix copied to clipboard!');
    }).catch(err => {
        alert('Copy failed: ' + err);
    });
}

function updateChartStyles(theme) {
    if (currentMatrix && currentMatrix.positions) {
        const activePill = document.querySelector('.token-pill.active');
        const posIdx = activePill ? parseInt(activePill.querySelector('.token-idx').textContent) : 0;
        renderInspectionDetail(isNaN(posIdx) ? 0 : posIdx);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
