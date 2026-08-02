// InterpLens Client Application JS

document.addEventListener('DOMContentLoaded', () => {
    // Theme Switcher Setup
    initTheme();

    // Fetch initial health & hardware stats
    fetchHealthStatus();

    // Event Listeners
    setupEventListeners();
});

let currentSessionData = null;
let currentMatrixData = null;
let probChart = null;

// --- Theme Management ---
function initTheme() {
    const themeBtn = document.getElementById('theme-toggle-btn');
    const storedTheme = localStorage.getItem('interplens_theme');
    
    let activeTheme = storedTheme;
    if (!activeTheme) {
        activeTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    setTheme(activeTheme);

    themeBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
    });

    // Listen for OS theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        if (!localStorage.getItem('interplens_theme')) {
            setTheme(e.matches ? 'dark' : 'light');
        }
    });
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('interplens_theme', theme);
    const label = document.querySelector('.theme-label');
    if (label) label.textContent = theme === 'dark' ? 'Dark' : 'Light';
    
    if (probChart) {
        updateChartTheme(theme);
    }
}

// --- API Calls & Health Check ---
async function fetchHealthStatus() {
    try {
        const res = await fetch('/api/health');
        if (!res.ok) return;
        const data = await res.json();
        
        // Update VRAM / Device badge
        const vramText = document.getElementById('vram-text');
        if (vramText && data.vram_usage) {
            const allocated = data.vram_usage.allocated_mb || 0;
            const total = data.vram_usage.total_mb || 0;
            vramText.textContent = `${data.device.toUpperCase()}: ${allocated.toFixed(0)} / ${total.toFixed(0)} MB`;
        }

        const modelDisplay = document.getElementById('model-name-display');
        if (modelDisplay && data.active_model) {
            modelDisplay.textContent = data.active_model;
        }
    } catch (err) {
        console.warn('Health check warning:', err);
    }
}

function setupEventListeners() {
    // Run button click
    const runBtn = document.getElementById('run-btn');
    runBtn.addEventListener('click', handleRunAnalysis);

    // Sample prompts buttons
    document.querySelectorAll('.btn-sample').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const promptText = e.target.getAttribute('data-prompt');
            document.getElementById('prompt-input').value = promptText;
            handleRunAnalysis();
        });
    });
}

// --- Main Execution Handler ---
async function handleRunAnalysis() {
    const promptInput = document.getElementById('prompt-input').value.trim();
    if (!promptInput) {
        alert('Please enter a prompt to analyze.');
        return;
    }

    const runBtn = document.getElementById('run-btn');
    const runBtnText = document.getElementById('run-btn-text');
    const spinner = document.getElementById('run-spinner');

    runBtn.disabled = true;
    spinner.classList.remove('hidden');
    runBtnText.textContent = 'Processing...';

    try {
        // Step 1: POST /api/run
        const modelName = document.getElementById('model-select').value;
        const runRes = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: promptInput, model_name: modelName })
        });

        if (!runRes.ok) {
            const errJson = await runRes.json();
            throw new Error(errJson.detail || 'Forward pass failed');
        }

        currentSessionData = await runRes.json();
        document.getElementById('session-tag-display').textContent = `Session: ${currentSessionData.session_id}`;

        // Render prompt tokens strip
        renderTokensStrip(currentSessionData.tokens);

        // Step 2: GET /api/analysis/logit-lens
        const topK = document.getElementById('topk-select').value;
        const applyLn = document.getElementById('ln-toggle').checked;
        
        const lensUrl = `/api/analysis/logit-lens?session_id=${currentSessionData.session_id}&top_k=${topK}&apply_ln=${applyLn}`;
        const lensRes = await fetch(lensUrl);
        if (!lensRes.ok) {
            const errJson = await lensRes.json();
            throw new Error(errJson.detail || 'Logit Lens extraction failed');
        }

        currentMatrixData = await lensRes.json();

        // Render Heatmap Matrix
        renderHeatmapMatrix(currentMatrixData);

        // Select first position by default for detail view
        if (currentMatrixData.positions && currentMatrixData.positions.length > 0) {
            renderPositionDetail(0);
        }

        fetchHealthStatus();

    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        runBtn.disabled = false;
        spinner.classList.add('hidden');
        runBtnText.textContent = '🚀 Run Analysis';
    }
}

// --- Render Tokens Strip ---
function renderTokensStrip(tokens) {
    const container = document.getElementById('tokens-strip');
    container.innerHTML = '';

    tokens.forEach((tokenStr, idx) => {
        const chip = document.createElement('div');
        chip.className = `token-chip ${idx === 0 ? 'selected' : ''}`;
        chip.innerHTML = `<strong>${idx}:</strong> ${escapeHtml(tokenStr)}`;
        chip.addEventListener('click', () => {
            document.querySelectorAll('.token-chip').forEach(c => c.classList.remove('selected'));
            chip.classList.add('selected');
            renderPositionDetail(idx);
        });
        container.appendChild(chip);
    });
}

// --- Render Logit Lens Heatmap Grid ---
function renderHeatmapMatrix(matrixData) {
    const container = document.getElementById('matrix-container');
    container.innerHTML = '';

    if (!matrixData.positions || matrixData.positions.length === 0) {
        container.innerHTML = '<div class="empty-state">No prediction data available.</div>';
        return;
    }

    const table = document.createElement('table');
    table.className = 'matrix-table';

    // Table Header (Positions / Tokens)
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    const cornerTh = document.createElement('th');
    cornerTh.textContent = 'Layer';
    headerRow.appendChild(cornerTh);

    matrixData.positions.forEach(posData => {
        const th = document.createElement('th');
        th.innerHTML = `Pos ${posData.position}<br><span style="color: var(--accent-primary); font-family: var(--font-mono);">${escapeHtml(posData.token)}</span>`;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Table Body (Layers 0..L)
    const tbody = document.createElement('tbody');
    const numLayers = matrixData.num_layers;

    for (let l = 0; l < numLayers; l++) {
        const tr = document.createElement('tr');
        
        // Layer label
        const layerTd = document.createElement('td');
        layerTd.style.fontWeight = 'bold';
        layerTd.style.fontSize = '11px';
        layerTd.style.color = 'var(--text-muted)';
        layerTd.textContent = l === 0 ? 'Embed' : `L${l-1}`;
        tr.appendChild(layerTd);

        // Position cells
        matrixData.positions.forEach(posData => {
            const td = document.createElement('td');
            td.className = 'matrix-cell';
            
            const layerRes = posData.layers[l];
            const top1 = layerRes && layerRes.top_tokens ? layerRes.top_tokens[0] : null;

            if (top1) {
                const prob = top1.probability;
                const pct = (prob * 100).toFixed(1);

                // Heatmap Color Calculation
                let bgStyle = '';
                if (prob > 0.6) {
                    bgStyle = `background-color: var(--cell-bg-high); color: #ffffff;`;
                } else if (prob > 0.2) {
                    bgStyle = `background-color: var(--cell-bg-mid); color: var(--text-primary);`;
                } else {
                    bgStyle = `background-color: var(--cell-bg-low); color: var(--text-secondary);`;
                }

                td.style = bgStyle;
                td.innerHTML = `
                    <div class="cell-token">${escapeHtml(top1.token)}</div>
                    <div class="cell-prob">${pct}%</div>
                `;

                td.addEventListener('click', () => {
                    renderPositionDetail(posData.position);
                });
            }
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    }

    table.appendChild(tbody);
    container.appendChild(table);
}

// --- Render Position Detail Drilldown ---
function renderPositionDetail(posIdx) {
    if (!currentMatrixData || !currentMatrixData.positions[posIdx]) return;

    const posData = currentMatrixData.positions[posIdx];

    const detailCard = document.getElementById('detail-card');
    detailCard.style.display = 'block';

    document.getElementById('detail-title').textContent = `Token Position ${posIdx}: "${posData.token}"`;
    document.getElementById('detail-subtitle').textContent = `Layer-by-layer top prediction trajectory`;

    // Populate Table
    const tbody = document.querySelector('#drilldown-table tbody');
    tbody.innerHTML = '';

    const layerLabels = [];
    const top1Probs = [];

    posData.layers.forEach((layerRes, lIdx) => {
        const lLabel = lIdx === 0 ? 'Embed' : `Layer ${lIdx-1}`;
        layerLabels.push(lLabel);

        const tr = document.createElement('tr');
        
        let rowHtml = `<td><strong>${lLabel}</strong></td>`;
        
        const top1 = layerRes.top_tokens[0];
        top1Probs.push(top1 ? top1.probability : 0);

        for (let i = 0; i < 3; i++) {
            const tok = layerRes.top_tokens[i];
            if (tok) {
                rowHtml += `
                    <td><code style="color: var(--accent-primary);">${escapeHtml(tok.token)}</code></td>
                    <td>${(tok.probability * 100).toFixed(1)}%</td>
                `;
            } else {
                rowHtml += `<td>-</td><td>-</td>`;
            }
        }

        tr.innerHTML = rowHtml;
        tbody.appendChild(tr);
    });

    // Render Chart.js Line Chart
    renderChart(layerLabels, top1Probs, posData.token);
}

function renderChart(labels, probs, tokenName) {
    const ctx = document.getElementById('prob-chart').getContext('2d');
    
    if (probChart) {
        probChart.destroy();
    }

    const currentTheme = document.documentElement.getAttribute('data-theme');
    const isDark = currentTheme === 'dark';

    const lineColor = isDark ? '#3b82f6' : '#2563eb';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const textColor = isDark ? '#9ca3af' : '#475569';

    probChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `Top #1 Prediction Probability`,
                data: probs.map(p => (p * 100).toFixed(1)),
                borderColor: lineColor,
                backgroundColor: isDark ? 'rgba(59, 130, 246, 0.15)' : 'rgba(37, 99, 235, 0.15)',
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    title: { display: true, text: 'Probability (%)', color: textColor },
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                },
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                }
            },
            plugins: {
                legend: { labels: { color: textColor } }
            }
        }
    });
}

function updateChartTheme(theme) {
    if (currentMatrixData && currentMatrixData.positions) {
        const selectedChip = document.querySelector('.token-chip.selected');
        const posIdx = selectedChip ? parseInt(selectedChip.textContent) : 0;
        renderPositionDetail(isNaN(posIdx) ? 0 : posIdx);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
