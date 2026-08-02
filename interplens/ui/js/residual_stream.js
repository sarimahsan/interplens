// Phase 3: Residual Stream Vector Analysis & Steering Controller

var normChart = null;
var cosChart = null;

async function fetchResidualStreamMetrics(sessionId, position = null) {
    if (!sessionId) return;

    try {
        const res = await window.API.getResidualStreamMetrics(sessionId, position);
        renderNormsChart(res);
        renderLayerCosChart(res);
        renderCosineMatrixGrid(res);
    } catch (err) {
        console.warn('Residual Stream metrics error:', err);
    }
}

function renderNormsChart(data) {
    const canvas = document.getElementById('resid-norm-chart');
    if (!canvas || !data.position_history) return;
    const ctx = canvas.getContext('2d');

    const labels = data.position_history.map(p => p.layer_name);
    const norms = data.position_history.map(p => p.norm);

    if (normChart) normChart.destroy();

    normChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: `L2 Vector Norm ||xₗ|| (Pos: "${data.selected_token}")`,
                data: norms,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'L2 Vector Norm', color: '#94a3b8', font: { size: 10 } } }
            },
            plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }
        }
    });
}

function renderLayerCosChart(data) {
    const canvas = document.getElementById('resid-cos-chart');
    if (!canvas || !data.layer_cosine_transitions) return;
    const ctx = canvas.getContext('2d');

    const labels = data.layer_cosine_transitions.map(t => `${t.from_layer} → ${t.to_layer}`);
    const cosVals = data.layer_cosine_transitions.map(t => t.similarities[data.selected_position || 0]);

    if (cosChart) cosChart.destroy();

    cosChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Consecutive Layer Cosine Similarity',
                data: cosVals,
                backgroundColor: 'rgba(6, 182, 212, 0.7)',
                borderColor: '#06b6d4',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { min: -1.0, max: 1.0, ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Cosine Similarity', color: '#94a3b8', font: { size: 10 } } }
            },
            plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }
        }
    });
}

function renderCosineMatrixGrid(data) {
    const container = document.getElementById('resid-cos-matrix-container');
    if (!container || !data.cosine_matrix) return;
    container.innerHTML = '';

    const labels = data.layer_labels;
    const matrix = data.cosine_matrix;

    const grid = document.createElement('div');
    grid.className = 'matrix-grid';
    grid.style.gridTemplateColumns = `80px repeat(${labels.length}, minmax(60px, 1fr))`;

    const cornerHeader = document.createElement('div');
    cornerHeader.className = 'grid-header-cell';
    cornerHeader.textContent = 'L / L';
    grid.appendChild(cornerHeader);

    labels.forEach(l => {
        const hdr = document.createElement('div');
        hdr.className = 'grid-header-cell';
        hdr.textContent = l;
        grid.appendChild(hdr);
    });

    for (let r = 0; r < labels.length; r++) {
        const lbl = document.createElement('div');
        lbl.className = 'grid-layer-label';
        lbl.textContent = labels[r];
        grid.appendChild(lbl);

        for (let c = 0; c < labels.length; c++) {
            const val = matrix[r][c];
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            const bgAlpha = Math.max(0.1, Math.abs(val));
            cell.style.backgroundColor = val >= 0 ? `rgba(6, 182, 212, ${bgAlpha * 0.85})` : `rgba(239, 68, 68, ${bgAlpha * 0.85})`;
            cell.innerHTML = `
                <span class="cell-tok">${val.toFixed(2)}</span>
            `;
            grid.appendChild(cell);
        }
    }

    container.appendChild(grid);
}

async function executeActivationSteering() {
    const prompt = document.getElementById('prompt-input') ? document.getElementById('prompt-input').value.trim() : 'The Eiffel Tower is located in';
    const targetLayer = parseInt(document.getElementById('steer-layer-select').value) || 0;
    const multiplier = parseFloat(document.getElementById('steer-multiplier').value) || 1.0;

    const btn = document.getElementById('steer-run-btn');
    if (btn) btn.disabled = true;

    try {
        const res = await window.API.steerResidualStream(prompt, targetLayer, multiplier);
        const card = document.getElementById('steer-result-card');
        if (card) card.style.display = 'block';

        if (document.getElementById('steer-top-tok')) {
            const tokStr = typeof window.formatTokenStr === 'function' ? window.formatTokenStr(res.top_steered_token) : (res.top_steered_token || '--');
            document.getElementById('steer-top-tok').textContent = tokStr;
        }
        if (document.getElementById('steer-top-prob')) {
            document.getElementById('steer-top-prob').textContent = res.top_steered_prob !== undefined && res.top_steered_prob !== null ? `${res.top_steered_prob}%` : '--%';
        }
        if (document.getElementById('steer-status-tag')) {
            document.getElementById('steer-status-tag').textContent = `Status: ${res.status || 'OK'}${res.error ? ' (' + res.error + ')' : ''}`;
        }
    } catch (err) {
        alert(`Steering error: ${err.message}`);
    } finally {
        if (btn) btn.disabled = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const steerBtn = document.getElementById('steer-run-btn');
    if (steerBtn) steerBtn.addEventListener('click', executeActivationSteering);
});

window.fetchResidualStreamMetrics = fetchResidualStreamMetrics;
window.executeActivationSteering = executeActivationSteering;
