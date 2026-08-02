// Logit Lens Analysis Engine Frontend Module

let currentMatrix = null;
let detailChart = null;
let activeMetric = 'prob';

function formatTokenStr(str) {
    if (!str) return '∅';
    return str.replace(/\n/g, '↵').replace(/\t/g, '⇥');
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
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
        layerLbl.textContent = l === 0 ? 'Embedding' : `Layer ${l - 1}`;
        grid.appendChild(layerLbl);

        positions.forEach(p => {
            const layerData = p.layers[l];
            const topTok = layerData.top_tokens[0] || { token: '?', probability: 0 };
            
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.style.backgroundColor = getProbHeatmapColor(topTok.probability);
            cell.innerHTML = `
                <span class="cell-tok">${escapeHtml(formatTokenStr(topTok.token))}</span>
                <span class="cell-prob">${(topTok.probability * 100).toFixed(1)}%</span>
            `;

            cell.addEventListener('mouseenter', (e) => showMatrixTooltip(e, p, l, layerData));
            cell.addEventListener('mousemove', (e) => positionMatrixTooltip(e));
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
    document.getElementById('resp-top-prob').textContent = `${(topPredictions[0].probability * 100).toFixed(1)}%`;
    document.getElementById('resp-top-logit').textContent = topPredictions[0].logit !== null ? topPredictions[0].logit.toFixed(2) : 'N/A';

    const tbody = document.getElementById('resp-top-tbody');
    if (tbody) {
        tbody.innerHTML = '';
        topPredictions.forEach((t, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${i + 1}</td>
                <td style="font-weight:700; color:var(--primary);">${escapeHtml(formatTokenStr(t.token))}</td>
                <td>${(t.probability * 100).toFixed(2)}%</td>
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
        const lName = lIdx === 0 ? 'Embed' : `Layer ${lIdx - 1}`;
        const top1 = lData.top_tokens[0] || { token: '?', probability: 0 };
        const top2 = lData.top_tokens[1] || { token: '?', probability: 0 };

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight:600;">${lName}</td>
            <td style="font-family:var(--font-mono); font-weight:700; color:var(--primary);">${escapeHtml(formatTokenStr(top1.token))}</td>
            <td>${(top1.probability * 100).toFixed(1)}%</td>
            <td style="font-family:var(--font-mono); color:var(--text-sub);">${escapeHtml(formatTokenStr(top2.token))}</td>
            <td>${(top2.probability * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

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
    const rect = tooltip.getBoundingClientRect();
    let x = e.clientX + 8;
    let y = e.clientY + 8;

    if (x + rect.width > window.innerWidth - 10) x = e.clientX - rect.width - 8;
    if (y + rect.height > window.innerHeight - 10) y = e.clientY - rect.height - 8;

    tooltip.style.left = `${Math.max(5, x)}px`;
    tooltip.style.top = `${Math.max(5, y)}px`;
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
    const data = pos.layers.map(l => (l.top_tokens[0].probability * 100));

    const canvas = document.getElementById('detail-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (detailChart) detailChart.destroy();

    detailChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Top-1 Probability (%)',
                data,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
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
                x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, min: 0, max: 100 }
            },
            plugins: { legend: { display: false } }
        }
    });

    renderTopPredictionsTable(currentMatrix);
}

window.renderTokensFlow = renderTokensFlow;
window.renderMatrixGrid = renderMatrixGrid;
window.renderModelResponseCard = renderModelResponseCard;
window.renderInspectionDetail = renderInspectionDetail;
