// Logit Lens Analysis Engine Frontend Module

var currentMatrix = window.currentMatrix || null;
var detailChart = window.detailChart || null;
var activeMetric = window.activeMetric || 'prob';

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
window.positionMatrixTooltip = positionMatrixTooltip;
window.hideMatrixTooltip = hideMatrixTooltip;
