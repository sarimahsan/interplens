// InterpLens Studio Attention Head Explorer & Arc Visualizer Engine

window.currentAttnData = null;
window.activeAttnView = 'matrix'; // 'matrix' | 'grid' | 'arc'

async function fetchAttentionHeadsData(layer = 0, head = 0, threshold = 0.02) {
    const sessId = (window.currentSession && window.currentSession.session_id) || '';
    if (!sessId) return;

    try {
        const data = await window.API.getAttentionHeads(sessId, layer, head, threshold);
        window.currentAttnData = data;
        renderAttentionExplorer(data);
    } catch (err) {
        console.error('Failed to fetch attention heads:', err);
    }
}

function renderAttentionExplorer(data) {
    if (!data) return;

    // Update controls metadata
    const layerSelect = document.getElementById('attn-layer-select');
    const headSelect = document.getElementById('attn-head-select');

    if (layerSelect && layerSelect.options.length !== data.num_layers) {
        layerSelect.innerHTML = '';
        for (let l = 0; l < data.num_layers; l++) {
            const opt = document.createElement('option');
            opt.value = l;
            opt.textContent = `Layer ${l}`;
            if (l === data.layer) opt.selected = true;
            layerSelect.appendChild(opt);
        }
    }

    if (headSelect && headSelect.options.length !== data.num_heads) {
        headSelect.innerHTML = '';
        for (let h = 0; h < data.num_heads; h++) {
            const opt = document.createElement('option');
            opt.value = h;
            opt.textContent = `Head ${h}`;
            if (h === data.head) opt.selected = true;
            headSelect.appendChild(opt);
        }
    }

    // Render active view mode
    if (window.activeAttnView === 'grid') {
        renderMultiHeadGrid(data);
    } else if (window.activeAttnView === 'arc') {
        renderArcDiagram(data);
    } else {
        renderDetailMatrix(data);
    }
}

function renderDetailMatrix(data) {
    const container = document.getElementById('attn-matrix-container');
    const gridView = document.getElementById('attn-grid-view');
    const arcView = document.getElementById('attn-arc-view');

    if (container) container.style.display = 'block';
    if (gridView) gridView.style.display = 'none';
    if (arcView) arcView.style.display = 'none';

    if (!container || !data.matrix) return;

    const tokens = data.tokens || [];
    const matrix = data.matrix;
    const n = tokens.length;

    let html = `
        <div style="margin-bottom: 12px; font-size: 13px; color: var(--text-muted);">
            Layer <strong>L${data.layer}</strong> • Head <strong>H${data.head}</strong> • $N \\times N$ Attention Matrix ($N=${n}$)
        </div>
        <div class="matrix-grid-wrapper" style="overflow-x: auto; max-width: 100%;">
            <table class="studio-table" style="border-collapse: collapse; font-family: var(--font-mono); font-size: 11px;">
                <thead>
                    <tr>
                        <th style="padding: 6px; background: var(--bg-darkest); color: var(--text-muted); position: sticky; left: 0; z-index: 2;">Query \\ Key</th>
                        ${tokens.map((tok, idx) => `<th style="padding: 6px; text-align: center; background: var(--bg-card);"><span style="color: var(--accent-cyan);">${escapeHtml(tok)}</span><br><span style="opacity: 0.6; font-size: 9px;">#${idx}</span></th>`).join('')}
                    </tr>
                </thead>
                <tbody>
    `;

    for (let i = 0; i < n; i++) {
        html += `<tr>`;
        html += `<td style="padding: 6px; font-weight: 600; background: var(--bg-card); color: var(--accent-purple); position: sticky; left: 0; z-index: 1;"><span style="color: var(--text-primary);">${escapeHtml(tokens[i])}</span> <span style="opacity: 0.6; font-size: 9px;">#${i}</span></td>`;
        
        for (let j = 0; j < n; j++) {
            const val = matrix[i][j];
            const alpha = Math.min(1.0, val * 1.5);
            const bg = `rgba(168, 85, 247, ${alpha.toFixed(2)})`; // Accent purple heat
            const color = alpha > 0.4 ? '#FFFFFF' : 'var(--text-muted)';
            html += `<td style="padding: 8px 6px; text-align: center; background: ${bg}; color: ${color}; font-weight: ${val > 0.3 ? '700' : '400'}; border: 1px solid rgba(255,255,255,0.05);" title="Q[${i}] '${tokens[i]}' -> K[${j}] '${tokens[j]}': ${(val * 100).toFixed(1)}%">
                ${val > 0.001 ? val.toFixed(3) : '0'}
            </td>`;
        }
        html += `</tr>`;
    }

    html += `</tbody></table></div>`;
    container.innerHTML = html;
}

function renderMultiHeadGrid(data) {
    const container = document.getElementById('attn-matrix-container');
    const gridView = document.getElementById('attn-grid-view');
    const arcView = document.getElementById('attn-arc-view');

    if (container) container.style.display = 'none';
    if (arcView) arcView.style.display = 'none';
    if (gridView) gridView.style.display = 'grid';

    if (!gridView || !data.grid) return;

    const grid = data.grid;
    let html = '';

    grid.forEach((headMat, hIdx) => {
        const isSelected = hIdx === data.head;
        const borderStyle = isSelected ? 'border: 2px solid var(--accent-purple); background: var(--bg-hover);' : 'border: 1px solid var(--border-color); background: var(--bg-card);';
        
        html += `
            <div class="head-card" style="border-radius: 6px; padding: 10px; cursor: pointer; ${borderStyle}" onclick="selectAttnHead(${hIdx})">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 12px; font-weight: 600; color: ${isSelected ? 'var(--accent-purple)' : 'var(--text-primary)'};">
                    <span>Head ${hIdx}</span>
                    ${isSelected ? '<span class="status-tag" style="background: var(--accent-purple); color: #fff;">Selected</span>' : ''}
                </div>
                <div style="display: grid; grid-template-columns: repeat(${data.tokens.length}, 1fr); gap: 1px; background: rgba(0,0,0,0.3); padding: 2px; border-radius: 3px;">
        `;

        const n = data.tokens.length;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const val = headMat[i][j];
                const alpha = Math.min(1.0, val * 1.5);
                html += `<div style="aspect-ratio: 1; background: rgba(168, 85, 247, ${alpha.toFixed(2)});" title="H${hIdx} Q[${i}]->K[${j}]: ${val.toFixed(2)}"></div>`;
            }
        }

        html += `</div></div>`;
    });

    gridView.innerHTML = html;
}

function renderArcDiagram(data) {
    const container = document.getElementById('attn-matrix-container');
    const gridView = document.getElementById('attn-grid-view');
    const arcView = document.getElementById('attn-arc-view');

    if (container) container.style.display = 'none';
    if (gridView) gridView.style.display = 'none';
    if (arcView) arcView.style.display = 'block';

    if (!arcView) return;

    const canvas = document.getElementById('attn-arc-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const tokens = data.tokens || [];
    const links = data.arc_links || [];
    const n = tokens.length;

    // Resize canvas
    const width = arcView.clientWidth || 800;
    const height = 340;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    ctx.clearRect(0, 0, width, height);

    if (n === 0) return;

    const padding = 60;
    const spacing = (width - padding * 2) / Math.max(1, n - 1);
    const baselineY = height - 60;

    // Draw arcs
    links.forEach(link => {
        const x1 = padding + link.source * spacing;
        const x2 = padding + link.target * spacing;
        const dist = Math.abs(x2 - x1);
        const arcHeight = Math.min(baselineY - 40, dist * 0.45);

        ctx.beginPath();
        ctx.moveTo(x1, baselineY);

        const controlY = baselineY - arcHeight;
        ctx.quadraticCurveTo((x1 + x2) / 2, controlY, x2, baselineY);

        const weight = link.weight;
        ctx.strokeStyle = `rgba(168, 85, 247, ${Math.min(1.0, weight * 2.0).toFixed(2)})`;
        ctx.lineWidth = Math.max(1, weight * 8);
        ctx.stroke();
    });

    // Draw token nodes along baseline
    tokens.forEach((tok, idx) => {
        const x = padding + idx * spacing;

        // Node dot
        ctx.beginPath();
        ctx.arc(x, baselineY, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#38BDF8';
        ctx.fill();

        // Label
        ctx.font = '11px monospace';
        ctx.fillStyle = '#F1F5F9';
        ctx.textAlign = 'center';
        ctx.fillText(tok, x, baselineY + 20);

        ctx.fillStyle = '#94A3B8';
        ctx.font = '9px monospace';
        ctx.fillText(`#${idx}`, x, baselineY + 34);
    });
}

function selectAttnHead(headIdx) {
    const headSelect = document.getElementById('attn-head-select');
    if (headSelect) headSelect.value = headIdx;

    const layer = parseInt(document.getElementById('attn-layer-select').value) || 0;
    const thresh = parseFloat(document.getElementById('attn-thresh-select').value) || 0.02;

    fetchAttentionHeadsData(layer, headIdx, thresh);
}

function setAttnViewMode(mode) {
    window.activeAttnView = mode;

    document.querySelectorAll('.btn-attn-view').forEach(btn => {
        if (btn.getAttribute('data-view') === mode) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    if (window.currentAttnData) {
        renderAttentionExplorer(window.currentAttnData);
    }
}

window.fetchAttentionHeadsData = fetchAttentionHeadsData;
window.selectAttnHead = selectAttnHead;
window.setAttnViewMode = setAttnViewMode;
