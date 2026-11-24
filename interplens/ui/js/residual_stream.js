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

const CAA_PRESETS = {
    sentiment: {
        name: 'Positive vs Negative',
        pos: [
            'I am so happy and grateful for this wonderful day!',
            'Everything is going amazingly well and I feel joyful.',
            'What a brilliant, inspiring, and uplifting experience!',
        ].join('\n'),
        neg: [
            'I am miserable and frustrated by this terrible day.',
            'Everything is going horribly wrong and I feel awful.',
            'What a depressing, disappointing, and dreadful experience.',
        ].join('\n'),
        prompt: 'Today I walked outside and thought to myself,',
    },
    formal: {
        name: 'Formal vs Casual',
        pos: [
            'Dear Colleagues, I am writing to formally present the quarterly findings.',
            'Pursuant to our previous correspondence, please find the enclosed documentation.',
            'It is imperative that we evaluate these empirical results with rigor.',
        ].join('\n'),
        neg: [
            'Hey guys, just dropping a quick note about what we found lol.',
            'Yo, check out this stuff I threw together real quick.',
            'Honestly this whole thing is super chill and kinda wild.',
        ].join('\n'),
        prompt: 'Regarding the project update, I wanted to say that',
    },
    science: {
        name: 'Science vs Poetry',
        pos: [
            'The gravitational force between two masses is proportional to the product of their masses.',
            'Mitochondria synthesize adenosine triphosphate through oxidative phosphorylation.',
            'Quantum entanglement exhibits non-local correlations between particle spin states.',
        ].join('\n'),
        neg: [
            'Beneath the silver veil of twilight, the whispering river sang to the moon.',
            'Golden petals drifted softly across the quiet garden of forgotten dreams.',
            'The autumn wind painted melodies of amber and crimson across the sky.',
        ].join('\n'),
        prompt: 'When we look up at the stars at night,',
    },
};

var activeSteeringVector = null;

function _populateSteeringLayers(numLayers) {
    const sel = document.getElementById('steer-layer-select');
    if (!sel || !numLayers || numLayers <= 0) return;
    const prev = sel.value;
    sel.innerHTML = '';
    const defaultLayer = Math.min(numLayers - 1, Math.max(0, Math.floor(numLayers / 2)));
    for (let l = 0; l < numLayers; l++) {
        const opt = document.createElement('option');
        opt.value = String(l);
        opt.textContent = `Layer ${l}`;
        if (String(l) === prev || (!prev && l === defaultLayer)) opt.selected = true;
        sel.appendChild(opt);
    }
}

async function computeContrastiveVectorUI() {
    const posRaw = (document.getElementById('caa-pos-prompts') || {}).value || '';
    const negRaw = (document.getElementById('caa-neg-prompts') || {}).value || '';
    const targetLayer = parseInt((document.getElementById('steer-layer-select') || {}).value, 10) || 0;
    const statusEl = document.getElementById('caa-status');
    const badgeEl = document.getElementById('caa-vector-badge');
    const btn = document.getElementById('caa-compute-btn');

    const posList = posRaw.split('\n').map(s => s.trim()).filter(Boolean);
    const negList = negRaw.split('\n').map(s => s.trim()).filter(Boolean);
    if (!posList.length || !negList.length) {
        if (statusEl) statusEl.textContent = 'Enter at least one positive prompt and one negative prompt.';
        return null;
    }

    if (btn) { btn.disabled = true; btn.textContent = 'Computing...'; }
    if (statusEl) statusEl.textContent = `Extracting Layer ${targetLayer} residuals across ${posList.length + negList.length} prompts...`;

    try {
        const vecInfo = await window.API.computeSteeringVector(posList, negList, targetLayer, `L${targetLayer} CAA Vector`, false);
        activeSteeringVector = vecInfo;
        if (badgeEl) badgeEl.textContent = `Vector #${vecInfo.vector_id} • L${vecInfo.target_layer} (‖v‖=${vecInfo.l2_norm})`;
        if (statusEl) statusEl.textContent = `Computed Difference-in-Means vector at Layer ${vecInfo.target_layer} (L2 norm: ${vecInfo.l2_norm}, Pos/Neg cos: ${vecInfo.pos_neg_cosine}).`;
        return vecInfo;
    } catch (err) {
        if (statusEl) statusEl.textContent = `Vector error: ${err.message}`;
        return null;
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '1. Compute Vector'; }
    }
}

async function executeActivationSteering() {
    const prompt = ((document.getElementById('caa-test-prompt') || {}).value || '').trim()
        || ((document.getElementById('prompt-input') || {}).value || '').trim()
        || 'Today I walked outside and thought to myself,';
    const targetLayer = parseInt((document.getElementById('steer-layer-select') || {}).value, 10) || 0;
    const multiplier = parseFloat((document.getElementById('steer-multiplier') || {}).value);
    const alpha = isNaN(multiplier) ? 2.0 : multiplier;
    const maxTokens = parseInt((document.getElementById('caa-max-tokens') || {}).value, 10) || 20;
    const statusEl = document.getElementById('caa-status');
    const btn = document.getElementById('steer-run-btn');

    if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }

    try {
        // Auto-compute vector if not yet computed or layer changed
        if (!activeSteeringVector || activeSteeringVector.target_layer !== targetLayer) {
            const vec = await computeContrastiveVectorUI();
            if (!vec) return;
        }

        if (statusEl) statusEl.textContent = `Generating ${maxTokens} tokens (Baseline vs Steered α=${alpha} at Layer ${targetLayer})...`;

        const res = await window.API.generateSteeredComparison(
            prompt,
            activeSteeringVector.vector_id,
            targetLayer,
            alpha,
            maxTokens,
            0.0
        );

        const card = document.getElementById('steer-result-card');
        if (card) card.style.display = 'block';

        const safePrompt = prompt.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const safeBase = (res.baseline_completion || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const safeSteer = (res.steered_completion || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        const baseEl = document.getElementById('caa-baseline-out');
        if (baseEl) baseEl.innerHTML = `<span style="opacity: 0.65;">${safePrompt}</span><strong style="color: var(--text-main);">${safeBase}</strong>`;

        const steerEl = document.getElementById('caa-steered-out');
        if (steerEl) steerEl.innerHTML = `<span style="opacity: 0.65;">${safePrompt}</span><strong style="color: #38bdf8;">${safeSteer}</strong>`;

        if (document.getElementById('caa-runtime-badge')) document.getElementById('caa-runtime-badge').textContent = `${res.runtime_ms} ms`;
        if (document.getElementById('caa-steered-alpha-lbl')) document.getElementById('caa-steered-alpha-lbl').textContent = `α = ${alpha >= 0 ? '+' : ''}${alpha}`;
        if (document.getElementById('caa-steered-layer-lbl')) document.getElementById('caa-steered-layer-lbl').textContent = `Layer ${res.target_layer}`;
        if (statusEl) statusEl.textContent = `Generated ${res.baseline_tokens.length} baseline / ${res.steered_tokens.length} steered tokens in ${res.runtime_ms} ms.`;
    } catch (err) {
        if (statusEl) statusEl.textContent = `Steering error: ${err.message}`;
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '2. Generate Baseline vs Steered'; }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const computeBtn = document.getElementById('caa-compute-btn');
    if (computeBtn) computeBtn.addEventListener('click', computeContrastiveVectorUI);

    const steerBtn = document.getElementById('steer-run-btn');
    if (steerBtn) steerBtn.addEventListener('click', executeActivationSteering);

    const alphaSlider = document.getElementById('steer-multiplier');
    const alphaLbl = document.getElementById('steer-alpha-val');
    if (alphaSlider && alphaLbl) {
        alphaSlider.addEventListener('input', () => {
            const v = parseFloat(alphaSlider.value);
            alphaLbl.textContent = `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
        });
    }

    document.querySelectorAll('.caa-preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.getAttribute('data-preset');
            const preset = CAA_PRESETS[key];
            if (!preset) return;
            if (document.getElementById('caa-pos-prompts')) document.getElementById('caa-pos-prompts').value = preset.pos;
            if (document.getElementById('caa-neg-prompts')) document.getElementById('caa-neg-prompts').value = preset.neg;
            if (document.getElementById('caa-test-prompt')) document.getElementById('caa-test-prompt').value = preset.prompt;
            activeSteeringVector = null;
            if (document.getElementById('caa-vector-badge')) document.getElementById('caa-vector-badge').textContent = `Preset: ${preset.name}`;
        });
    });
});

window.fetchResidualStreamMetrics = async function (sessionId, position = null) {
    if (!sessionId) return;
    try {
        const res = await window.API.getResidualStreamMetrics(sessionId, position);
        if (res.layer_labels && res.layer_labels.length > 1) {
            const nLayers = res.layer_labels.filter(l => l.startsWith('L')).length || res.layer_labels.length;
            _populateSteeringLayers(nLayers);
        }
        renderNormsChart(res);
        renderLayerCosChart(res);
        renderCosineMatrixGrid(res);
    } catch (err) {
        console.warn('Residual Stream metrics error:', err);
    }
};
window.executeActivationSteering = executeActivationSteering;
window.computeContrastiveVectorUI = computeContrastiveVectorUI;
