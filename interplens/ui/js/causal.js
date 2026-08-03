// InterpLens Studio Automated Causal Interventions & ROME Causal Tracing Visualizer

window.currentCausalData = null;

async function executeCausalPatchingSweep() {
    const cleanInput = document.getElementById('causal-clean-input');
    const corruptInput = document.getElementById('causal-corrupt-input');
    const targetInput = document.getElementById('causal-target-input');

    if (!cleanInput || !corruptInput) return;

    const cleanPrompt = cleanInput.value.trim();
    const corruptPrompt = corruptInput.value.trim();
    const targetToken = targetInput ? targetInput.value.trim() : null;

    if (!cleanPrompt || !corruptPrompt) {
        alert('Please provide both Clean and Corrupted prompt strings.');
        return;
    }

    const runBtn = document.getElementById('causal-run-btn');
    if (runBtn) runBtn.disabled = true;

    try {
        const data = await window.API.runCausalPatching(cleanPrompt, corruptPrompt, targetToken);
        window.currentCausalData = data;
        renderCausalPatchingResults(data);
    } catch (err) {
        alert(`Causal Patching Error: ${err.message}`);
    } finally {
        if (runBtn) runBtn.disabled = false;
    }
}

function loadCausalPreset(presetType) {
    const cleanInput = document.getElementById('causal-clean-input');
    const corruptInput = document.getElementById('causal-corrupt-input');
    const targetInput = document.getElementById('causal-target-input');

    if (!cleanInput || !corruptInput) return;

    if (presetType === 'eiffel') {
        cleanInput.value = "The Eiffel Tower is located in the city of";
        corruptInput.value = "The Colosseum is located in the city of";
        if (targetInput) targetInput.value = "Paris";
    } else if (presetType === 'gender') {
        cleanInput.value = "The doctor said that he was going to";
        corruptInput.value = "The doctor said that she was going to";
        if (targetInput) targetInput.value = "he";
    } else if (presetType === 'plural') {
        cleanInput.value = "The keys to the cabinet are on the";
        corruptInput.value = "The key to the cabinet is on the";
        if (targetInput) targetInput.value = "are";
    }

    executeCausalPatchingSweep();
}

function renderCausalPatchingResults(data) {
    if (!data) return;

    // 1. Metric Readout Cards
    if (document.getElementById('causal-target-display')) {
        document.getElementById('causal-target-display').textContent = `'${escapeHtml(data.target_token)}'`;
    }
    if (document.getElementById('causal-clean-diff')) {
        document.getElementById('causal-clean-diff').textContent = data.baseline_clean_logit_diff.toFixed(3);
    }
    if (document.getElementById('causal-corrupt-diff')) {
        document.getElementById('causal-corrupt-diff').textContent = data.baseline_corrupt_logit_diff.toFixed(3);
    }
    if (document.getElementById('causal-max-recovery')) {
        document.getElementById('causal-max-recovery').textContent = `${data.max_recovery_percentage.toFixed(1)}% (L${data.max_recovery_layer}, P${data.max_recovery_position})`;
    }

    // 2. Render Causal Tracing Heatmap Matrix (num_layers x seq_len)
    const container = document.getElementById('causal-heatmap-container');
    if (!container) return;

    const layers = data.num_layers;
    const tokens = data.clean_tokens;
    const seqLen = data.seq_len;
    const matrix = data.heatmap_matrix;

    let html = `
        <div style="margin-bottom: 12px; font-size: 13px; color: var(--text-muted);">
            Logit Difference Recovery Heatmap matrix (<strong>${layers} Layers × ${seqLen} Token Positions</strong>) for target prediction <strong>'${escapeHtml(data.target_token)}'</strong>:
        </div>
        <div class="table-scroll" style="overflow-x: auto; padding: 10px; background: var(--bg-darkest); border-radius: 8px; border: 1px solid var(--border-color);">
            <table style="border-collapse: collapse; margin: 0 auto; font-family: var(--font-mono); font-size: 11px;">
                <thead>
                    <tr>
                        <th style="padding: 6px 10px; text-align: right; color: var(--text-muted);">Layer</th>
    `;

    tokens.forEach((tok, p) => {
        html += `
            <th style="padding: 6px 8px; text-align: center; color: var(--primary); min-width: 65px; border-bottom: 1px solid var(--border-color);">
                <div style="font-weight: 700;">${escapeHtml(tok)}</div>
                <div style="font-size: 9px; opacity: 0.7;">#${p}</div>
            </th>
        `;
    });

    html += `</tr></thead><tbody>`;

    for (let l = 0; l < layers; l++) {
        html += `
            <tr>
                <td style="padding: 6px 10px; font-weight: 700; color: var(--text-muted); text-align: right; border-right: 1px solid var(--border-color);">L${l}</td>
        `;

        for (let p = 0; p < seqLen; p++) {
            const recovery = matrix[l][p];
            const isMax = l === data.max_recovery_layer && p === data.max_recovery_position;

            // Color gradient from dark slate blue to glowing red/amber
            let bg, textColor;
            if (recovery <= 5) {
                bg = 'rgba(30, 41, 59, 0.6)';
                textColor = 'var(--text-muted)';
            } else if (recovery <= 30) {
                bg = 'rgba(56, 189, 248, 0.35)';
                textColor = 'var(--text-primary)';
            } else if (recovery <= 70) {
                bg = 'rgba(245, 158, 11, 0.7)';
                textColor = '#ffffff';
            } else {
                bg = 'rgba(239, 68, 68, 0.85)';
                textColor = '#ffffff';
            }

            const borderStyle = isMax ? 'border: 2px solid #38bdf8; transform: scale(1.05);' : 'border: 1px solid rgba(255,255,255,0.05);';

            html += `
                <td style="padding: 8px 6px; text-align: center; background: ${bg}; color: ${textColor}; ${borderStyle} cursor: pointer; border-radius: 3px; transition: transform 0.15s ease;"
                    title="Layer L${l}, Token '${tokens[p]}' (#${p}): Logit Diff Recovery = ${recovery.toFixed(1)}%"
                    onclick="inspectCausalCell(${l}, ${p})">
                    <span style="font-weight: 700;">${recovery.toFixed(0)}%</span>
                </td>
            `;
        }

        html += `</tr>`;
    }

    html += `</tbody></table></div>`;
    container.innerHTML = html;
}

function inspectCausalCell(layer, position) {
    if (!window.currentCausalData) return;

    const data = window.currentCausalData;
    const cell = data.cells.find(c => c.layer === layer && c.position === position);
    if (!cell) return;

    const detailContainer = document.getElementById('causal-cell-detail');
    if (!detailContainer) return;

    detailContainer.innerHTML = `
        <div style="display: flex; gap: 16px; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <span style="font-size: 11px; color: var(--text-muted);">Selected Intervention Cell:</span>
                <strong style="color: var(--accent-green); margin-left: 6px;">Layer L${cell.layer} • Token #${cell.position} ('${escapeHtml(cell.clean_token)}')</strong>
            </div>
            <div style="display: flex; gap: 16px; font-family: var(--font-mono); font-size: 12px;">
                <span>Clean Token: <strong>'${escapeHtml(cell.clean_token)}'</strong></span>
                <span>Corrupt Token: <strong>'${escapeHtml(cell.corrupt_token)}'</strong></span>
                <span>Logit Diff Recovery: <strong style="color: var(--accent-green);">${cell.logit_diff_recovery.toFixed(2)}%</strong></span>
            </div>
        </div>
    `;
}

window.executeCausalPatchingSweep = executeCausalPatchingSweep;
window.loadCausalPreset = loadCausalPreset;
window.inspectCausalCell = inspectCausalCell;
