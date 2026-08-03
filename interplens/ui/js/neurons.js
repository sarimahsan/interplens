// InterpLens Studio Neuron Activation & Token Attribution Explorer Engine

window.currentNeuronData = null;
window.currentAttributionData = null;

async function fetchNeuronData(layer = 0, position = null, topK = 10, neuronIdx = null) {
    const sessId = (window.currentSession && window.currentSession.session_id) || '';
    if (!sessId) return;

    try {
        const neuronRes = await window.API.getNeuronActivations(sessId, layer, position, topK, neuronIdx);
        window.currentNeuronData = neuronRes;
        renderNeuronExplorer(neuronRes);

        const attrRes = await window.API.getTokenAttribution(sessId, position);
        window.currentAttributionData = attrRes;
        renderTokenAttribution(attrRes);
    } catch (err) {
        console.error('Failed to fetch neuron data:', err);
    }
}

function renderNeuronExplorer(data) {
    if (!data) return;

    // Populate layer select dropdown
    const layerSelect = document.getElementById('neuron-layer-select');
    const numLayers = (window.currentSession && window.currentSession.model_info && window.currentSession.model_info.num_layers) || 12;

    if (layerSelect && layerSelect.options.length !== numLayers) {
        layerSelect.innerHTML = '';
        for (let l = 0; l < numLayers; l++) {
            const opt = document.createElement('option');
            opt.value = l;
            opt.textContent = `Layer ${l}`;
            if (l === data.layer) opt.selected = true;
            layerSelect.appendChild(opt);
        }
    }

    // Populate position select dropdown
    const posSelect = document.getElementById('neuron-pos-select');
    const tokens = data.tokens || [];
    if (posSelect && posSelect.options.length !== tokens.length) {
        posSelect.innerHTML = '';
        tokens.forEach((tok, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = `#${idx} '${tok}'`;
            if (idx === data.position) opt.selected = true;
            posSelect.appendChild(opt);
        });
    }

    // 1. Render Top Firing Neurons Bar Chart
    const barContainer = document.getElementById('neuron-bars-container');
    if (barContainer) {
        const topNeurons = data.top_neurons || [];
        const maxVal = topNeurons.length > 0 ? Math.max(1.0, Math.abs(topNeurons[0].activation)) : 1.0;

        let barHtml = `
            <div style="margin-bottom: 10px; font-size: 13px; color: var(--text-muted);">
                Top-Firing MLP Neurons at Layer <strong>L${data.layer}</strong> for Token <strong>'${escapeHtml(data.selected_token)}'</strong> (#${data.position})
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
        `;

        topNeurons.forEach((n, rank) => {
            const isSelected = n.neuron_idx === data.selected_neuron_idx;
            const pct = Math.min(100, Math.max(5, (Math.abs(n.activation) / maxVal) * 100));
            const barBg = isSelected ? 'var(--accent-green)' : 'rgba(52, 211, 153, 0.4)';
            const borderStyle = isSelected ? 'border: 1px solid var(--accent-green); background: rgba(52, 211, 153, 0.15);' : 'border: 1px solid var(--border-color); background: var(--bg-card);';

            barHtml += `
                <div class="neuron-bar-row" style="display: flex; align-items: center; gap: 12px; padding: 6px 10px; border-radius: 6px; cursor: pointer; ${borderStyle}" onclick="selectSingleNeuron(${n.neuron_idx})">
                    <span style="width: 24px; font-size: 11px; font-weight: 700; color: var(--text-muted);">#${rank + 1}</span>
                    <span style="width: 110px; font-size: 12px; font-weight: 600; font-family: var(--font-mono); color: ${isSelected ? 'var(--accent-green)' : 'var(--text-primary)'};">Neuron N${n.neuron_idx}</span>
                    <div style="flex: 1; height: 16px; background: rgba(0,0,0,0.3); border-radius: 3px; overflow: hidden;">
                        <div style="width: ${pct}%; height: 100%; background: ${barBg}; border-radius: 3px; transition: width 0.3s ease;"></div>
                    </div>
                    <span style="width: 60px; text-align: right; font-size: 12px; font-family: var(--font-mono); font-weight: 600; color: var(--text-primary);">${n.activation > 0 ? '+' : ''}${n.activation.toFixed(3)}</span>
                </div>
            `;
        });

        barHtml += `</div>`;
        barContainer.innerHTML = barHtml;
    }

    // 2. Render Single Neuron Prompt Text Lighting Strip
    const stripContainer = document.getElementById('neuron-lighting-container');
    if (stripContainer) {
        const strip = data.lighting_strip || [];
        const maxStripVal = strip.reduce((max, t) => Math.max(max, Math.abs(t.activation)), 0.001);

        let stripHtml = `
            <div style="margin-bottom: 10px; font-size: 13px; color: var(--text-muted);">
                Activation Strip for <strong>Neuron N${data.selected_neuron_idx}</strong> (Layer L${data.layer}) across prompt text:
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; padding: 12px; background: var(--bg-darkest); border-radius: 6px; border: 1px solid var(--border-color);">
        `;

        strip.forEach(t => {
            const opacity = Math.min(1.0, (Math.abs(t.activation) / maxStripVal)).toFixed(2);
            const chipBg = `rgba(52, 211, 153, ${opacity})`; // Green highlight
            const textColor = opacity > 0.4 ? '#FFFFFF' : 'var(--text-primary)';

            stripHtml += `
                <div class="lighting-chip" style="padding: 6px 10px; border-radius: 4px; background: ${chipBg}; color: ${textColor}; border: 1px solid rgba(255,255,255,0.1); font-family: var(--font-mono); font-size: 12px; font-weight: 600;" title="Token '${t.token}' (#${t.position}): Activation = ${t.activation}">
                    <span>${escapeHtml(t.token)}</span>
                    <span style="font-size: 9px; opacity: 0.75; margin-left: 4px;">(${t.activation > 0 ? '+' : ''}${t.activation})</span>
                </div>
            `;
        });

        stripHtml += `</div>`;
        stripContainer.innerHTML = stripHtml;
    }
}

function renderTokenAttribution(data) {
    const container = document.getElementById('token-attribution-container');
    if (!container || !data) return;

    const attributions = data.attributions || [];
    let html = `
        <div style="margin-bottom: 10px; font-size: 13px; color: var(--text-muted);">
            Input Token Attribution Influence (<strong>Attention Rollout</strong>) for target prediction <strong>'${escapeHtml(data.target_token)}'</strong> (#${data.target_position})
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; padding: 12px; background: var(--bg-darkest); border-radius: 6px; border: 1px solid var(--border-color);">
    `;

    attributions.forEach(item => {
        const opacity = item.score.toFixed(2);
        const chipBg = `rgba(56, 189, 248, ${opacity})`; // Accent cyan attribution highlight
        const textColor = opacity > 0.4 ? '#FFFFFF' : 'var(--text-primary)';

        html += `
            <div class="attribution-chip" style="padding: 6px 10px; border-radius: 4px; background: ${chipBg}; color: ${textColor}; border: 1px solid rgba(255,255,255,0.1); font-family: var(--font-mono); font-size: 12px; font-weight: 600;" title="Input Token '${item.token}' (#${item.position}): Attribution Score = ${(item.score * 100).toFixed(1)}%">
                <span>${escapeHtml(item.token)}</span>
                <span style="font-size: 9px; opacity: 0.85; margin-left: 4px;">${(item.score * 100).toFixed(0)}%</span>
            </div>
        `;
    });

    html += `</div>`;
    container.innerHTML = html;
}

function selectSingleNeuron(neuronIdx) {
    const layer = parseInt(document.getElementById('neuron-layer-select').value) || 0;
    const pos = parseInt(document.getElementById('neuron-pos-select').value) || 0;
    const topK = parseInt(document.getElementById('neuron-topk-select').value) || 10;
    fetchNeuronData(layer, pos, topK, neuronIdx);
}

function triggerNeuronFetch() {
    const layer = parseInt(document.getElementById('neuron-layer-select').value) || 0;
    const pos = parseInt(document.getElementById('neuron-pos-select').value) || 0;
    const topK = parseInt(document.getElementById('neuron-topk-select').value) || 10;
    fetchNeuronData(layer, pos, topK, null);
}

window.fetchNeuronData = fetchNeuronData;
window.selectSingleNeuron = selectSingleNeuron;
window.triggerNeuronFetch = triggerNeuronFetch;
