// Model Architecture Topology Interactive Graph Renderer

async function fetchModelTopologyData() {
    try {
        const top = await window.API.getModelTopology();
        renderModelTopologyOverview(top);
        renderModelTopologyDiagram(top);
    } catch (err) {
        console.warn('Model topology fetch error:', err);
    }
}

function renderModelTopologyOverview(top) {
    if (document.getElementById('topo-model-name')) document.getElementById('topo-model-name').textContent = top.model_name || 'Transformer';
    if (document.getElementById('topo-param-count')) document.getElementById('topo-param-count').textContent = top.total_parameters_formatted || '0';
    if (document.getElementById('topo-layer-count')) document.getElementById('topo-layer-count').textContent = `${top.num_layers} Layers`;
    if (document.getElementById('topo-dim-count')) document.getElementById('topo-dim-count').textContent = `${top.hidden_dim}d`;
    if (document.getElementById('topo-head-count')) document.getElementById('topo-head-count').textContent = `${top.num_heads} Heads (${top.head_dim}d/head)`;
    if (document.getElementById('topo-vocab-size')) document.getElementById('topo-vocab-size').textContent = `${top.vocab_size ? top.vocab_size.toLocaleString() : '50,257'} Tokens`;
    if (document.getElementById('topo-dtype')) document.getElementById('topo-dtype').textContent = `${top.dtype || 'fp32'} on ${top.device || 'cpu'}`;
}

function renderModelTopologyDiagram(top) {
    const container = document.getElementById('topology-graph-container');
    if (!container || !top.nodes) return;
    container.innerHTML = '';

    const flowWrapper = document.createElement('div');
    flowWrapper.className = 'topo-flow-wrapper';
    flowWrapper.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 16px; padding: 24px 10px; width: 100%; max-width: 850px; margin: 0 auto;';

    top.nodes.forEach((node, idx) => {
        const card = document.createElement('div');
        card.className = 'topo-node-card';
        card.style.cssText = `
            width: 100%;
            background-color: var(--bg-panel);
            border: 1px solid var(--border-highlight);
            border-left: 5px solid ${node.color || '#3b82f6'};
            border-radius: 8px;
            padding: 14px 18px;
            box-shadow: var(--shadow-sm);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        `;

        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-weight: 700; font-size: 14px; color: var(--text-main); display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: ${node.color};"></span>
                    ${node.title}
                </div>
                <span class="status-tag" style="background-color: ${node.color}22; color: ${node.color}; border: 1px solid ${node.color}44;">${node.badge}</span>
            </div>
            <div style="font-family: var(--font-mono); font-size: 11px; color: var(--primary); margin-bottom: 6px;">${node.subtitle}</div>
            <div style="font-size: 12px; color: var(--text-sub); line-height: 1.4;">${node.description}</div>
        `;

        flowWrapper.appendChild(card);

        // Add downward connection arrow except for last node
        if (idx < top.nodes.length - 1) {
            const arrow = document.createElement('div');
            arrow.className = 'topo-arrow';
            arrow.style.cssText = 'display: flex; flex-direction: column; align-items: center; color: var(--border-highlight); font-size: 16px; font-weight: bold; margin: -6px 0;';
            arrow.innerHTML = `
                <div style="width: 2px; height: 16px; background-color: var(--border-highlight);"></div>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            `;
            flowWrapper.appendChild(arrow);
        }
    });

    container.appendChild(flowWrapper);
}

window.fetchModelTopologyData = fetchModelTopologyData;
