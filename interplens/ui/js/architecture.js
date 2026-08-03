// Model Architecture Topology Interactive Graph & Parameter Breakdown Renderer

async function fetchModelTopologyData() {
    try {
        const top = await window.API.getModelTopology();
        renderModelTopologyOverview(top);
        renderParameterBreakdown(top);
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

function renderParameterBreakdown(top) {
    const container = document.getElementById('parameter-breakdown-container');
    if (!container || !top.parameter_breakdown) return;

    const breakdown = top.parameter_breakdown;
    const totalFmt = top.total_parameters_formatted || `${(top.total_parameters / 1e6).toFixed(1)}M`;

    // 1. Render Multi-Color Stacked Horizontal Bar
    let barHtml = `
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 13px; color: var(--text-muted);">
                <span>Total Model Parameter Budget: <strong style="color: var(--text-primary); font-size: 14px;">${totalFmt} (${top.total_parameters.toLocaleString()} parameters)</strong></span>
            </div>
            <div style="display: flex; height: 18px; width: 100%; border-radius: 6px; overflow: hidden; background: rgba(0,0,0,0.3); border: 1px solid var(--border-color);">
    `;

    breakdown.forEach(item => {
        if (item.percentage > 0) {
            barHtml += `<div style="width: ${item.percentage}%; height: 100%; background-color: ${item.color};" title="${item.category}: ${item.formatted} (${item.percentage}%)"></div>`;
        }
    });

    barHtml += `</div></div>`;

    // 2. Render Parameter Breakdown Table
    let tableHtml = `
        <div class="table-scroll">
            <table class="studio-table" style="width: 100%; border-collapse: collapse; font-size: 12px;">
                <thead>
                    <tr>
                        <th style="text-align: left; padding: 8px 12px; background: var(--bg-darkest);">Component Sublayer</th>
                        <th style="text-align: right; padding: 8px 12px; background: var(--bg-darkest);">Parameter Count</th>
                        <th style="text-align: right; padding: 8px 12px; background: var(--bg-darkest);">Percentage Share</th>
                        <th style="text-align: center; padding: 8px 12px; background: var(--bg-darkest);">Proportion Bar</th>
                    </tr>
                </thead>
                <tbody>
    `;

    breakdown.forEach(item => {
        tableHtml += `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 10px 12px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 3px; background-color: ${item.color};"></span>
                    <span>${item.category}</span>
                    ${item.note ? `<span style="font-size: 10px; opacity: 0.7; font-weight: 400; font-style: italic;">(${item.note})</span>` : ''}
                </td>
                <td style="padding: 10px 12px; text-align: right; font-family: var(--font-mono); font-weight: 700; color: var(--text-primary);">
                    ${item.formatted} <span style="font-size: 10px; opacity: 0.6; font-weight: 400;">(${item.count.toLocaleString()})</span>
                </td>
                <td style="padding: 10px 12px; text-align: right; font-family: var(--font-mono); font-weight: 700; color: ${item.color};">
                    ${item.percentage.toFixed(2)}%
                </td>
                <td style="padding: 10px 12px; width: 140px;">
                    <div style="height: 8px; width: 100%; background: rgba(0,0,0,0.3); border-radius: 3px; overflow: hidden;">
                        <div style="width: ${Math.min(100, item.percentage)}%; height: 100%; background-color: ${item.color}; border-radius: 3px;"></div>
                    </div>
                </td>
            </tr>
        `;
    });

    tableHtml += `</tbody></table></div>`;

    container.innerHTML = barHtml + tableHtml;
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
