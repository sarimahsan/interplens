// Fast Circuits Engine: Direct Logit Attribution (DLA) & Gradient Attribution Patching (AtP)

var CircuitsEngine = window.CircuitsEngine || {
    dlaChart: null,

    async fetchDLA() {
        const sessId = (window.currentMatrix && window.currentMatrix.session_id) || (window.currentSession && window.currentSession.session_id);
        const statusEl = document.getElementById('dla-status');
        if (!sessId) {
            if (statusEl) statusEl.textContent = 'Run a prompt first (in Logit Lens) to compute Direct Logit Attribution.';
            return;
        }

        const targetTok = (document.getElementById('dla-target-input') || {}).value || null;
        const foilTok = (document.getElementById('dla-foil-input') || {}).value || null;
        if (statusEl) statusEl.textContent = 'Computing Direct Logit Attribution...';

        try {
            const data = await window.API.getDirectLogitAttribution(sessId, null, targetTok ? targetTok.trim() : null, foilTok ? foilTok.trim() : null);
            this.renderDLA(data);
            if (statusEl) statusEl.textContent = `Decomposed logit diff for "${data.target_token}" vs "${data.foil_token}" at pos ${data.position}.`;
        } catch (err) {
            if (statusEl) statusEl.textContent = `DLA error: ${err.message}`;
        }
    },

    renderDLA(data) {
        const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        setEl('dla-target-badge', `"${data.target_token}"`);
        setEl('dla-foil-badge', `"${data.foil_token}"`);
        setEl('dla-final-diff', `${data.final_logit_diff >= 0 ? '+' : ''}${data.final_logit_diff}`);
        setEl('dla-embed-val', `${data.embed_contribution >= 0 ? '+' : ''}${data.embed_contribution}`);

        const canvas = document.getElementById('dla-layer-chart');
        if (canvas && typeof Chart !== 'undefined') {
            const labels = data.layer_contributions.map(lc => `L${lc.layer}`);
            const attnData = data.layer_contributions.map(lc => lc.attn_contribution);
            const mlpData = data.layer_contributions.map(lc => lc.mlp_contribution);

            if (this.dlaChart) {
                this.dlaChart.data.labels = labels;
                this.dlaChart.data.datasets[0].data = attnData;
                this.dlaChart.data.datasets[1].data = mlpData;
                this.dlaChart.update();
            } else {
                this.dlaChart = new Chart(canvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels,
                        datasets: [
                            { label: 'Attention Sublayer', data: attnData, backgroundColor: 'rgba(139, 92, 246, 0.8)', borderColor: '#8b5cf6', borderWidth: 1 },
                            { label: 'MLP Sublayer', data: mlpData, backgroundColor: 'rgba(236, 72, 153, 0.8)', borderColor: '#ec4899', borderWidth: 1 },
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                            y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.08)' }, title: { display: true, text: 'Logit Diff Contribution', color: '#94a3b8', font: { size: 11 } } }
                        },
                        plugins: { legend: { labels: { color: '#cbd5e1', font: { size: 11 } } } }
                    }
                });
            }
        }

        const tbody = document.getElementById('dla-top-components-tbody');
        if (tbody && data.top_components) {
            tbody.innerHTML = '';
            const maxAbs = Math.max(...data.top_components.map(c => Math.abs(c.contribution)), 0.01);
            data.top_components.forEach(c => {
                const pos = c.contribution >= 0;
                const color = pos ? '#10b981' : '#ef4444';
                const pct = Math.min(100, Math.round((Math.abs(c.contribution) / maxAbs) * 100));
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight: 600;">${c.component}</td>
                    <td><span class="badge-token">${c.kind.toUpperCase()}</span></td>
                    <td style="font-family: var(--font-mono); font-weight: 700; color: ${color};">${pos ? '+' : ''}${c.contribution.toFixed(4)}</td>
                    <td style="width: 140px;">
                        <div style="height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden;">
                            <div style="width: ${pct}%; height: 100%; background: ${color};"></div>
                        </div>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
    },

    async runAtP() {
        const cleanPrompt = (document.getElementById('atp-clean-input') || {}).value || '';
        const corruptPrompt = (document.getElementById('atp-corrupt-input') || {}).value || '';
        const targetTok = (document.getElementById('atp-target-input') || {}).value || null;
        const foilTok = (document.getElementById('atp-foil-input') || {}).value || null;
        const statusEl = document.getElementById('atp-status');
        const btn = document.getElementById('btn-run-atp');

        if (!cleanPrompt.trim() || !corruptPrompt.trim()) {
            if (statusEl) statusEl.textContent = 'Enter both Clean and Corrupt prompts.';
            return;
        }

        if (btn) { btn.disabled = true; btn.textContent = 'Running AtP...'; }
        if (statusEl) statusEl.textContent = 'Running 2 forward passes + 1 backward pass...';

        try {
            const data = await window.API.runAttributionPatching(
                cleanPrompt.trim(),
                corruptPrompt.trim(),
                targetTok ? targetTok.trim() : null,
                foilTok ? foilTok.trim() : null
            );
            this.renderAtP(data);
            if (statusEl) statusEl.textContent = `Completed in ${data.runtime_ms} ms • Peak: Layer ${data.max_attribution_layer}, Pos ${data.max_attribution_position} (${data.max_attribution_pct}%).`;
        } catch (err) {
            if (statusEl) statusEl.textContent = `AtP error: ${err.message}`;
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = 'Run Fast AtP (2 Fwd + 1 Bwd)'; }
        }
    },

    renderAtP(data) {
        const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        setEl('atp-runtime-badge', `${data.runtime_ms} ms`);
        setEl('atp-target-pair', `"${data.target_token}" vs "${data.foil_token}"`);
        setEl('atp-peak-cell', `L${data.max_attribution_layer} • Pos ${data.max_attribution_position} (${data.max_attribution_pct}%)`);

        // Render Layer x Token Position heatmap
        const container = document.getElementById('atp-heatmap-container');
        if (container && data.heatmap_matrix) {
            const flat = data.heatmap_matrix.flat().map(v => Math.abs(v));
            const maxAbs = Math.max(...flat, 1.0);

            let html = '<div class="table-scroll"><table class="studio-table" style="text-align: center; font-size: 11px;"><thead><tr><th>Layer</th>';
            data.clean_tokens.forEach((tok, idx) => {
                const safeTok = String(tok).replace(/</g, '&lt;').replace(/>/g, '&gt;');
                html += `<th title="Pos ${idx}: ${safeTok}">${safeTok}</th>`;
            });
            html += '</tr></thead><tbody>';

            data.heatmap_matrix.forEach((row, lIdx) => {
                html += `<tr><td style="font-weight: 700; font-family: var(--font-mono);">L${lIdx}</td>`;
                row.forEach((val, pIdx) => {
                    const intensity = Math.min(1, Math.abs(val) / maxAbs);
                    const bg = val >= 0
                        ? `rgba(16, 185, 129, ${(0.08 + intensity * 0.72).toFixed(2)})`
                        : `rgba(239, 68, 68, ${(0.08 + intensity * 0.72).toFixed(2)})`;
                    html += `<td style="background: ${bg}; font-family: var(--font-mono);" title="L${lIdx} Pos ${pIdx} (${data.clean_tokens[pIdx]}): ${val}%">${val.toFixed(1)}%</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody></table></div>';
            container.innerHTML = html;
        }

        // Render top causal components (Attn/MLP)
        const tbody = document.getElementById('atp-components-tbody');
        if (tbody && data.top_components) {
            tbody.innerHTML = '';
            if (data.top_components.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No sublayer gradients captured for this architecture.</td></tr>';
                return;
            }
            data.top_components.forEach(c => {
                const pos = c.recovery_pct >= 0;
                const color = pos ? '#10b981' : '#ef4444';
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight: 600;">${c.component}</td>
                    <td><span class="badge-token">${c.kind.toUpperCase()}</span></td>
                    <td style="font-family: var(--font-mono);">${c.score.toFixed(4)}</td>
                    <td style="font-family: var(--font-mono); font-weight: 700; color: ${color};">${pos ? '+' : ''}${c.recovery_pct.toFixed(2)}%</td>
                `;
                tbody.appendChild(tr);
            });
        }
    },

    bindEvents() {
        const dlaBtn = document.getElementById('btn-run-dla');
        if (dlaBtn && !dlaBtn.dataset.bound) {
            dlaBtn.dataset.bound = '1';
            dlaBtn.addEventListener('click', () => this.fetchDLA());
        }
        const atpBtn = document.getElementById('btn-run-atp');
        if (atpBtn && !atpBtn.dataset.bound) {
            atpBtn.dataset.bound = '1';
            atpBtn.addEventListener('click', () => this.runAtP());
        }
    }
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => CircuitsEngine.bindEvents());
} else {
    CircuitsEngine.bindEvents();
}

window.CircuitsEngine = CircuitsEngine;
