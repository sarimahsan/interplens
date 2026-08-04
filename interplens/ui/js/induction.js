// --- InterpLens Phase 6: Induction Head Auto-Detector Engine ---

(function() {
    let currentInductionData = null;

    function initInductionEngine() {
        const runBtn = document.getElementById('run-induction-btn');
        if (runBtn) {
            runBtn.addEventListener('click', () => {
                fetchAndRenderInductionHeads();
            });
        }

        const seqLenSelect = document.getElementById('induction-seq-len');
        if (seqLenSelect) {
            seqLenSelect.addEventListener('change', () => {
                fetchAndRenderInductionHeads();
            });
        }

        const thresholdSlider = document.getElementById('induction-threshold');
        const thresholdVal = document.getElementById('induction-threshold-val');
        if (thresholdSlider) {
            thresholdSlider.addEventListener('input', (e) => {
                if (thresholdVal) thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
            });
            thresholdSlider.addEventListener('change', () => {
                fetchAndRenderInductionHeads();
            });
        }
    }

    async function fetchAndRenderInductionHeads() {
        const gridContainer = document.getElementById('induction-heatmap-grid');
        const loader = document.getElementById('induction-loading-spinner');

        const seqLen = parseInt(document.getElementById('induction-seq-len')?.value || '20', 10);
        const threshold = parseFloat(document.getElementById('induction-threshold')?.value || '0.15');

        if (loader) loader.classList.remove('hidden');
        if (gridContainer) gridContainer.style.opacity = '0.5';

        try {
            const data = await window.API.getInductionHeads(seqLen, threshold, 10);
            currentInductionData = data;

            renderSummaryMetrics(data);
            renderHeatmapGrid(data, threshold);
            renderTopHeadsTable(data, threshold);
        } catch (err) {
            console.error("Induction Detector Error:", err);
            if (gridContainer) {
                gridContainer.innerHTML = `<div class="error-banner" style="padding: 20px; color: #ef4444; text-align: center;">❌ Failed to run induction head auto-detection: ${err.message}</div>`;
            }
        } finally {
            if (loader) loader.classList.add('hidden');
            if (gridContainer) gridContainer.style.opacity = '1';
        }
    }

    function renderSummaryMetrics(data) {
        const scannedEl = document.getElementById('ind-kpi-scanned');
        const flaggedEl = document.getElementById('ind-kpi-flagged');
        const topHeadEl = document.getElementById('ind-kpi-top');

        if (scannedEl) scannedEl.textContent = `${data.total_heads_scanned} Heads (${data.num_layers}L × ${data.num_heads}H)`;
        if (flaggedEl) flaggedEl.textContent = `${data.flagged_count} Active`;

        if (topHeadEl && data.top_induction_heads && data.top_induction_heads.length > 0) {
            const top = data.top_induction_heads[0];
            topHeadEl.textContent = `L${top.layer} H${top.head} (${(top.score * 100).toFixed(1)}%)`;
        } else if (topHeadEl) {
            topHeadEl.textContent = 'None';
        }
    }

    function renderHeatmapGrid(data, threshold) {
        const gridContainer = document.getElementById('induction-heatmap-grid');
        if (!gridContainer) return;

        const layers = data.num_layers;
        const heads = data.num_heads;
        const matrix = data.matrix_scores;

        let html = `<div class="ind-grid-wrapper" style="grid-template-columns: 60px repeat(${heads}, 1fr);">`;

        // Column Headers (Heads 0..H-1)
        html += `<div class="ind-grid-corner">Layer \\ Head</div>`;
        for (let h = 0; h < heads; h++) {
            html += `<div class="ind-grid-col-head">H${h}</div>`;
        }

        // Grid Rows
        for (let l = 0; l < layers; l++) {
            html += `<div class="ind-grid-row-head">L${l}</div>`;
            for (let h = 0; h < heads; h++) {
                const score = matrix[l] ? (matrix[l][h] || 0.0) : 0.0;
                const isFlagged = score >= threshold;
                const opacity = Math.min(1.0, Math.max(0.08, score * 3.5));
                const bgColor = isFlagged 
                    ? `rgba(59, 130, 246, ${opacity})` 
                    : `rgba(148, 163, 184, ${Math.max(0.05, opacity * 0.4)})`;
                const border = isFlagged ? `1px solid var(--primary)` : `1px solid rgba(255, 255, 255, 0.05)`;

                html += `
                    <div class="ind-cell ${isFlagged ? 'flagged' : ''}" 
                         style="background-color: ${bgColor}; border: ${border};" 
                         title="Layer ${l}, Head ${h} | Induction Score: ${score.toFixed(4)} ${isFlagged ? '(Active Induction Head)' : ''}">
                        <span class="ind-cell-score">${score.toFixed(2)}</span>
                    </div>
                `;
            }
        }
        html += `</div>`;

        gridContainer.innerHTML = html;
    }

    function renderTopHeadsTable(data, threshold) {
        const tableBody = document.getElementById('induction-top-table-body');
        if (!tableBody) return;

        if (!data.top_induction_heads || data.top_induction_heads.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; opacity:0.6;">No high-scoring induction heads detected in this model.</td></tr>`;
            return;
        }

        let html = '';
        data.top_induction_heads.forEach((item, idx) => {
            const pct = (item.score * 100).toFixed(1);
            const isFlagged = item.score >= threshold;

            html += `
                <tr>
                    <td><strong>#${idx + 1}</strong></td>
                    <td><span class="head-badge">Layer ${item.layer}, Head ${item.head}</span></td>
                    <td><strong>${item.score.toFixed(4)}</strong> (${pct}%)</td>
                    <td>
                        <div class="meter-bar-container">
                            <div class="meter-bar-fill ${isFlagged ? 'active-fill' : ''}" style="width: ${Math.min(100, item.score * 200)}%;"></div>
                        </div>
                    </td>
                    <td>
                        ${isFlagged 
                            ? `<span class="tag-badge tag-active">⚡ Active Induction Head</span>` 
                            : `<span class="tag-badge tag-neutral">Standard Head</span>`}
                    </td>
                </tr>
            `;
        });

        tableBody.innerHTML = html;
    }

    // Expose engine to global window scope
    window.InductionEngine = {
        init: initInductionEngine,
        fetch: fetchAndRenderInductionHeads
    };

    document.addEventListener('DOMContentLoaded', () => {
        initInductionEngine();
    });
})();
