// GPU & Hardware Profiler Module

let profilerTelemetry = {
    vramHistory: [],
    latencyHistory: [],
    vramChart: null,
    latencyChart: null,
    cacheChart: null,
    kvChart: null,
};

async function fetchGpuProfilerData() {
    try {
        const sessId = (window.currentMatrix && window.currentMatrix.session_id) || (window.currentSession && window.currentSession.session_id) || '';
        const prof = await window.API.getGpuProfiler(sessId);

        // 1. Device Profile Specs
        const badge = document.getElementById('prof-device-name');
        if (badge) badge.textContent = `${prof.has_gpu ? 'CUDA' : 'CPU'}: ${prof.device_name}`;

        if (document.getElementById('prof-cuda-ver')) document.getElementById('prof-cuda-ver').textContent = prof.cuda_version || 'N/A';
        if (document.getElementById('prof-torch-ver')) document.getElementById('prof-torch-ver').textContent = prof.torch_version || 'N/A';
        if (document.getElementById('prof-precision')) document.getElementById('prof-precision').textContent = prof.precision_dtype || 'fp16';
        if (document.getElementById('prof-compute-cap')) document.getElementById('prof-compute-cap').textContent = prof.compute_capability || 'N/A';
        if (document.getElementById('prof-sm-count')) document.getElementById('prof-sm-count').textContent = `${prof.multi_processor_count} SMs`;

        // 2. Live VRAM Gauge & Peak Marker
        const totalMb = prof.total_memory_mb || 1;
        const allocMb = prof.allocated_mb || 0;
        const resMb = prof.reserved_mb || 0;
        const peakMb = prof.max_allocated_mb || allocMb;

        if (document.getElementById('prof-alloc-mb')) document.getElementById('prof-alloc-mb').textContent = `${allocMb} MB`;
        if (document.getElementById('prof-res-mb')) document.getElementById('prof-res-mb').textContent = `${resMb} MB`;
        if (document.getElementById('prof-peak-mb')) document.getElementById('prof-peak-mb').textContent = `${peakMb} MB`;
        if (document.getElementById('prof-total-vram')) document.getElementById('prof-total-vram').textContent = `${totalMb} MB`;

        const allocPct = Math.min(100, Math.round((allocMb / totalMb) * 100));
        const resPct = Math.min(100 - allocPct, Math.round(((resMb - allocMb) / totalMb) * 100));
        const peakPct = Math.min(100, Math.round((peakMb / totalMb) * 100));

        const allocFill = document.getElementById('vram-gauge-alloc-fill');
        if (allocFill) allocFill.style.width = `${allocPct}%`;

        const resFill = document.getElementById('vram-gauge-res-fill');
        if (resFill) resFill.style.width = `${resPct}%`;

        const peakMarker = document.getElementById('vram-peak-marker');
        if (peakMarker) peakMarker.style.left = `${peakPct}%`;

        // 3. LRU Session Cache Table with Evict Button
        const sessCountEl = document.getElementById('prof-session-count');
        if (sessCountEl) sessCountEl.textContent = `${prof.sessions ? prof.sessions.length : 0} / ${prof.max_sessions || 3} Sessions`;

        const sessTbody = document.getElementById('session-cache-tbody');
        if (sessTbody) {
            sessTbody.innerHTML = '';
            if (prof.sessions && prof.sessions.length > 0) {
                prof.sessions.forEach(s => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="font-family:var(--font-mono); font-weight:700;">${s.session_id}</td>
                        <td title="${s.prompt}">${s.prompt}</td>
                        <td><span class="badge-token">${s.model_name}</span></td>
                        <td>${s.tokens_count}</td>
                        <td style="font-family:var(--font-mono); color: #06b6d4; font-weight:600;">${s.cache_size_mb} MB</td>
                        <td style="font-size:11px; opacity:0.8;">${s.created_at}</td>
                        <td>
                            <button class="session-evict-btn" onclick="evictSessionById('${s.session_id}')">Evict</button>
                        </td>
                    `;
                    sessTbody.appendChild(tr);
                });
            } else {
                sessTbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No active cached sessions in LRU store.</td></tr>';
            }
        }

        // 4. Update Telemetry History & Render Charts
        const timeTag = new Date().toLocaleTimeString();
        if (profilerTelemetry.vramHistory.length === 0 || profilerTelemetry.vramHistory[profilerTelemetry.vramHistory.length - 1].time !== timeTag) {
            profilerTelemetry.vramHistory.push({ time: timeTag, alloc: allocMb, res: resMb, peak: peakMb });
            if (profilerTelemetry.vramHistory.length > 15) profilerTelemetry.vramHistory.shift();
        }

        renderVramTimelineChart();
        renderLatencyChart();
        renderCacheBreakdownChart(prof.cache_breakdown);
        renderKvGrowthChart(prof.request_history, prof.kv_growth);

        // 5. Render 64-Block VRAM Memory Topology Map
        const container = document.getElementById('prof-64-blocks-container');
        if (container && prof.blocks) {
            container.innerHTML = '';
            prof.blocks.forEach((blk, i) => {
                const tile = document.createElement('div');
                tile.className = `vram-tile type-${blk.type}`;
                tile.title = `VRAM Topology Block #${i + 1}: ${blk.label} (${blk.mb} MB)`;
                container.appendChild(tile);
            });
        }

        // 6. Render Layer Memory Footprint Table
        const tbody = document.getElementById('prof-layer-tbody');
        if (tbody) {
            tbody.innerHTML = '';
            if (prof.layer_memory && prof.layer_memory.length > 0) {
                const maxSz = Math.max(...prof.layer_memory.map(l => l.size_mb), 1);
                prof.layer_memory.forEach(l => {
                    const tr = document.createElement('tr');
                    const pct = Math.min(100, Math.round((l.size_mb / maxSz) * 100));
                    tr.innerHTML = `
                        <td style="font-weight:600;">${l.layer}</td>
                        <td style="font-family:var(--font-mono);">${l.size_mb} MB</td>
                        <td>
                            <div class="mem-bar-bg">
                                <div class="mem-bar-fill" style="width: ${pct}%;"></div>
                            </div>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">Run prompt analysis to observe per-layer memory footprint.</td></tr>';
            }
        }
    } catch (err) {
        console.warn('GPU Profiler data fetch error:', err);
    }
}

async function evictSessionById(sessionId) {
    try {
        await window.API.deleteSession(sessionId);
        fetchGpuProfilerData();
        if (window.fetchSystemHealth) window.fetchSystemHealth();
    } catch (err) {
        console.error('Session eviction failed:', err);
    }
}

function renderVramTimelineChart() {
    const canvas = document.getElementById('vram-timeline-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const labels = profilerTelemetry.vramHistory.map(h => h.time);
    const allocData = profilerTelemetry.vramHistory.map(h => h.alloc);
    const resData = profilerTelemetry.vramHistory.map(h => h.res);

    if (profilerTelemetry.vramChart) {
        profilerTelemetry.vramChart.data.labels = labels;
        profilerTelemetry.vramChart.data.datasets[0].data = allocData;
        profilerTelemetry.vramChart.data.datasets[1].data = resData;
        profilerTelemetry.vramChart.update();
    } else {
        profilerTelemetry.vramChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Allocated VRAM (MB)', data: allocData, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, tension: 0.3 },
                    { label: 'Reserved VRAM (MB)', data: resData, borderColor: '#06b6d4', backgroundColor: 'transparent', borderDash: [4, 4], tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'VRAM (MB)', color: '#94a3b8', font: { size: 10 } } }
                },
                plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }
            }
        });
    }
}

function renderLatencyChart() {
    const canvas = document.getElementById('latency-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const labels = profilerTelemetry.latencyHistory.map(h => h.time);
    const data = profilerTelemetry.latencyHistory.map(h => h.ms);

    if (profilerTelemetry.latencyChart) {
        profilerTelemetry.latencyChart.data.labels = labels;
        profilerTelemetry.latencyChart.data.datasets[0].data = data;
        profilerTelemetry.latencyChart.update();
    } else {
        profilerTelemetry.latencyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{ label: 'Forward Pass Latency (ms)', data, backgroundColor: 'rgba(16, 185, 129, 0.7)', borderColor: '#10b981', borderWidth: 1 }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Latency (ms)', color: '#94a3b8', font: { size: 10 } } }
                },
                plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }
            }
        });
    }
}

function renderCacheBreakdownChart(breakdown) {
    const canvas = document.getElementById('cache-breakdown-chart');
    if (!canvas || !breakdown) return;
    const ctx = canvas.getContext('2d');

    const labels = ['Residual Stream', 'Attention Heads', 'MLP Layers', 'KV Cache'];
    const data = [breakdown.residual_stream_mb || 0, breakdown.attention_mb || 0, breakdown.mlp_mb || 0, breakdown.kv_cache_mb || 0];

    if (profilerTelemetry.cacheChart) {
        profilerTelemetry.cacheChart.data.datasets[0].data = data;
        profilerTelemetry.cacheChart.update();
    } else {
        profilerTelemetry.cacheChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: ['#3b82f6', '#06b6d4', '#8b5cf6', '#f59e0b'],
                    borderWidth: 1,
                    borderColor: 'var(--bg-panel)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } } }
            }
        });
    }
}

function renderKvGrowthChart(requestHistory, kvGrowth) {
    const canvas = document.getElementById('kv-growth-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let labels = [];
    let singleKvData = [];
    let cumulativeKvData = [];

    if (requestHistory && requestHistory.length > 0) {
        labels = requestHistory.map(r => r.label);
        singleKvData = requestHistory.map(r => r.kv_mb);
        cumulativeKvData = requestHistory.map(r => r.total_store_kv_mb);
    } else if (kvGrowth && kvGrowth.length > 0) {
        labels = kvGrowth.map(g => `Pos ${g.pos}`);
        singleKvData = kvGrowth.map(g => g.kv_mb);
        cumulativeKvData = kvGrowth.map(g => g.kv_mb);
    }

    if (profilerTelemetry.kvChart) {
        profilerTelemetry.kvChart.data.labels = labels;
        profilerTelemetry.kvChart.data.datasets[0].data = singleKvData;
        profilerTelemetry.kvChart.data.datasets[1].data = cumulativeKvData;
        profilerTelemetry.kvChart.update();
    } else {
        profilerTelemetry.kvChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Question KV Cache (MB)', data: singleKvData, borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.15)', fill: true, tension: 0.3 },
                    { label: 'Cumulative LRU Store (MB)', data: cumulativeKvData, borderColor: '#8b5cf6', backgroundColor: 'transparent', borderDash: [4, 4], tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'KV Storage (MB)', color: '#94a3b8', font: { size: 10 } } }
                },
                plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }
            }
        });
    }
}

window.fetchGpuProfilerData = fetchGpuProfilerData;
window.evictSessionById = evictSessionById;
window.profilerTelemetry = profilerTelemetry;
