// InterpLens Studio Core Application Entry Point & State Persistence Engine

let telemetryWs = null;
let telemetryPollInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    fetchSystemHealth();
    initTelemetryWebSocket();
    registerEventListeners();
    restoreStudioState();
});

window.currentSession = null;
window.currentMatrix = null;

// --- System Health & Live VRAM Telemetry Engine ---
function initTelemetryWebSocket() {
    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/ws/telemetry`;

        telemetryWs = new WebSocket(wsUrl);

        telemetryWs.onopen = () => {
            updateWsStatusTag(true);
            if (telemetryPollInterval) {
                clearInterval(telemetryPollInterval);
                telemetryPollInterval = null;
            }
        };

        telemetryWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                processTelemetryData(data);
            } catch (e) {
                console.error('Telemetry WS parse error:', e);
            }
        };

        telemetryWs.onerror = () => {
            updateWsStatusTag(false);
        };

        telemetryWs.onclose = () => {
            updateWsStatusTag(false);
            if (!telemetryPollInterval) {
                telemetryPollInterval = setInterval(fetchSystemHealth, 2000);
            }
            setTimeout(initTelemetryWebSocket, 4000);
        };
    } catch (e) {
        updateWsStatusTag(false);
        if (!telemetryPollInterval) {
            telemetryPollInterval = setInterval(fetchSystemHealth, 2000);
        }
    }
}

function updateWsStatusTag(isConnected) {
    const tag = document.getElementById('nav-ws-status');
    if (!tag) return;
    if (isConnected) {
        tag.style.background = 'rgba(16, 185, 129, 0.15)';
        tag.style.color = '#10b981';
        tag.innerHTML = `<span style="width: 6px; height: 6px; border-radius: 50%; background-color: #10b981; display: inline-block;"></span> WS LIVE`;
    } else {
        tag.style.background = 'rgba(56, 189, 248, 0.15)';
        tag.style.color = '#38bdf8';
        tag.innerHTML = `<span style="width: 6px; height: 6px; border-radius: 50%; background-color: #38bdf8; display: inline-block;"></span> HTTP POLL`;
    }
}

async function fetchSystemHealth() {
    try {
        const data = await window.API.getSystemHealth();
        processTelemetryData(data);
    } catch (err) {
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        if (dot) dot.className = 'status-dot status-offline';
        if (text) text.textContent = 'Disconnected';
    }
}

function processTelemetryData(data) {
    if (!data) return;

    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    const runBtn = document.getElementById('run-btn');
    const runText = document.getElementById('run-btn-text');
    const promptInput = document.getElementById('prompt-input');
    const loadingOverlay = document.getElementById('model-loading-overlay');

    const activeModel = data.active_model || 'Model';

    if (data.status === 'online') {
        if (dot) dot.className = 'status-dot status-online';
        if (text) {
            text.textContent = data.warning ? `Online (${data.warning})` : 'Backend Online';
            if (data.warning) text.style.color = 'var(--accent-amber)';
        }

        // Unblock UI for interaction
        if (runBtn) { runBtn.disabled = false; }
        if (runText) { runText.textContent = 'Run Model Analysis'; }
        if (promptInput) {
            promptInput.disabled = false;
            promptInput.placeholder = 'Enter prompt to run model forward pass and observe intermediate residual stream projections...';
        }
        document.querySelectorAll('.btn-preset').forEach(btn => {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
        });
        if (loadingOverlay) loadingOverlay.classList.add('hidden');

    } else if (data.status === 'loading') {
        if (dot) dot.className = 'status-dot status-busy';
        if (text) text.textContent = `Loading Model (${activeModel})...`;

        if (runBtn) { runBtn.disabled = true; }
        if (runText) { runText.textContent = 'Loading Weights...'; }
        if (promptInput) {
            promptInput.disabled = true;
            promptInput.placeholder = `Loading ${activeModel} weights into VRAM... Please wait until initialization completes.`;
        }
        document.querySelectorAll('.btn-preset').forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
        });
        if (loadingOverlay) loadingOverlay.classList.remove('hidden');

    } else {
        if (dot) dot.className = 'status-dot status-offline';
        if (text) text.textContent = 'Backend Offline';
    }

    if (data.active_model && document.getElementById('nav-model-name')) {
        document.getElementById('nav-model-name').textContent = data.active_model;
    }
    if (data.device && document.getElementById('nav-device-tag')) {
        document.getElementById('nav-device-tag').textContent = data.device.toUpperCase();
    }

    // Live Header VRAM display update
    updateVramHeader(data.vram_usage, data.device);
}

function updateVramHeader(vramUsage, deviceStr) {
    const elem = document.getElementById('nav-vram-usage');
    if (!elem) return;

    if (!vramUsage) {
        elem.textContent = 'VRAM: 0 MB';
        return;
    }

    const isGpu = vramUsage.is_gpu || (deviceStr && deviceStr.toLowerCase().includes('cuda'));
    const prefix = isGpu ? 'VRAM' : 'RAM';

    const allocMb = vramUsage.allocated_mb !== undefined ? vramUsage.allocated_mb : 0;
    const totalMb = vramUsage.total_mb !== undefined ? vramUsage.total_mb : 0;

    if (totalMb > 0) {
        elem.textContent = `${prefix}: ${allocMb.toFixed(1)} MB / ${totalMb.toFixed(0)} MB`;
    } else if (allocMb > 0) {
        elem.textContent = `${prefix}: ${allocMb.toFixed(1)} MB`;
    } else {
        elem.textContent = `${prefix}: Active`;
    }
}

// --- Theme Management ---
function initThemeManager() {
    const themeBtn = document.getElementById('theme-switch-btn') || document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('interplens_studio_theme') || 'dark';

    applyTheme(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            const next = current === 'dark' ? 'light' : 'dark';
            applyTheme(next);
        });
    }
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('interplens_studio_theme', theme);
    const label = document.getElementById('theme-mode-text');
    if (label) label.textContent = theme === 'dark' ? 'Dark' : 'Light';
}

// --- Event Listeners & Router ---
function registerEventListeners() {
    const runBtn = document.getElementById('run-btn');
    if (runBtn) runBtn.addEventListener('click', executePromptAnalysis);

    // Model Discovery Report Modal Event Listeners
    const reportBtn = document.getElementById('btn-model-report');
    const reportModal = document.getElementById('model-report-modal');
    const closeReportBtn = document.getElementById('close-model-report-modal');

    if (reportBtn) {
        reportBtn.addEventListener('click', async () => {
            if (reportModal) reportModal.style.display = 'flex';
            const modalBody = document.getElementById('model-report-modal-body');
            if (modalBody) modalBody.innerHTML = '<div style="padding:20px; text-align:center; color:#94a3b8;">Loading automated model discovery report...</div>';
            try {
                const res = await fetch('/api/model/report');
                if (res.ok) {
                    const data = await res.json();
                    if (modalBody) modalBody.innerHTML = renderModelReportHtml(data);
                } else {
                    if (modalBody) modalBody.innerHTML = '<div style="padding:20px; color:#ef4444;">Failed to load model discovery report from server.</div>';
                }
            } catch (err) {
                if (modalBody) modalBody.innerHTML = '<div style="padding:20px; color:#ef4444;">Error connecting to model discovery report endpoint.</div>';
            }
        });
    }

    if (closeReportBtn) {
        closeReportBtn.addEventListener('click', () => {
            if (reportModal) reportModal.style.display = 'none';
        });
    }

    if (reportModal) {
        reportModal.addEventListener('click', (e) => {
            if (e.target === reportModal) reportModal.style.display = 'none';
        });
    }

    // Sidebar Tab Router
    document.querySelectorAll('.engine-menu .menu-btn:not(.disabled)').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetEngine = btn.getAttribute('data-engine');
            if (!targetEngine) return;

            // Save active tab in localStorage
            localStorage.setItem('interplens_active_tab', targetEngine);

            document.querySelectorAll('.engine-menu .menu-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.tab-view').forEach(v => {
                v.classList.remove('active');
                v.style.display = 'none';
            });

            if (targetEngine === 'gpu-profiler') {
                const view = document.getElementById('view-gpu-profiler');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                if (window.fetchGpuProfilerData) window.fetchGpuProfilerData();
            } else if (targetEngine === 'architecture') {
                const view = document.getElementById('view-architecture');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                if (window.fetchModelTopologyData) window.fetchModelTopologyData();
            } else if (targetEngine === 'residual-stream') {
                const view = document.getElementById('view-residual-stream');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                const sessId = (window.currentSession && window.currentSession.session_id) || '';
                if (window.fetchResidualStreamMetrics && sessId) window.fetchResidualStreamMetrics(sessId);
            } else if (targetEngine === 'attention') {
                const view = document.getElementById('view-attention');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                if (window.fetchAttentionHeadsData) window.fetchAttentionHeadsData(0, 0, 0.02);
            } else if (targetEngine === 'neurons') {
                const view = document.getElementById('view-neurons');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                if (window.fetchNeuronData) window.fetchNeuronData(0, null, 10, null);
            } else if (targetEngine === 'causal') {
                const view = document.getElementById('view-causal');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
            } else if (targetEngine === 'induction') {
                const view = document.getElementById('view-induction');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
                if (window.InductionEngine && window.InductionEngine.fetch) {
                    window.InductionEngine.fetch();
                }
            } else {
                const view = document.getElementById('view-logit-lens');
                if (view) {
                    view.style.display = 'block';
                    requestAnimationFrame(() => view.classList.add('active'));
                }
            }
        });
    });

    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const promptText = e.target.getAttribute('data-prompt');
            if (promptText) {
                document.getElementById('prompt-input').value = promptText;
                executePromptAnalysis();
            }
        });
    });

    // Save prompt inputs on change
    const promptInput = document.getElementById('prompt-input');
    if (promptInput) {
        promptInput.addEventListener('input', () => {
            localStorage.setItem('interplens_last_prompt', promptInput.value);
        });
    }

    const cleanInput = document.getElementById('causal-clean-input');
    const corruptInput = document.getElementById('causal-corrupt-input');
    const targetInput = document.getElementById('causal-target-input');

    if (cleanInput) cleanInput.addEventListener('input', () => localStorage.setItem('interplens_causal_clean', cleanInput.value));
    if (corruptInput) corruptInput.addEventListener('input', () => localStorage.setItem('interplens_causal_corrupt', corruptInput.value));
    if (targetInput) targetInput.addEventListener('input', () => localStorage.setItem('interplens_causal_target', targetInput.value));

    // Metric Toggles (Prob, KL, Entropy, Top-5 Trajectories)
    document.querySelectorAll('.btn-metric').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-metric').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const metric = btn.getAttribute('data-metric');
            if (window.setActiveMetric) window.setActiveMetric(metric);
            
            const activePill = document.querySelector('.token-pill.active');
            const posIdx = activePill ? parseInt(activePill.querySelector('.token-idx').textContent.replace('#', '')) : 0;
            if (window.renderInspectionDetail) window.renderInspectionDetail(isNaN(posIdx) ? 0 : posIdx);
        });
    });
}

// --- Execution Handler & State Saver ---
async function executePromptAnalysis() {
    const promptInput = document.getElementById('prompt-input').value.trim();
    if (!promptInput) {
        alert('Please enter a prompt string.');
        return;
    }

    // Save prompt text to localStorage
    localStorage.setItem('interplens_last_prompt', promptInput);

    const runBtn = document.getElementById('run-btn');
    const runText = document.getElementById('run-btn-text');
    const spinner = document.getElementById('run-spinner');

    if (runBtn) runBtn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');
    if (runText) runText.textContent = 'Running Model...';

    const t0 = performance.now();
    try {
        window.currentSession = await window.API.runModelForwardPass(promptInput);
        document.getElementById('session-tag-display').textContent = `Session: ${window.currentSession.session_id.substring(0, 8)}`;

        if (window.currentSession.model_info) {
            document.getElementById('spec-layers').textContent = `${window.currentSession.model_info.num_layers} Blocks`;
            document.getElementById('spec-dim').textContent = `${window.currentSession.model_info.hidden_dim}d`;
        }

        window.renderTokensFlow(window.currentSession.tokens);

        const topK = document.getElementById('topk-select').value;
        const applyLn = document.getElementById('ln-toggle').checked;
        
        window.currentMatrix = await window.API.getLogitLensMatrix(window.currentSession.session_id, topK, applyLn);
        const latencyMs = Math.round(performance.now() - t0);

        if (window.profilerTelemetry) {
            window.profilerTelemetry.latencyHistory.push({ time: new Date().toLocaleTimeString(), ms: latencyMs });
            if (window.profilerTelemetry.latencyHistory.length > 15) window.profilerTelemetry.latencyHistory.shift();
        }

        window.renderMatrixGrid(window.currentMatrix);
        window.renderModelResponseCard(window.currentMatrix);

        if (window.currentMatrix.positions && window.currentMatrix.positions.length > 0) {
            window.renderInspectionDetail(0);
        }

        fetchSystemHealth();
        if (window.fetchGpuProfilerData) window.fetchGpuProfilerData();
        if (window.fetchResidualStreamMetrics && window.currentSession.session_id) {
            window.fetchResidualStreamMetrics(window.currentSession.session_id);
        }
        if (window.fetchAttentionHeadsData && window.currentSession.session_id) {
            window.fetchAttentionHeadsData(0, 0, 0.02);
        }
        if (window.fetchNeuronData && window.currentSession.session_id) {
            window.fetchNeuronData(0, null, 10, null);
        }

    } catch (err) {
        console.warn(`Analysis Error: ${err.message}`);
    } finally {
        if (runBtn) runBtn.disabled = false;
        if (spinner) spinner.classList.add('hidden');
        if (runText) runText.textContent = 'Run Model Analysis';
    }
}

// --- Restore State & Active Tab on Page Reload ---
function restoreStudioState() {
    const savedPrompt = localStorage.getItem('interplens_last_prompt');
    const savedClean = localStorage.getItem('interplens_causal_clean');
    const savedCorrupt = localStorage.getItem('interplens_causal_corrupt');
    const savedTarget = localStorage.getItem('interplens_causal_target');
    const savedTab = localStorage.getItem('interplens_active_tab');

    // Restore causal inputs
    if (savedClean && document.getElementById('causal-clean-input')) {
        document.getElementById('causal-clean-input').value = savedClean;
    }
    if (savedCorrupt && document.getElementById('causal-corrupt-input')) {
        document.getElementById('causal-corrupt-input').value = savedCorrupt;
    }
    if (savedTarget && document.getElementById('causal-target-input')) {
        document.getElementById('causal-target-input').value = savedTarget;
    }

    // Restore main prompt and re-run analysis
    if (savedPrompt && document.getElementById('prompt-input')) {
        document.getElementById('prompt-input').value = savedPrompt;
        executePromptAnalysis();
    }

    // Restore active tab
    if (savedTab) {
        const targetBtn = document.querySelector(`.engine-menu .menu-btn[data-engine="${savedTab}"]`);
        if (targetBtn) {
            targetBtn.click();
        }
    }
}

window.fetchSystemHealth = fetchSystemHealth;

function renderModelReportHtml(data) {
    if (!data) return '<div style="padding:20px; color:#ef4444;">No report data available.</div>';

    const sf = data.static_fingerprint || {};
    const ec = (data.engine_capabilities && data.engine_capabilities.engines) ? data.engine_capabilities.engines : {};

    const confPct = ((data.discovery_confidence || 1.0) * 100).toFixed(1);
    const capLevel = data.capability_level !== undefined ? data.capability_level : 5;
    const capName = data.capability_level_name || 'Full Support';

    let html = `
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <!-- Key Metrics Bar -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.2);">
                <div>
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Confidence</div>
                    <div style="font-size: 18px; font-weight: 700; color: #38bdf8;">${confPct}%</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Capability Level</div>
                    <div style="font-size: 14px; font-weight: 700; color: #f8fafc;">Level ${capLevel} (${escapeHtml(capName)})</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Layers Stack</div>
                    <div style="font-size: 18px; font-weight: 700; color: #f8fafc;">${sf.num_layers || 0} Blocks</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Hidden Dim</div>
                    <div style="font-size: 18px; font-weight: 700; color: #f8fafc;">${sf.hidden_size || 0}d</div>
                </div>
            </div>

            <!-- Static Architecture Geometry -->
            <div>
                <h4 style="margin: 0 0 10px 0; font-size: 14px; font-weight: 600; color: #38bdf8;">📐 Model Architecture & Static Fingerprint</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 12px; border: 1px solid #334155; border-radius: 6px; overflow: hidden;">
                    <thead>
                        <tr style="background: rgba(30, 41, 59, 0.9); color: #94a3b8; text-align: left;">
                            <th style="padding: 8px 12px; border-bottom: 1px solid #334155;">Parameter</th>
                            <th style="padding: 8px 12px; border-bottom: 1px solid #334155;">Metric</th>
                            <th style="padding: 8px 12px; border-bottom: 1px solid #334155;">Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 8px 12px; font-weight: 600; color: #f8fafc;">Model Identifier</td>
                            <td style="padding: 8px 12px; color: #38bdf8; font-family: monospace;">${escapeHtml(data.model_name || 'N/A')}</td>
                            <td style="padding: 8px 12px; color: #94a3b8;">Active registered module name</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 8px 12px; font-weight: 600; color: #f8fafc;">Architecture Strategy</td>
                            <td style="padding: 8px 12px; color: #f8fafc; font-family: monospace;">${escapeHtml((data.architecture_id || '').toUpperCase())}</td>
                            <td style="padding: 8px 12px; color: #94a3b8;">Hook discovery adapter strategy</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 8px 12px; font-weight: 600; color: #f8fafc;">Attention Heads (Q / KV)</td>
                            <td style="padding: 8px 12px; color: #f8fafc;">${sf.num_heads || 0} Q / ${sf.num_kv_heads || sf.num_heads || 0} KV</td>
                            <td style="padding: 8px 12px; color: #94a3b8;">MHSA Query & Key-Value heads</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #1e293b;">
                            <td style="padding: 8px 12px; font-weight: 600; color: #f8fafc;">Vocabulary Size</td>
                            <td style="padding: 8px 12px; color: #f8fafc;">${(sf.vocab_size || 0).toLocaleString()} Tokens</td>
                            <td style="padding: 8px 12px; color: #94a3b8;">Embedding dictionary dimension</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 12px; font-weight: 600; color: #f8fafc;">Embeddings & MoE</td>
                            <td style="padding: 8px 12px; color: #f8fafc;">RoPE: ${sf.has_rope ? 'Yes' : 'No'} | MoE: ${sf.is_moe ? 'Yes' : 'No'}</td>
                            <td style="padding: 8px 12px; color: #94a3b8;">Rotary embeddings & MoE presence</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Engine Capabilities Matrix -->
            <div>
                <h4 style="margin: 0 0 10px 0; font-size: 14px; font-weight: 600; color: #38bdf8;">⚡ Interpretability Engine Audit Matrix</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 12px; border: 1px solid #334155; border-radius: 6px; overflow: hidden;">
                    <thead>
                        <tr style="background: rgba(30, 41, 59, 0.9); color: #94a3b8; text-align: left;">
                            <th style="padding: 8px 12px; border-bottom: 1px solid #334155;">Engine Name</th>
                            <th style="padding: 8px 12px; border-bottom: 1px solid #334155;">Audit Status</th>
                            <th style="padding: 8px 12px; border-bottom: 1px solid #334155;">Audit Notes & Capabilities</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    Object.keys(ec).forEach((engKey) => {
        const eng = ec[engKey];
        const status = (eng.status || 'unavailable').toUpperCase();
        let badge = '<span style="padding: 2px 8px; border-radius: 4px; background: rgba(239, 68, 68, 0.15); color: #ef4444; font-weight: 600; font-size: 11px;">UNAVAILABLE</span>';
        if (status === 'SUPPORTED') {
            badge = '<span style="padding: 2px 8px; border-radius: 4px; background: rgba(16, 185, 129, 0.15); color: #10b981; font-weight: 600; font-size: 11px;">✓ SUPPORTED</span>';
        } else if (status === 'PARTIAL') {
            badge = '<span style="padding: 2px 8px; border-radius: 4px; background: rgba(245, 158, 11, 0.15); color: #f59e0b; font-weight: 600; font-size: 11px;">⚠ PARTIAL</span>';
        }

        html += `
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 8px 12px; font-weight: 600; color: #f8fafc;">${escapeHtml(eng.engine_name || engKey)}</td>
                <td style="padding: 8px 12px;">${badge}</td>
                <td style="padding: 8px 12px; color: #94a3b8;">${escapeHtml(eng.reason || 'N/A')}</td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;
    return html;
}
