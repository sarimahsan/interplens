// InterpLens Studio Core Application Entry Point

var healthPollTimer = window.healthPollTimer || null;

document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    fetchSystemHealth();
    registerEventListeners();
});

window.currentSession = null;
window.currentMatrix = null;

// --- System Health Polling & Model Lock Engine ---
async function fetchSystemHealth() {
    try {
        const data = await window.API.getSystemHealth();

        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        const runBtn = document.getElementById('run-btn');
        const runText = document.getElementById('run-btn-text');
        const promptInput = document.getElementById('prompt-input');
        const loadingOverlay = document.getElementById('model-loading-overlay');
        const overlayTitle = document.getElementById('overlay-model-title');

        const activeModel = data.active_model || 'Model';

        if (data.status === 'online') {
            if (dot) dot.className = 'status-dot status-online';
            if (text) text.textContent = 'Backend Online';

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

            // Block UI while model is loading
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
            if (overlayTitle) overlayTitle.textContent = `Loading Model Weights (${activeModel}) into VRAM...`;

            // Fast poll every 2s while loading
            clearTimeout(healthPollTimer);
            healthPollTimer = setTimeout(fetchSystemHealth, 2000);

        } else if (data.status === 'error') {
            if (dot) dot.className = 'status-dot status-offline';
            if (text) text.textContent = `Error: ${data.error || 'Model load failed'}`;

            if (runBtn) { runBtn.disabled = true; }
            if (runText) { runText.textContent = 'Model Load Error'; }
            if (promptInput) { promptInput.disabled = true; }
            if (loadingOverlay) loadingOverlay.classList.remove('hidden');
            if (overlayTitle) overlayTitle.textContent = `Model Load Error: ${data.error || 'Check server logs'}`;
        } else {
            if (dot) dot.className = 'status-dot status-busy';
            if (text) text.textContent = 'Initializing...';
        }

        if (document.getElementById('nav-model-name')) document.getElementById('nav-model-name').textContent = activeModel;
        if (document.getElementById('nav-device-tag')) document.getElementById('nav-device-tag').textContent = data.device ? data.device.toUpperCase() : 'CPU';

        if (data.vram_usage && data.vram_usage.total_mb > 0) {
            if (document.getElementById('nav-vram-usage')) document.getElementById('nav-vram-usage').textContent = `VRAM: ${data.vram_usage.allocated_mb}MB / ${data.vram_usage.total_mb}MB`;
        } else {
            if (document.getElementById('nav-vram-usage')) document.getElementById('nav-vram-usage').textContent = 'CPU RAM Active';
        }
    } catch (err) {
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        if (dot) dot.className = 'status-dot status-offline';
        if (text) text.textContent = 'Backend Offline';
    }
}

// --- Theme Management ---
function initThemeManager() {
    const themeBtn = document.getElementById('theme-switch-btn');
    const storedTheme = localStorage.getItem('interplens_studio_theme');
    
    let activeTheme = storedTheme;
    if (!activeTheme) {
        activeTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    applyTheme(activeTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
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

    // Sidebar Tab Router
    document.querySelectorAll('.engine-menu .menu-btn:not(.disabled)').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetEngine = btn.getAttribute('data-engine');
            if (!targetEngine) return;

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
            document.getElementById('prompt-input').value = promptText;
            executePromptAnalysis();
        });
    });

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

// --- Execution Handler ---
async function executePromptAnalysis() {
    const promptInput = document.getElementById('prompt-input').value.trim();
    if (!promptInput) {
        alert('Please enter a prompt string.');
        return;
    }

    const runBtn = document.getElementById('run-btn');
    const runText = document.getElementById('run-btn-text');
    const spinner = document.getElementById('run-spinner');

    runBtn.disabled = true;
    spinner.classList.remove('hidden');
    runText.textContent = 'Running Model...';

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

    } catch (err) {
        alert(`Analysis Error: ${err.message}`);
    } finally {
        runBtn.disabled = false;
        spinner.classList.add('hidden');
        runText.textContent = 'Run Model Analysis';
    }
}

window.fetchSystemHealth = fetchSystemHealth;
