// InterpLens Studio Core Application Entry Point

document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    fetchSystemHealth();
    registerEventListeners();
});

window.currentSession = null;
window.currentMatrix = null;

// --- System Health Polling ---
async function fetchSystemHealth() {
    try {
        const data = await window.API.getSystemHealth();

        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');

        if (data.status === 'online') {
            if (dot) dot.className = 'status-dot status-online';
            if (text) text.textContent = 'Backend Online';
        } else if (data.status === 'loading') {
            if (dot) dot.className = 'status-dot status-busy';
            if (text) text.textContent = `Loading Model (${data.active_model})...`;
        } else if (data.status === 'error') {
            if (dot) dot.className = 'status-dot status-offline';
            if (text) text.textContent = `Error: ${data.error || 'Model load failed'}`;
        } else {
            if (dot) dot.className = 'status-dot status-busy';
            if (text) text.textContent = 'Initializing...';
        }

        if (document.getElementById('nav-model-name')) document.getElementById('nav-model-name').textContent = data.active_model || 'None';
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

            const viewLogit = document.getElementById('view-logit-lens');
            const viewGpu = document.getElementById('view-gpu-profiler');

            if (targetEngine === 'gpu-profiler') {
                if (viewLogit) {
                    viewLogit.classList.remove('active');
                    viewLogit.style.display = 'none';
                }
                if (viewGpu) {
                    viewGpu.style.display = 'block';
                    requestAnimationFrame(() => viewGpu.classList.add('active'));
                }
                if (window.fetchGpuProfilerData) window.fetchGpuProfilerData();
            } else {
                if (viewGpu) {
                    viewGpu.classList.remove('active');
                    viewGpu.style.display = 'none';
                }
                if (viewLogit) {
                    viewLogit.style.display = 'block';
                    requestAnimationFrame(() => viewLogit.classList.add('active'));
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

    } catch (err) {
        alert(`Analysis Error: ${err.message}`);
    } finally {
        runBtn.disabled = false;
        spinner.classList.add('hidden');
        runText.textContent = 'Run Model Analysis';
    }
}

window.fetchSystemHealth = fetchSystemHealth;
