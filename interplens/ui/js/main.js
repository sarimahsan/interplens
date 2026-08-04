// InterpLens Studio Core Application Entry Point & State Persistence Engine

var healthPollTimer = window.healthPollTimer || null;

document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    fetchSystemHealth();
    registerEventListeners();
    restoreStudioState();
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

        } else {
            if (dot) dot.className = 'status-dot status-offline';
            if (text) text.textContent = 'Backend Offline';
        }

        if (data.active_model && document.getElementById('nav-model-name')) {
            document.getElementById('nav-model-name').textContent = data.active_model;
        }

    } catch (err) {
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        if (dot) dot.className = 'status-dot status-offline';
        if (text) text.textContent = 'Disconnected';
    } finally {
        if (!healthPollTimer) {
            healthPollTimer = setInterval(fetchSystemHealth, 5000);
        }
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
