// InterpLens Documentation Portal • Interactive Logic & Enhanced Command Palette

document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    initCopyButtons();
    initTabSwitchers();
    initScrollSpy();
    initCommandPalette();
    initMathRendering();
});

// --- Mathematical Typography Rendering (KaTeX) ---
function initMathRendering() {
    function runRender() {
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(document.body, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                throwOnError: false
            });
        }
    }

    if (typeof renderMathInElement === 'function') {
        runRender();
    } else {
        window.addEventListener('load', runRender);
    }
}

// --- Theme Management ---
function initThemeManager() {
    const themeBtn = document.getElementById('theme-toggle-btn');
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');
    const savedTheme = localStorage.getItem('interplens_docs_theme') || 'dark';

    applyTheme(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            const next = current === 'dark' ? 'light' : 'dark';
            applyTheme(next);
        });
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('interplens_docs_theme', theme);
        if (themeText) themeText.textContent = theme === 'dark' ? 'Dark' : 'Light';
        if (themeIcon) {
            themeIcon.innerHTML = theme === 'dark' 
                ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`
                : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
        }
    }
}

// --- Copy Code Blocks ---
function initCopyButtons() {
    document.querySelectorAll('.copy-pill-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const container = btn.closest('.spread-code-box');
            const codeBlock = container ? container.querySelector('code') : null;
            if (codeBlock) {
                const text = codeBlock.innerText.trim();
                navigator.clipboard.writeText(text).then(() => {
                    const originalHTML = btn.innerHTML;
                    btn.innerHTML = `<span style="color: var(--brand-primary); font-weight: 700;">✓ Copied!</span>`;
                    setTimeout(() => {
                        btn.innerHTML = originalHTML;
                    }, 2000);
                }).catch(() => {
                    const textArea = document.createElement('textarea');
                    textArea.value = text;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    btn.innerText = '✓ Copied!';
                    setTimeout(() => { btn.innerText = 'Copy'; }, 2000);
                });
            }
        });
    });
}

// --- Interactive Code Tabs ---
function initTabSwitchers() {
    document.querySelectorAll('.tab-container').forEach(container => {
        const btns = container.querySelectorAll('.tab-btn');
        const contents = container.querySelectorAll('.tab-content');

        btns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');

                btns.forEach(b => b.classList.remove('active'));
                contents.forEach(c => c.classList.remove('active'));

                btn.classList.add('active');
                const activeContent = container.querySelector(`.tab-content[data-tab="${targetTab}"]`);
                if (activeContent) activeContent.classList.add('active');
            });
        });
    });
}

// --- ScrollSpy Active Link Highlighter ---
function initScrollSpy() {
    const sections = document.querySelectorAll('section[id], div[id^="engine-"], div[id^="snippet-"]');
    const navLinks = document.querySelectorAll('.sidebar-nav-link');

    window.addEventListener('scroll', () => {
        let currentSectionId = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 130;
            const sectionHeight = section.offsetHeight;
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                currentSectionId = section.getAttribute('id');
            }
        });

        if (currentSectionId) {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${currentSectionId}`) {
                    link.classList.add('active');
                }
            });
        }
    }, { passive: true });
}

// --- Command Palette Search (Cmd+K / Ctrl+K) ---
function initCommandPalette() {
    const modal = document.getElementById('search-modal');
    const trigger = document.getElementById('search-trigger-btn');
    const input = document.getElementById('palette-search-input');
    const resultsContainer = document.getElementById('palette-results-list');

    if (!modal || !input || !resultsContainer) return;

    const searchableItems = [
        // Guides & Setup
        { title: "Installation Options (pip, uv, poetry)", category: "Setup", href: "#section-quickstart" },
        { title: "Environment Variables Configuration", category: "Config", href: "#section-env-vars" },
        { title: "Fail-Safe Fallbacks & Robustness Matrix", category: "Fallbacks", href: "#section-fallbacks" },
        { title: "Capability Matrix (L0 to L4 Degradation)", category: "Architecture", href: "#section-capabilities" },
        { title: "Major Supported Models & Architectures", category: "Models", href: "#section-supported-models" },
        { title: "Meta Llama 3 & 3.2 Family", category: "Models", href: "#section-supported-models" },
        { title: "Alibaba Qwen 2.5 Family", category: "Models", href: "#section-supported-models" },
        { title: "Mistral & Mixtral MoE Family", category: "Models", href: "#section-supported-models" },
        { title: "Google Gemma & Gemma 2 Family", category: "Models", href: "#section-supported-models" },
        { title: "EleutherAI Pythia & GPT-NeoX", category: "Models", href: "#section-supported-models" },

        // Recipes & Snippets
        { title: "Recipe 1: Zero-Copy GPU Memory Attach", category: "Recipe", href: "#snippet-gpu-attach" },
        { title: "Recipe 2: HuggingFace Pretrained Model Loading", category: "Recipe", href: "#snippet-hf-pretrained" },
        { title: "Recipe 3: Gated Models with HF Token (Llama 3)", category: "Recipe", href: "#snippet-gated-models" },
        { title: "Recipe 4: Custom PyTorch Module (AutoHooker)", category: "Recipe", href: "#snippet-bare-pytorch" },
        { title: "Recipe 5: Google Colab Proxy Launch", category: "Recipe", href: "#snippet-colab" },
        { title: "Recipe 6: Activation Steering Vector Injection", category: "Recipe", href: "#snippet-steering" },
        { title: "Recipe 7: Automated Causal ROME Patching Sweeps", category: "Recipe", href: "#snippet-causal-sweep" },
        { title: "Recipe 8: Induction Head Auto-Detection Sweep", category: "Recipe", href: "#snippet-induction-sweep" },
        { title: "Recipe 9: Custom Architecture Strategy Plugin", category: "Recipe", href: "#snippet-custom-plugin" },
        { title: "Recipe 10: PDF Diagnostic Report Export", category: "Recipe", href: "#snippet-pdf-export" },

        // Engines
        { title: "Logit Lens & Entropy Dynamics Engine", category: "Engine", href: "#engine-logit-lens" },
        { title: "Residual Stream Inspector & Vector Drift", category: "Engine", href: "#engine-residual" },
        { title: "Attention Head Explorer & Arc Diagrams", category: "Engine", href: "#engine-attention" },
        { title: "MLP Neuron Explorer & Prompt Text Lighting", category: "Engine", href: "#engine-neurons" },
        { title: "Causal Interventions & ROME Tracing", category: "Engine", href: "#engine-causal" },
        { title: "Architecture Topology & Parameter Shares", category: "Engine", href: "#engine-topology" },

        // APIs & Reference
        { title: "interplens.launch() Signature & Args", category: "Python API", href: "#section-api-python" },
        { title: "interplens.inspect() Pure-Python API", category: "Python API", href: "#section-api-python" },
        { title: "interplens.analysis Submodule Methods", category: "Python API", href: "#section-api-python" },
        { title: "interplens.utils Device & PDF Tools", category: "Python API", href: "#section-api-python" },
        { title: "FastAPI REST & WebSocket Endpoints", category: "REST API", href: "#section-api-rest" },
        { title: "CLI Console Script Flags (interplens launch)", category: "CLI", href: "#section-cli-options" },
        { title: "Structured Exception Hierarchy", category: "Exceptions", href: "#section-exceptions" },
    ];

    function openModal() {
        modal.classList.add('open');
        input.value = '';
        renderResults(searchableItems);
        setTimeout(() => input.focus(), 50);
    }

    function closeModal() {
        modal.classList.remove('open');
    }

    function renderResults(items) {
        if (!items.length) {
            resultsContainer.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 0.9rem;">No matching topics found.</div>`;
            return;
        }

        resultsContainer.innerHTML = items.map((item, idx) => `
            <a href="${item.href}" class="search-result-item ${idx === 0 ? 'highlighted' : ''}" data-index="${idx}">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
                    <span>${item.title}</span>
                </div>
                <span class="search-result-category">${item.category}</span>
            </a>
        `).join('');

        resultsContainer.querySelectorAll('.search-result-item').forEach(link => {
            link.addEventListener('click', () => closeModal());
        });
    }

    if (trigger) {
        trigger.addEventListener('click', openModal);
    }

    // Keyboard Navigation
    window.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            if (modal.classList.contains('open')) {
                closeModal();
            } else {
                openModal();
            }
        } else if (e.key === 'Escape' && modal.classList.contains('open')) {
            closeModal();
        }
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    input.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        if (!query) {
            renderResults(searchableItems);
            return;
        }

        const filtered = searchableItems.filter(item => 
            item.title.toLowerCase().includes(query) || 
            item.category.toLowerCase().includes(query)
        );
        renderResults(filtered);
    });
}
