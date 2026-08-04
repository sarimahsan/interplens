// InterpLens Documentation Portal • Interactive Logic & Theme System

document.addEventListener('DOMContentLoaded', () => {
    initThemeManager();
    initCopyButtons();
    initTabSwitchers();
    initScrollSpy();
    initSearchFilter();
});

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
                ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`
                : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
        }
    }
}

// --- Copy Code Blocks ---
function initCopyButtons() {
    document.querySelectorAll('.copy-pill-btn, .code-copy-btn, .code-copy-button').forEach(btn => {
        btn.addEventListener('click', () => {
            const container = btn.closest('.spread-code-box') || btn.closest('.code-block') || btn.closest('.code-display-container');
            const codeBlock = container ? container.querySelector('code') : null;
            if (codeBlock) {
                navigator.clipboard.writeText(codeBlock.innerText).then(() => {
                    const originalText = btn.innerHTML;
                    btn.innerHTML = `<span style="color: var(--brand-primary);">✓ Copied!</span>`;
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                    }, 2000);
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
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.sidebar-link');

    window.addEventListener('scroll', () => {
        let currentSectionId = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                currentSectionId = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSectionId}`) {
                link.classList.add('active');
            }
        });
    });
}

// --- Search Filter ---
function initSearchFilter() {
    const searchInput = document.getElementById('docs-search-input');
    if (!searchInput) return;

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        const cards = document.querySelectorAll('.feature-card, .doc-section');

        cards.forEach(card => {
            const text = card.innerText.toLowerCase();
            if (text.includes(query)) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    });
}
