// InterpLens Documentation — Minimal Vanilla JS (Tabs, Copy, Theme, Search, Scroll-Spy)

document.addEventListener('DOMContentLoaded', () => {
    // 0. Smooth Reading Progress Bar & Gentle Scroll Reveal
    const progressWrap = document.createElement('div');
    progressWrap.className = 'scroll-progress';
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress-bar';
    progressWrap.appendChild(progressBar);
    document.body.appendChild(progressWrap);

    let rafPending = false;
    function updateScrollProgress() {
        rafPending = false;
        const doc = document.documentElement;
        const maxScroll = Math.max(1, doc.scrollHeight - doc.clientHeight);
        const pct = Math.min(100, Math.max(0, (window.scrollY / maxScroll) * 100));
        progressBar.style.width = `${pct.toFixed(2)}%`;
    }
    window.addEventListener('scroll', () => {
        if (!rafPending) {
            rafPending = true;
            requestAnimationFrame(updateScrollProgress);
        }
    }, { passive: true });
    updateScrollProgress();

    const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!prefersReduced && 'IntersectionObserver' in window) {
        const revealSections = Array.from(document.querySelectorAll('.docs-section'));
        revealSections.forEach((sec, idx) => {
            if (idx === 0) return; // keep hero immediately visible with zero delay
            sec.classList.add('reveal-ready');
        });
        const revealObserver = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                    revealObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.06, rootMargin: '0px 0px -30px 0px' });
        revealSections.forEach((sec, idx) => {
            if (idx > 0) revealObserver.observe(sec);
        });
    }

    // 1. Theme Toggle
    const html = document.documentElement;
    const themeBtn = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('interplens_docs_theme') || 'dark';
    html.setAttribute('data-theme', savedTheme);
    if (themeBtn) themeBtn.textContent = savedTheme === 'dark' ? 'Light' : 'Dark';

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('interplens_docs_theme', next);
            themeBtn.textContent = next === 'dark' ? 'Light' : 'Dark';
        });
    }

    // 2. Mobile Sidebar Drawer
    const mobileBtn = document.getElementById('mobile-menu-toggle');
    const sidebar = document.getElementById('docs-sidebar');
    if (mobileBtn && sidebar) {
        mobileBtn.addEventListener('click', () => sidebar.classList.toggle('open'));
        sidebar.querySelectorAll('a').forEach(a => {
            a.addEventListener('click', () => sidebar.classList.remove('open'));
        });
    }

    // 3. Quickstart Code Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const panel = document.getElementById(target);
            if (panel) panel.classList.add('active');
        });
    });

    // 4. Copy Buttons (Inline + Code Blocks)
    async function copyText(text, btn) {
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
            }
            const orig = btn.textContent;
            btn.textContent = 'Copied';
            setTimeout(() => { btn.textContent = orig; }, 1400);
        } catch (_) {}
    }

    document.querySelectorAll('.copy-inline-btn').forEach(btn => {
        btn.addEventListener('click', () => copyText(btn.getAttribute('data-copy') || '', btn));
    });

    document.querySelectorAll('.code-copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const code = btn.closest('.code-block')?.querySelector('pre code')?.textContent || '';
            copyText(code.trim(), btn);
        });
    });

    // 5. Scroll-Spy for Sidebar Links
    const navLinks = Array.from(document.querySelectorAll('.docs-sidebar .nav-link'));
    const sections = navLinks
        .map(l => document.querySelector(l.getAttribute('href')))
        .filter(Boolean);

    if (sections.length && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = `#${entry.target.id}`;
                    navLinks.forEach(l => l.classList.toggle('active', l.getAttribute('href') === id));
                }
            });
        }, { rootMargin: '-20% 0px -65% 0px' });
        sections.forEach(sec => observer.observe(sec));
    }

    // 6. Cmd+K Quick Search Modal
    const searchModal = document.getElementById('search-modal');
    const searchTrigger = document.getElementById('search-trigger');
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    const indexItems = [
        { title: 'Overview & Benchmarks', href: '#overview', tag: 'Intro' },
        { title: "What's New in v2.0.0", href: '#whats-new', tag: 'Release' },
        { title: 'Full v2.0.0 Release Notes & Changelog', href: 'releases.html', tag: 'Page' },
        { title: 'Quickstart (CLI, Python, Colab A100/T4)', href: '#quickstart', tag: 'Setup' },
        { title: 'Analysis Engines (Logit Lens, PCA, Attention, Hardware)', href: '#engines', tag: 'Engines' },
        { title: 'Direct Logit Attribution (DLA) & Attribution Patching (AtP)', href: '#fast-circuits', tag: 'v2.0' },
        { title: 'Contrastive Activation Steering (CAA)', href: '#steering', tag: 'v2.0' },
        { title: 'Supported Models (Qwen, Llama, Mixtral, OLMoE, Gemma, GPT-2)', href: '#models', tag: 'Models' },
        { title: 'Python API (launch, inspect, stop_server)', href: '#python-api', tag: 'API' },
        { title: 'Environment Variables & BibTeX Citation', href: '#config', tag: 'Config' },
    ];

    function renderSearch(q) {
        if (!searchResults) return;
        const term = (q || '').toLowerCase().trim();
        const matches = term
            ? indexItems.filter(i => i.title.toLowerCase().includes(term) || i.tag.toLowerCase().includes(term))
            : indexItems;
        searchResults.innerHTML = matches.length
            ? matches.map(m => `<a class="search-item" href="${m.href}"><span>${m.title}</span><span>${m.tag}</span></a>`).join('')
            : '<div style="padding: 14px; font-size: 13px; color: var(--text-muted);">No matching sections.</div>';
        searchResults.querySelectorAll('a').forEach(a => {
            a.addEventListener('click', () => closeSearch());
        });
    }

    function openSearch() {
        if (!searchModal) return;
        searchModal.classList.add('open');
        renderSearch('');
        if (searchInput) {
            searchInput.value = '';
            searchInput.focus();
        }
    }

    function closeSearch() {
        if (searchModal) searchModal.classList.remove('open');
    }

    if (searchTrigger) searchTrigger.addEventListener('click', openSearch);
    if (searchInput) searchInput.addEventListener('input', e => renderSearch(e.target.value));
    if (searchModal) {
        searchModal.addEventListener('click', e => {
            if (e.target === searchModal) closeSearch();
        });
    }
    document.addEventListener('keydown', e => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            openSearch();
        } else if (e.key === 'Escape') {
            closeSearch();
        }
    });
});
