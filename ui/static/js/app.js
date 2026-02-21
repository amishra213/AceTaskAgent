/**
 * AceTaskAgent UI - Main Application
 * Handles routing, WebSocket connection, and shared utilities.
 */

const App = {
    ws: null,
    currentPage: 'langgraph',
    reconnectAttempts: 0,
    maxReconnect: 10,
    handlers: {},

    init() {
        this.setupNav();
        this.connectWebSocket();

        // Navigate from hash
        const hash = window.location.hash.replace('#', '');
        if (hash) this.navigate(hash);
    },

    // ---- Navigation ----
    setupNav() {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.dataset.page;
                this.navigate(page);
            });
        });
    },

    navigate(page) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

        // Show target
        const pageEl = document.getElementById(`page-${page}`);
        const navEl = document.querySelector(`[data-page="${page}"]`);

        if (pageEl) {
            pageEl.classList.add('active');
            this.currentPage = page;
            window.location.hash = page;
        }
        if (navEl) navEl.classList.add('active');

        // Refresh page data
        switch (page) {
            case 'alerts': Alerts.load(); break;
            case 'langgraph':
                if (typeof LangGraphViewer !== 'undefined') LangGraphViewer.onActivate();
                break;
        }

        lucide.createIcons();
    },

    // ---- WebSocket ----
    connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.reconnectAttempts = 0;
            this.updateConnectionStatus(true);
        };

        this.ws.onclose = () => {
            this.updateConnectionStatus(false);
            this.scheduleReconnect();
        };

        this.ws.onerror = () => {
            this.updateConnectionStatus(false);
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this.handleWSMessage(msg);
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };

        // Ping every 25s
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 25000);
    },

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnect) return;
        this.reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
        setTimeout(() => this.connectWebSocket(), delay);
    },

    updateConnectionStatus(connected) {
        const el = document.getElementById('ws-status');
        const dot = el.querySelector('.status-dot');
        const text = el.querySelector('.status-text');
        dot.className = `status-dot ${connected ? 'connected' : 'disconnected'}`;
        text.textContent = connected ? 'Connected' : 'Disconnected';
    },

    handleWSMessage(msg) {
        const { type, data } = msg;

        // Dispatch to handlers
        switch (type) {
            case 'connected':
                if (data.alerts) Alerts.updateBadge(data.alerts.length);
                break;

            case 'execution_started':
            case 'execution_update':
            case 'execution_completed':
            case 'execution_failed':
            case 'execution_cancelled':
                if (data.alert) {
                    Alerts.addAlert(data.alert);
                    this.showToast(data.alert.severity, data.alert.title, data.alert.message);
                }
                break;
        }

        // Notify registered handlers
        if (this.handlers[type]) {
            this.handlers[type].forEach(fn => fn(data));
        }
    },

    on(event, handler) {
        if (!this.handlers[event]) this.handlers[event] = [];
        this.handlers[event].push(handler);
    },

    // ---- API Helper ----
    async api(url, opts = {}) {
        const defaults = {
            headers: { 'Content-Type': 'application/json' },
        };
        const res = await fetch(url, { ...defaults, ...opts });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'API error');
        }
        return res.json();
    },

    // ---- Toast Notifications ----
    showToast(severity, title, message, duration = 5000) {
        const container = document.getElementById('toast-container');
        const id = 'toast-' + Date.now();

        const icons = {
            info: 'info',
            success: 'check-circle-2',
            warning: 'alert-triangle',
            error: 'alert-circle',
            critical: 'alert-circle'
        };

        const toast = document.createElement('div');
        toast.className = `toast ${severity === 'critical' ? 'error' : severity}`;
        toast.id = id;
        toast.innerHTML = `
            <i data-lucide="${icons[severity] || 'info'}" class="toast-icon"></i>
            <span class="toast-text"><strong>${this.esc(title)}</strong> ${this.esc(message)}</span>
            <i data-lucide="x" class="toast-close" onclick="App.dismissToast('${id}')"></i>
        `;

        container.appendChild(toast);
        lucide.createIcons({ node: toast });

        setTimeout(() => this.dismissToast(id), duration);
    },

    dismissToast(id) {
        const toast = document.getElementById(id);
        if (toast) {
            toast.style.animation = 'toastOut 0.3s ease forwards';
            setTimeout(() => toast.remove(), 300);
        }
    },

    // ---- Utilities ----
    esc(str) {
        if (!str) return '';
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    },

    timeAgo(isoStr) {
        if (!isoStr) return '';
        const now = new Date();
        const then = new Date(isoStr);
        const diff = Math.floor((now - then) / 1000);
        if (diff < 60) return 'just now';
        if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
        return `${Math.floor(diff/86400)}d ago`;
    },

    formatDuration(ms) {
        if (!ms) return '-';
        if (ms < 1000) return `${ms}ms`;
        return `${(ms / 1000).toFixed(1)}s`;
    }
};
