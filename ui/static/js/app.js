/**
 * AceTaskAgent UI - Main Application
 * Handles routing, WebSocket connection, and shared utilities.
 */

const App = {
    ws: null,
    currentPage: 'dashboard',
    reconnectAttempts: 0,
    maxReconnect: 10,
    handlers: {},

    init() {
        this.setupNav();
        this.connectWebSocket();
        this.loadDashboard();

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
            case 'dashboard': this.loadDashboard(); break;
            case 'designer': Designer.init(); break;
            case 'executions': Executions.load(); break;
            case 'monitor': Monitor.refresh(); break;
            case 'alerts': Alerts.load(); break;
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
                if (data.stats) this.updateStats(data.stats);
                if (data.alerts) Alerts.updateBadge(data.alerts.length);
                break;

            case 'execution_started':
            case 'execution_update':
            case 'execution_completed':
            case 'execution_failed':
            case 'execution_cancelled':
                Executions.handleUpdate(data);
                Monitor.handleUpdate(type, data);
                this.loadDashboard();
                if (data.alert) {
                    Alerts.addAlert(data.alert);
                    this.showToast(data.alert.severity, data.alert.title, data.alert.message);
                }
                break;

            case 'task_started':
            case 'task_completed':
            case 'task_failed':
            case 'task_progress':
                Monitor.handleTaskUpdate(type, data);
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

    // ---- Dashboard ----
    async loadDashboard() {
        try {
            const [wfRes, execRes, alertRes] = await Promise.all([
                this.api('/api/workflows'),
                this.api('/api/executions'),
                this.api('/api/alerts?limit=10'),
            ]);

            document.getElementById('stat-workflows').textContent = wfRes.workflows?.length || 0;

            const execs = execRes.executions || [];
            const running = execs.filter(e => e.status === 'running').length;
            const completed = execs.filter(e => e.status === 'completed').length;
            const failed = execs.filter(e => e.status === 'failed').length;

            document.getElementById('stat-running').textContent = running;
            document.getElementById('stat-completed').textContent = completed;
            document.getElementById('stat-failed').textContent = failed;

            // Recent executions
            const recentExecEl = document.getElementById('recent-executions-list');
            if (execs.length === 0) {
                recentExecEl.innerHTML = '<div class="empty-state small"><p>No executions yet</p></div>';
            } else {
                recentExecEl.innerHTML = execs.slice(0, 5).map(e => `
                    <div class="execution-card" onclick="App.navigate('executions')">
                        <div class="exec-header">
                            <span class="exec-title">${this.esc(e.workflow_name)}</span>
                            <span class="exec-status ${e.status}">${e.status}</span>
                        </div>
                        <div class="exec-meta">
                            <span>${e.tasks?.length || 0} tasks</span>
                            <span>${this.timeAgo(e.started_at)}</span>
                        </div>
                        <div class="exec-progress-bar">
                            <div class="exec-progress-fill ${e.status}" style="width:${e.progress || 0}%"></div>
                        </div>
                    </div>
                `).join('');
            }

            // Recent alerts
            const alerts = alertRes.alerts || [];
            const alertsEl = document.getElementById('recent-alerts-list');
            if (alerts.length === 0) {
                alertsEl.innerHTML = '<div class="empty-state small"><p>No alerts</p></div>';
            } else {
                alertsEl.innerHTML = alerts.slice(0, 5).map(a => `
                    <div class="alert-card">
                        <div class="alert-icon ${a.severity}"><i data-lucide="${this.alertIcon(a.severity)}"></i></div>
                        <div class="alert-content">
                            <div class="alert-title">${this.esc(a.title)}</div>
                            <div class="alert-message">${this.esc(a.message)}</div>
                        </div>
                        <span class="alert-time">${this.timeAgo(a.timestamp)}</span>
                    </div>
                `).join('');
            }

            lucide.createIcons();
        } catch (e) {
            console.error('Dashboard load error:', e);
        }
    },

    updateStats(stats) {
        if (stats.running !== undefined) document.getElementById('stat-running').textContent = stats.running;
        if (stats.completed !== undefined) document.getElementById('stat-completed').textContent = stats.completed;
        if (stats.failed !== undefined) document.getElementById('stat-failed').textContent = stats.failed;
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

    alertIcon(severity) {
        return { info: 'info', warning: 'alert-triangle', error: 'alert-circle', critical: 'zap' }[severity] || 'info';
    },

    formatDuration(ms) {
        if (!ms) return '-';
        if (ms < 1000) return `${ms}ms`;
        return `${(ms / 1000).toFixed(1)}s`;
    }
};
