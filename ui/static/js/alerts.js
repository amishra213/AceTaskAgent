/**
 * Alerts - Notification and alert management.
 */

const Alerts = {
    alerts: [],
    filter: 'all',

    async load() {
        try {
            const res = await App.api('/api/alerts?limit=100');
            this.alerts = res.alerts || [];
            this.renderList();
            this.updateBadge(this.alerts.filter(a => !a.acknowledged).length);
        } catch (e) {
            console.error('Failed to load alerts:', e);
        }
    },

    filter(severity) {
        this.filter = severity;
        document.querySelectorAll('#page-alerts .filter-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.filter === severity);
        });
        this.renderList();
    },

    renderList() {
        const el = document.getElementById('alerts-list');
        let list = this.alerts;

        if (this.filter !== 'all') {
            list = list.filter(a => a.severity === this.filter);
        }

        if (list.length === 0) {
            el.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="bell-off"></i>
                    <p>${this.filter === 'all' ? 'No alerts at this time' : `No ${this.filter} alerts`}</p>
                </div>
            `;
            lucide.createIcons();
            return;
        }

        el.innerHTML = list.map(alert => {
            const icon = App.alertIcon(alert.severity);
            return `
                <div class="alert-card ${alert.acknowledged ? 'acknowledged' : ''}" id="alert-${alert.id}">
                    <div class="alert-icon ${alert.severity}">
                        <i data-lucide="${icon}"></i>
                    </div>
                    <div class="alert-content">
                        <div class="alert-title">${App.esc(alert.title)}</div>
                        <div class="alert-message" title="${App.esc(alert.message)}">${App.esc(alert.message)}</div>
                        ${alert.source ? `<div style="font-size:10px;color:var(--text-muted);margin-top:3px">Source: ${App.esc(alert.source)}</div>` : ''}
                    </div>
                    <span class="alert-time">${App.timeAgo(alert.timestamp)}</span>
                    <div class="alert-actions">
                        ${!alert.acknowledged ? `
                            <button class="btn btn-icon" title="Acknowledge" onclick="Alerts.acknowledge('${alert.id}')">
                                <i data-lucide="check"></i>
                            </button>
                        ` : ''}
                        ${alert.execution_id ? `
                            <button class="btn btn-icon" title="View Execution" onclick="Monitor.activeExecutionId='${alert.execution_id}';App.navigate('monitor')">
                                <i data-lucide="external-link"></i>
                            </button>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');

        lucide.createIcons();
    },

    async acknowledge(alertId) {
        try {
            await App.api('/api/alerts/acknowledge', {
                method: 'POST',
                body: JSON.stringify({ alert_id: alertId }),
            });
            // Update local state
            const alert = this.alerts.find(a => a.id === alertId);
            if (alert) alert.acknowledged = true;
            this.renderList();
            this.updateBadge(this.alerts.filter(a => !a.acknowledged).length);
        } catch (e) {
            App.showToast('error', 'Error', e.message);
        }
    },

    async clearAll() {
        if (!confirm('Clear all alerts?')) return;
        try {
            await App.api('/api/alerts/clear', { method: 'POST' });
            this.alerts = [];
            this.renderList();
            this.updateBadge(0);
            App.showToast('info', 'Cleared', 'All alerts cleared.');
        } catch (e) {
            App.showToast('error', 'Error', e.message);
        }
    },

    addAlert(alertData) {
        // Add from WebSocket event
        this.alerts.unshift(alertData);
        this.updateBadge(this.alerts.filter(a => !a.acknowledged).length);

        if (App.currentPage === 'alerts') {
            this.renderList();
        }
    },

    updateBadge(count) {
        const badge = document.getElementById('alert-badge');
        if (count > 0) {
            badge.textContent = count > 99 ? '99+' : count;
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    },
};
