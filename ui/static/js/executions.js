/**
 * Executions - Workflow execution list and detail view.
 */

const Executions = {
    executions: [],
    filter: 'all',

    async load() {
        try {
            const res = await App.api('/api/executions?limit=50');
            this.executions = res.executions || [];
            this.renderList();
        } catch (e) {
            console.error('Failed to load executions:', e);
        }

        // Set up filter buttons
        document.querySelectorAll('#page-executions .filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('#page-executions .filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.filter = btn.dataset.filter;
                this.renderList();
            });
        });
    },

    async refresh() {
        await this.load();
        App.showToast('info', 'Refreshed', 'Execution list updated.');
    },

    renderList() {
        const el = document.getElementById('executions-list');
        let list = this.executions;

        if (this.filter !== 'all') {
            list = list.filter(e => e.status === this.filter);
        }

        if (list.length === 0) {
            el.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="play-circle"></i>
                    <p>${this.filter === 'all' ? 'No executions yet. Run a workflow to see results here.' : `No ${this.filter} executions.`}</p>
                </div>
            `;
            lucide.createIcons();
            return;
        }

        el.innerHTML = list.map(exec => {
            const tasks = exec.tasks || [];
            const completedTasks = tasks.filter(t => t.status === 'completed').length;
            const failedTasks = tasks.filter(t => t.status === 'failed').length;
            const runningTasks = tasks.filter(t => t.status === 'running').length;

            return `
                <div class="execution-card" onclick="Executions.showDetail('${exec.id}')">
                    <div class="exec-header">
                        <span class="exec-title">${App.esc(exec.workflow_name)}</span>
                        <span class="exec-status ${exec.status}">${exec.status}</span>
                    </div>
                    <div class="exec-meta">
                        <span>${App.esc(exec.objective?.substring(0, 60) || '')}${exec.objective?.length > 60 ? '...' : ''}</span>
                    </div>
                    <div class="exec-meta" style="margin-top:6px">
                        <span>${tasks.length} tasks</span>
                        <span>${completedTasks} done</span>
                        ${failedTasks ? `<span style="color:var(--accent-red)">${failedTasks} failed</span>` : ''}
                        ${runningTasks ? `<span style="color:var(--accent-blue)">${runningTasks} running</span>` : ''}
                        <span>${App.timeAgo(exec.started_at)}</span>
                    </div>
                    <div class="exec-progress-bar">
                        <div class="exec-progress-fill ${exec.status}" style="width:${exec.progress || 0}%"></div>
                    </div>
                    <div class="exec-tasks">
                        ${tasks.map(t => `
                            <span class="exec-task-chip ${t.status}">
                                ${App.esc(t.name)}
                            </span>
                        `).join('')}
                    </div>
                    ${exec.status === 'running' ? `
                        <div style="margin-top:10px">
                            <button class="btn btn-sm btn-danger" onclick="event.stopPropagation();Executions.cancel('${exec.id}')">
                                <i data-lucide="square"></i> Cancel
                            </button>
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');

        lucide.createIcons();
    },

    async showDetail(execId) {
        // Navigate to monitor with this execution
        Monitor.activeExecutionId = execId;
        App.navigate('monitor');
    },

    async cancel(execId) {
        try {
            await App.api(`/api/executions/${execId}/cancel`, { method: 'POST' });
            App.showToast('warning', 'Cancelled', 'Execution cancelled.');
            await this.load();
        } catch (e) {
            App.showToast('error', 'Cancel Failed', e.message);
        }
    },

    handleUpdate(data) {
        if (!data.execution) return;
        const exec = data.execution;

        // Update or add execution
        const idx = this.executions.findIndex(e => e.id === exec.id);
        if (idx >= 0) {
            this.executions[idx] = exec;
        } else {
            this.executions.unshift(exec);
        }

        if (App.currentPage === 'executions') {
            this.renderList();
        }
    },
};
