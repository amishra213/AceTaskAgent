/**
 * Monitor - Real-time execution monitoring dashboard.
 */

const Monitor = {
    activeExecutionId: null,
    logs: [],

    async refresh() {
        if (this.activeExecutionId) {
            await this.loadExecution(this.activeExecutionId);
        } else {
            // Show latest running execution
            try {
                const res = await App.api('/api/executions?status=running&limit=1');
                const execs = res.executions || [];
                if (execs.length > 0) {
                    this.activeExecutionId = execs[0].id;
                    await this.loadExecution(execs[0].id);
                } else {
                    // Show latest of any
                    const allRes = await App.api('/api/executions?limit=1');
                    const allExecs = allRes.executions || [];
                    if (allExecs.length > 0) {
                        this.activeExecutionId = allExecs[0].id;
                        await this.loadExecution(allExecs[0].id);
                    } else {
                        this.renderEmpty();
                    }
                }
            } catch (e) {
                console.error('Monitor refresh error:', e);
            }
        }
    },

    async loadExecution(execId) {
        try {
            const exec = await App.api(`/api/executions/${execId}`);
            this.renderMonitor(exec);
        } catch (e) {
            console.error('Failed to load execution:', e);
        }
    },

    renderEmpty() {
        document.getElementById('monitor-active-tasks').innerHTML = `
            <div class="empty-state small"><i data-lucide="loader"></i><p>No active tasks</p></div>
        `;
        document.getElementById('monitor-log').innerHTML = `
            <div class="empty-state small"><i data-lucide="terminal"></i><p>Logs will appear here</p></div>
        `;
        document.getElementById('monitor-progress').innerHTML = `
            <div class="empty-state small"><i data-lucide="bar-chart-3"></i><p>No tasks to monitor</p></div>
        `;
        lucide.createIcons();
    },

    renderMonitor(exec) {
        const tasks = exec.tasks || [];
        const activeTasks = tasks.filter(t => t.status === 'running');
        const logs = exec.logs || [];

        // Active Tasks
        const activeEl = document.getElementById('monitor-active-tasks');
        if (activeTasks.length === 0 && exec.status !== 'running') {
            const statusIcon = exec.status === 'completed' ? 'check-circle-2' : 
                              exec.status === 'failed' ? 'alert-circle' : 'loader';
            const statusColor = exec.status === 'completed' ? 'var(--accent-green)' : 
                               exec.status === 'failed' ? 'var(--accent-red)' : 'var(--text-muted)';
            activeEl.innerHTML = `
                <div style="padding:16px;text-align:center">
                    <div style="margin-bottom:8px;color:${statusColor}"><i data-lucide="${statusIcon}"></i></div>
                    <div style="font-size:14px;font-weight:600;margin-bottom:4px">
                        ${App.esc(exec.workflow_name)}
                    </div>
                    <div style="font-size:12px;color:var(--text-secondary)">
                        Status: <span class="exec-status ${exec.status}">${exec.status}</span>
                    </div>
                    <div style="font-size:12px;color:var(--text-muted);margin-top:8px">
                        ${exec.objective ? App.esc(exec.objective.substring(0, 100)) : ''}
                    </div>
                </div>
            `;
        } else if (activeTasks.length === 0) {
            activeEl.innerHTML = `
                <div style="padding:16px;text-align:center">
                    <div style="color:var(--accent-blue);margin-bottom:8px"><i data-lucide="loader" class="spin"></i></div>
                    <div style="font-size:13px;color:var(--text-secondary)">Preparing tasks...</div>
                </div>
            `;
        } else {
            activeEl.innerHTML = activeTasks.map(t => `
                <div style="padding:10px;border-bottom:1px solid var(--border-color)">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <span style="font-weight:600;font-size:13px">${App.esc(t.name)}</span>
                        <span style="font-size:11px;color:var(--accent-blue)">${Math.round(t.progress || 0)}%</span>
                    </div>
                    <div class="task-progress-bar">
                        <div class="task-progress-fill" style="width:${t.progress || 0}%;background:var(--accent-blue)"></div>
                    </div>
                    <div style="font-size:11px;color:var(--text-muted);margin-top:4px">${t.agent_type} agent</div>
                </div>
            `).join('');
        }

        // Logs
        const logEl = document.getElementById('monitor-log');
        if (logs.length === 0) {
            logEl.innerHTML = `<div class="empty-state small"><i data-lucide="terminal"></i><p>No logs yet</p></div>`;
        } else {
            logEl.innerHTML = logs.slice(-30).map(l => `
                <div class="log-entry ${l.level || 'info'}">
                    <span class="log-timestamp">${new Date(l.timestamp).toLocaleTimeString()}</span>
                    ${App.esc(l.message)}
                </div>
            `).join('');
            logEl.scrollTop = logEl.scrollHeight;
        }

        // Progress bars for all tasks
        const progressEl = document.getElementById('monitor-progress');
        if (tasks.length === 0) {
            progressEl.innerHTML = `<div class="empty-state small"><p>No tasks</p></div>`;
        } else {
            progressEl.innerHTML = tasks.map(t => {
                const color = t.status === 'completed' ? 'var(--accent-green)' :
                             t.status === 'failed' ? 'var(--accent-red)' :
                             t.status === 'running' ? 'var(--accent-blue)' : 'var(--text-muted)';
                const statusIcon = t.status === 'completed' ? '✓' :
                                  t.status === 'failed' ? '✗' :
                                  t.status === 'running' ? '⟳' : '○';
                return `
                    <div class="task-progress-item">
                        <span style="color:${color};font-size:14px;width:20px">${statusIcon}</span>
                        <span class="task-progress-name">${App.esc(t.name)}</span>
                        <div class="task-progress-bar">
                            <div class="task-progress-fill" style="width:${t.progress || 0}%;background:${color}"></div>
                        </div>
                        <span class="task-progress-pct">${Math.round(t.progress || 0)}%</span>
                        <span style="min-width:50px;font-size:11px;color:var(--text-muted);text-align:right">
                            ${t.duration_ms ? App.formatDuration(t.duration_ms) : '-'}
                        </span>
                    </div>
                `;
            }).join('');
        }

        lucide.createIcons();
    },

    handleUpdate(type, data) {
        if (!data.execution) return;

        if (this.activeExecutionId === data.execution.id) {
            if (App.currentPage === 'monitor') {
                this.renderMonitor(data.execution);
            }
        }

        // Auto-select new running execution
        if (type === 'execution_started') {
            this.activeExecutionId = data.execution.id;
            if (App.currentPage === 'monitor') {
                this.renderMonitor(data.execution);
            }
        }
    },

    handleTaskUpdate(type, data) {
        if (!data.execution_id) return;

        // Refresh if we're watching this execution
        if (this.activeExecutionId === data.execution_id && App.currentPage === 'monitor') {
            this.loadExecution(data.execution_id);
        }
    },
};
