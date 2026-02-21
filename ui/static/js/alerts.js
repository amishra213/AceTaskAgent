/**
 * Alerts / Log Viewer
 *
 * Reads structured log entries from /api/logs/recent and renders them
 * as a filterable, searchable table grouped by level:
 *   All | Info | Warning | Error | Critical
 *
 * Each row shows: timestamp · level badge · message · category · source · tags
 * Clicking a row expands the full metadata block.
 */

const Alerts = {
    // ── State ─────────────────────────────────────────────────────────────
    _entries:       [],      // all entries from last fetch
    _filtered:      [],      // after level + search filter
    _currentLevel:  'all',
    _searchText:    '',
    _limit:         100,
    _autoRefresh:   false,
    _autoTimer:     null,
    _expanded:      new Set(),  // row indices that are expanded

    // ── Level config ──────────────────────────────────────────────────────
    _levels: {
        info:     { label: 'INFO',     color: 'var(--accent-blue)',   icon: 'info' },
        warning:  { label: 'WARN',     color: 'var(--accent-yellow)', icon: 'alert-triangle' },
        error:    { label: 'ERROR',    color: 'var(--accent-red)',    icon: 'x-circle' },
        critical: { label: 'CRITICAL', color: '#ff2d55',              icon: 'alert-octagon' },
        debug:    { label: 'DEBUG',    color: 'var(--text-muted)',    icon: 'bug' },
    },

    // ── Bootstrap ─────────────────────────────────────────────────────────
    async load() {
        await this.refresh();
    },

    async refresh() {
        const listEl = document.getElementById('alerts-list');
        if (listEl) {
            listEl.innerHTML = `
                <div class="log-loading">
                    <span class="lg-parse-spinner"></span>
                    <span>Loading logs…</span>
                </div>`;
        }
        try {
            const params = new URLSearchParams({ limit: this._limit });
            const res = await App.api(`/api/logs/recent?${params}`);
            this._entries = (res.logs || []).reverse(); // newest first
            this._applyFilter();
            this._updateCounts();
            this._renderList();
            this._updateSubtitle();
        } catch (e) {
            if (listEl) {
                listEl.innerHTML = `
                    <div class="empty-state">
                        <i data-lucide="alert-triangle"></i>
                        <p>Failed to load logs: ${App.esc(e.message)}</p>
                    </div>`;
                lucide.createIcons();
            }
        }
    },

    // ── Filter / Search ───────────────────────────────────────────────────
    filter(level) {
        this._currentLevel = level;
        this._expanded.clear();

        // Update stat card active state
        document.querySelectorAll('.log-stat-card').forEach(c => {
            c.classList.toggle('active', c.dataset.filter === level);
        });

        this._applyFilter();
        this._renderList();
        this._updateSubtitle();
    },

    onSearch(text) {
        this._searchText = text.trim().toLowerCase();
        this._expanded.clear();
        this._applyFilter();
        this._renderList();
        this._updateSubtitle();
    },

    onLimitChange(val) {
        this._limit = parseInt(val, 10) || 100;
        this.refresh();
    },

    _applyFilter() {
        let list = this._entries;

        if (this._currentLevel !== 'all') {
            list = list.filter(e =>
                (e.level || '').toLowerCase() === this._currentLevel);
        }

        if (this._searchText) {
            const q = this._searchText;
            list = list.filter(e =>
                (e.message  || '').toLowerCase().includes(q) ||
                (e.category || '').toLowerCase().includes(q) ||
                (e.level    || '').toLowerCase().includes(q) ||
                ((e.source && e.source.module) || '').toLowerCase().includes(q) ||
                ((e.source && e.source.function) || '').toLowerCase().includes(q) ||
                (Array.isArray(e.tags) ? e.tags.join(' ') : '').toLowerCase().includes(q)
            );
        }

        this._filtered = list;
    },

    // ── Count update ──────────────────────────────────────────────────────
    _updateCounts() {
        const counts = { all: this._entries.length, info: 0, warning: 0, error: 0, critical: 0 };
        for (const e of this._entries) {
            const lv = (e.level || '').toLowerCase();
            if (lv in counts) counts[lv]++;
        }
        for (const [k, v] of Object.entries(counts)) {
            const el = document.getElementById(`log-count-${k}`);
            if (el) el.textContent = v;
        }
        // Update nav badge with errors + criticals
        const badgeCount = counts.error + counts.critical;
        const badge = document.getElementById('alert-badge');
        if (badge) {
            if (badgeCount > 0) {
                badge.textContent = badgeCount > 99 ? '99+' : badgeCount;
                badge.classList.remove('hidden');
            } else {
                badge.classList.add('hidden');
            }
        }
    },

    _updateSubtitle() {
        const el = document.getElementById('alerts-subtitle');
        if (!el) return;
        const total = this._entries.length;
        const shown = this._filtered.length;
        const levelStr = this._currentLevel === 'all' ? 'all levels' : this._currentLevel.toUpperCase();
        let txt = `${shown} of ${total} entries · ${levelStr}`;
        if (this._searchText) txt += ` · search: "${this._searchText}"`;
        el.textContent = txt;
    },

    // ── Rendering ─────────────────────────────────────────────────────────
    _renderList() {
        const el = document.getElementById('alerts-list');
        if (!el) return;

        if (this._filtered.length === 0) {
            el.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="file-text"></i>
                    <p>${this._searchText
                        ? `No entries match "${App.esc(this._searchText)}"`
                        : (this._currentLevel === 'all'
                            ? 'No log entries found'
                            : `No ${this._currentLevel.toUpperCase()} entries`)
                    }</p>
                </div>`;
            lucide.createIcons();
            return;
        }

        el.innerHTML = this._filtered.map((entry, idx) => {
            return this._renderEntry(entry, idx);
        }).join('');

        lucide.createIcons();
    },

    _renderEntry(entry, idx) {
        const lv     = (entry.level || 'info').toLowerCase();
        const cfg    = this._levels[lv] || this._levels.info;
        const ts     = this._formatTs(entry.timestamp);
        const msg    = App.esc(entry.message || '—');
        const cat    = App.esc(entry.category || '');
        const srcMod = entry.source && entry.source.module   ? App.esc(entry.source.module)   : '';
        const srcFn  = entry.source && entry.source.function ? App.esc(entry.source.function) : '';
        const srcLn  = entry.source && entry.source.line     ? entry.source.line               : '';
        const tags   = Array.isArray(entry.tags)
            ? entry.tags.map(t => `<span class="log-tag">${App.esc(t)}</span>`).join('')
            : '';
        const isExpanded = this._expanded.has(idx);
        const hasMeta    = entry.metadata && Object.keys(entry.metadata).length > 0;
        const metaJson   = hasMeta
            ? App.esc(JSON.stringify(entry.metadata, null, 2))
            : '';

        const srcStr = [srcMod, srcFn ? `${srcFn}()` : '', srcLn ? `L${srcLn}` : '']
            .filter(Boolean).join(' · ');

        return `
        <div class="log-entry log-entry--${lv} ${isExpanded ? 'expanded' : ''}"
             onclick="Alerts.toggleExpand(${idx})">
            <div class="log-entry-main">
                <span class="log-ts">${ts}</span>
                <span class="log-badge log-badge--${lv}">${cfg.label}</span>
                <span class="log-msg">${msg}</span>
                <div class="log-meta-chips">
                    ${cat ? `<span class="log-chip log-chip--cat">${cat}</span>` : ''}
                    ${srcStr ? `<span class="log-chip log-chip--src">${srcStr}</span>` : ''}
                    ${tags}
                </div>
                ${(hasMeta || entry.correlation_id || entry.session_id)
                    ? `<span class="log-expand-arrow">${isExpanded ? '▲' : '▼'}</span>`
                    : ''}
            </div>
            ${isExpanded ? `
            <div class="log-entry-detail" onclick="event.stopPropagation()">
                ${entry.session_id     ? `<div class="log-detail-row"><span class="log-detail-key">session</span><code>${App.esc(entry.session_id)}</code></div>` : ''}
                ${entry.correlation_id ? `<div class="log-detail-row"><span class="log-detail-key">correlation</span><code>${App.esc(entry.correlation_id)}</code></div>` : ''}
                ${hasMeta ? `
                <div class="log-detail-row log-detail-meta">
                    <span class="log-detail-key">metadata</span>
                    <pre class="log-meta-pre">${metaJson}</pre>
                </div>` : ''}
            </div>` : ''}
        </div>`;
    },

    toggleExpand(idx) {
        if (this._expanded.has(idx)) {
            this._expanded.delete(idx);
        } else {
            this._expanded.add(idx);
        }
        this._renderList();
    },

    // ── Auto-refresh ──────────────────────────────────────────────────────
    toggleAutoRefresh() {
        this._autoRefresh = !this._autoRefresh;
        const btn   = document.getElementById('alerts-autorefresh-btn');
        const label = document.getElementById('alerts-autorefresh-label');

        if (this._autoRefresh) {
            if (btn)   btn.classList.add('active');
            if (label) label.textContent = 'On';
            this._autoTimer = setInterval(() => this.refresh(), 10_000);
        } else {
            if (btn)   btn.classList.remove('active');
            if (label) label.textContent = 'Auto';
            clearInterval(this._autoTimer);
        }
    },

    // ── Utilities ─────────────────────────────────────────────────────────
    _formatTs(ts) {
        if (!ts) return '—';
        try {
            const d = new Date(ts);
            const pad = n => String(n).padStart(2, '0');
            return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} `
                 + `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
        } catch (_) { return ts; }
    },

    // Legacy compat — keep addAlert so WebSocket handler doesn't break
    addAlert(alertData) {
        // Convert a WS alert event into a pseudo-log entry and prepend
        const entry = {
            timestamp:  alertData.timestamp || new Date().toISOString(),
            level:      (alertData.severity || 'info').toUpperCase(),
            message:    `[${alertData.title || 'Alert'}] ${alertData.message || ''}`,
            category:   alertData.source || 'alert',
            tags:       ['websocket'],
            metadata:   {},
            source:     {},
        };
        this._entries.unshift(entry);
        this._applyFilter();
        this._updateCounts();
        if (App.currentPage === 'alerts') this._renderList();
    },

    updateBadge(count) {
        const badge = document.getElementById('alert-badge');
        if (!badge) return;
        if (count > 0) {
            badge.textContent = count > 99 ? '99+' : count;
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    },
};
