/**
 * LangGraph Workflow Viewer
 * Reads the real LangGraph workflow topology from /api/langgraph/workflow and
 * renders it as an interactive SVG DAG with:
 *  - Hierarchical layer layout (Sugiyama-style)
 *  - Color-coded node categories (core / subagent / synthesis / control / terminal)
 *  - Three edge styles (direct / conditional / chain)
 *  - Chain-group highlight overlays
 *  - Pan / zoom via mouse wheel + drag
 *  - Node click → detail panel
 *  - Hover tooltips
 */

const LangGraphViewer = (() => {
    // ── State ────────────────────────────────────────────────────────────────
    let _data        = null;   // raw API response
    let _positions   = {};     // nodeId → {x, y}
    let _zoom        = 1;
    let _panX        = 0;
    let _panY        = 0;
    let _dragging    = false;
    let _lastMouse   = { x: 0, y: 0 };
    let _selected    = null;
    let _showChains  = true;
    let _tooltip     = null;

    // Layout constants
    const NODE_W       = 172;
    const NODE_H       = 56;
    const H_GAP        = 60;   // horizontal gap between nodes in same layer
    const V_GAP        = 90;   // vertical gap between layers
    const LAYER_PAD    = 60;   // top padding

    // ── Helpers ──────────────────────────────────────────────────────────────
    function _svg()       { return document.getElementById('lg-svg'); }
    function _root()      { return document.getElementById('lg-graph-root'); }
    function _container() { return document.getElementById('lg-graph-container'); }

    function _esc(s) {
        return String(s || '')
            .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
    }

    function _svgEl(tag, attrs = {}) {
        const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
        return el;
    }

    // ── Initialise ──────────────────────────────────────────────────────────
    async function init() {
        _setupSvgInteraction();
        _buildTooltip();
        try {
            const res = await App.api('/api/langgraph/workflow');
            _data = res;
            _buildLayout();
            _render();
            _buildLegend();
            _buildStats();
        } catch (e) {
            _root().innerHTML = `
                <text x="50%" y="50%" fill="#ef4444" text-anchor="middle"
                      font-family="Inter,sans-serif" font-size="14">
                    Failed to load workflow: ${_esc(e.message)}
                </text>`;
        }
    }

    // ── Layout ───────────────────────────────────────────────────────────────
    function _buildLayout() {
        // Group nodes by layer
        const layers = {};
        for (const n of _data.nodes) {
            const layer = n.layer ?? 5;
            (layers[layer] = layers[layer] || []).push(n);
        }

        // Order sub-agent nodes sensibly (alphabetical within layer 4)
        const subagentOrder = [
            'execute_task',
            'execute_pdf_task',
            'execute_ocr_task',
            'execute_web_search_task',
            'execute_excel_task',
            'execute_code_interpreter_task',
            'execute_data_extraction_task',
            'execute_problem_solver_task',
            'execute_document_task',
        ];
        if (layers[4]) {
            layers[4].sort((a, b) => {
                const ai = subagentOrder.indexOf(a.id);
                const bi = subagentOrder.indexOf(b.id);
                return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
            });
        }

        const sortedLayers = Object.keys(layers).map(Number).sort((a, b) => a - b);

        // Compute total canvas width needed (widest layer)
        let maxNodesInLayer = 0;
        for (const ln of sortedLayers) maxNodesInLayer = Math.max(maxNodesInLayer, layers[ln].length);
        const canvasW = maxNodesInLayer * (NODE_W + H_GAP) + H_GAP;

        for (const ln of sortedLayers) {
            const layerNodes = layers[ln];
            const count = layerNodes.length;
            const rowW = count * (NODE_W + H_GAP) - H_GAP;
            const startX = (canvasW - rowW) / 2;
            const y = LAYER_PAD + ln * (NODE_H + V_GAP);
            layerNodes.forEach((n, i) => {
                _positions[n.id] = { x: startX + i * (NODE_W + H_GAP), y };
            });
        }

        // Resize SVG
        const totalH = (sortedLayers[sortedLayers.length - 1] + 1) * (NODE_H + V_GAP) + LAYER_PAD * 2;
        const svg = _svg();
        svg.setAttribute('width', canvasW);
        svg.setAttribute('height', totalH);
        svg.setAttribute('viewBox', `0 0 ${canvasW} ${totalH}`);
    }

    // ── Rendering ────────────────────────────────────────────────────────────
    function _render() {
        const root = _root();
        root.innerHTML = '';

        // Layer band backgrounds
        _renderLayerBands(root);

        // Chain overlays (behind edges)
        if (_showChains) _renderChainGroups(root);

        // Edges
        _renderEdges(root);

        // Nodes
        _renderNodes(root);

        // Apply transform
        _applyTransform();
    }

    function _renderLayerBands(root) {
        const layerLabels = {
            0: 'Entry',
            1: 'Task Queue',
            2: 'Analysis',
            3: 'Breakdown / Control',
            4: 'Sub-Agent Execution',
            5: 'Observer / Auto-Synthesis',
            6: 'Aggregation',
            7: 'Synthesis',
            8: 'Debate',
            9: 'End',
        };

        const layers = {};
        for (const n of _data.nodes) {
            const ly = n.layer ?? 5;
            (layers[ly] = layers[ly] || []).push(n);
        }

        const svg = _svg();
        const totalW = parseFloat(svg.getAttribute('width'));
        for (const [ln, nodes] of Object.entries(layers)) {
            const y = LAYER_PAD + Number(ln) * (NODE_H + V_GAP) - 12;
            const g = _svgEl('g', { class: 'lg-layer-band' });

            const rect = _svgEl('rect', {
                x: 0, y: y - 4,
                width: totalW, height: NODE_H + 20,
                rx: 4,
                fill: 'rgba(255,255,255,0.018)',
                stroke: 'rgba(255,255,255,0.04)',
                'stroke-width': 1,
            });
            g.appendChild(rect);

            const label = _svgEl('text', {
                x: 8, y: y + NODE_H / 2 + 1,
                fill: 'rgba(255,255,255,0.18)',
                'font-size': '10',
                'font-family': 'Inter,sans-serif',
                'font-weight': '500',
                'dominant-baseline': 'middle',
            });
            label.textContent = layerLabels[ln] || `Layer ${ln}`;
            g.appendChild(label);
            root.appendChild(g);
        }
    }

    function _renderChainGroups(root) {
        if (!_data.chain_groups) return;
        for (const cg of _data.chain_groups) {
            const positions = cg.nodes.map(id => _positions[id]).filter(Boolean);
            if (positions.length < 2) continue;

            const minX = Math.min(...positions.map(p => p.x)) - 10;
            const minY = Math.min(...positions.map(p => p.y)) - 10;
            const maxX = Math.max(...positions.map(p => p.x + NODE_W)) + 10;
            const maxY = Math.max(...positions.map(p => p.y + NODE_H)) + 10;

            const g = _svgEl('g', { class: 'lg-chain-group', 'data-chain': cg.id });
            const rect = _svgEl('rect', {
                x: minX, y: minY,
                width: maxX - minX, height: maxY - minY,
                rx: 12,
                fill: cg.color + '12',
                stroke: cg.color + '55',
                'stroke-width': 1.5,
                'stroke-dasharray': '6 4',
            });
            g.appendChild(rect);

            const labelEl = _svgEl('text', {
                x: minX + 8, y: minY + 14,
                fill: cg.color,
                'font-size': '9',
                'font-family': 'Inter,sans-serif',
                'font-weight': '700',
                opacity: '0.85',
                'letter-spacing': '0.5',
            });
            labelEl.textContent = cg.label.toUpperCase();
            g.appendChild(labelEl);
            root.appendChild(g);
        }
    }

    function _renderEdges(root) {
        // Bucket edges by target so we can offset parallel edges
        const edgesByPair = {};
        for (const e of _data.edges) {
            const key = `${e.source}→${e.target}`;
            (edgesByPair[key] = edgesByPair[key] || []).push(e);
        }

        // Sort: direct first, then conditional, then chain
        const order = { direct: 0, conditional: 1, chain: 2 };
        const sorted = [..._data.edges].sort((a, b) => (order[a.type] || 0) - (order[b.type] || 0));

        for (const e of sorted) {
            const src = _positions[e.source];
            const tgt = _positions[e.target];
            if (!src || !tgt) continue;

            const key = `${e.source}→${e.target}`;
            const group = edgesByPair[key];
            const idx = group.indexOf(e);
            const offset = (idx - (group.length - 1) / 2) * 12;

            _drawEdge(root, e, src, tgt, offset);
        }
    }

    function _drawEdge(root, edge, src, tgt, offset) {
        const sx = src.x + NODE_W / 2 + offset;
        const sy = src.y + NODE_H;
        const tx = tgt.x + NODE_W / 2 + offset;
        const ty = tgt.y;

        // Determine style
        let stroke, dash, marker, width;
        if (edge.type === 'chain') {
            stroke = '#f97316'; dash = '8 4'; marker = 'url(#arrow-chain)'; width = 2.5;
        } else if (edge.type === 'conditional') {
            stroke = '#9ca3af'; dash = '5 4'; marker = 'url(#arrow-conditional)'; width = 1.5;
        } else {
            stroke = '#6b7280'; dash = 'none'; marker = 'url(#arrow-direct)'; width = 2;
        }

        const g = _svgEl('g', { class: `lg-edge lg-edge-${edge.type}`, 'data-edge': edge.id });

        // Bezier path
        const cy1 = sy + (ty - sy) * 0.45;
        const cy2 = ty - (ty - sy) * 0.45;
        const d = `M ${sx},${sy} C ${sx},${cy1} ${tx},${cy2} ${tx},${ty}`;

        const path = _svgEl('path', {
            d, fill: 'none', stroke, 'stroke-width': width,
            'stroke-dasharray': dash,
            'marker-end': marker,
            opacity: '0.75',
        });
        g.appendChild(path);

        // Invisible wider hit area for hover
        const hitPath = _svgEl('path', {
            d, fill: 'none', stroke: 'transparent', 'stroke-width': 12,
        });
        g.appendChild(hitPath);

        // Label on longer edges
        if (edge.label) {
            const midX = (sx + tx) / 2;
            const midY = (sy + ty) / 2;
            const bg = _svgEl('rect', {
                x: midX - 40, y: midY - 8,
                width: 80, height: 14,
                rx: 3, fill: '#0f1117cc',
            });
            const txt = _svgEl('text', {
                x: midX, y: midY + 1,
                fill: stroke,
                'font-size': '8.5',
                'font-family': 'Inter,sans-serif',
                'text-anchor': 'middle',
                'dominant-baseline': 'middle',
                opacity: '0.9',
            });
            txt.textContent = edge.label.length > 20 ? edge.label.slice(0, 19) + '…' : edge.label;
            g.appendChild(bg);
            g.appendChild(txt);
        }

        g.addEventListener('mouseenter', () => {
            path.setAttribute('opacity', '1');
            path.setAttribute('stroke-width', String(width + 1.5));
        });
        g.addEventListener('mouseleave', () => {
            path.setAttribute('opacity', '0.75');
            path.setAttribute('stroke-width', String(width));
        });

        root.appendChild(g);
    }

    function _renderNodes(root) {
        for (const node of _data.nodes) {
            const pos = _positions[node.id];
            if (!pos) continue;
            _drawNode(root, node, pos);
        }
    }

    function _drawNode(root, node, pos) {
        const { x, y } = pos;
        const color = node.color || '#6b7280';
        const isEnd = node.id === '__end__';
        const isEntry = node.id === _data.entry_point;

        const g = _svgEl('g', {
            class: `lg-node lg-node-${node.category}`,
            'data-id': node.id,
            transform: `translate(${x},${y})`,
            style: 'cursor:pointer',
        });

        // Drop shadow
        const shadow = _svgEl('rect', {
            x: 3, y: 4,
            width: NODE_W, height: NODE_H,
            rx: 10,
            fill: 'rgba(0,0,0,0.45)',
        });
        g.appendChild(shadow);

        // Body
        const body = _svgEl('rect', {
            x: 0, y: 0,
            width: NODE_W, height: NODE_H,
            rx: 10,
            fill: isEnd ? '#1f2937' : '#1a1d2e',
            stroke: color,
            'stroke-width': isEntry ? 2.5 : 1.5,
        });
        g.appendChild(body);

        // Top accent bar
        const accent = _svgEl('rect', {
            x: 0, y: 0,
            width: NODE_W, height: 5,
            rx: 10,
            fill: color,
        });
        // Clip the bottom radius of the accent bar
        const accentCover = _svgEl('rect', {
            x: 0, y: 3,
            width: NODE_W, height: 4,
            fill: color,
        });
        g.appendChild(accent);
        g.appendChild(accentCover);

        // Icon circle
        const iconCircle = _svgEl('circle', {
            cx: 26, cy: NODE_H / 2 + 3,
            r: 14,
            fill: color + '28',
        });
        g.appendChild(iconCircle);

        // Category initial letter as icon fallback
        const letter = _svgEl('text', {
            x: 26, y: NODE_H / 2 + 3,
            fill: color,
            'font-size': '13',
            'font-family': 'Inter,sans-serif',
            'font-weight': '700',
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
        });
        const iconChar = _iconChar(node);
        letter.textContent = iconChar;
        g.appendChild(letter);

        // Label
        const labelEl = _svgEl('text', {
            x: 48, y: NODE_H / 2 - 1,
            fill: '#e4e6f0',
            'font-size': '11.5',
            'font-family': 'Inter,sans-serif',
            'font-weight': '600',
            'dominant-baseline': 'middle',
        });
        const labelText = node.label.length > 16 ? node.label.slice(0, 15) + '…' : node.label;
        labelEl.textContent = labelText;
        g.appendChild(labelEl);

        // Category sub-label
        const catEl = _svgEl('text', {
            x: 48, y: NODE_H / 2 + 13,
            fill: '#6b7099',
            'font-size': '9.5',
            'font-family': 'Inter,sans-serif',
            'dominant-baseline': 'middle',
        });
        catEl.textContent = node.category;
        g.appendChild(catEl);

        // Entry / End badges
        if (isEntry) {
            const badge = _svgEl('rect', {
                x: NODE_W - 44, y: 8,
                width: 36, height: 14,
                rx: 5, fill: '#22c55e33',
                stroke: '#22c55e88',
                'stroke-width': 1,
            });
            const badgeTxt = _svgEl('text', {
                x: NODE_W - 26, y: 15,
                fill: '#22c55e',
                'font-size': '8',
                'font-family': 'Inter,sans-serif',
                'font-weight': '700',
                'text-anchor': 'middle',
                'dominant-baseline': 'middle',
            });
            badgeTxt.textContent = 'ENTRY';
            g.appendChild(badge);
            g.appendChild(badgeTxt);
        }
        if (isEnd) {
            const badge = _svgEl('rect', {
                x: NODE_W - 38, y: 8,
                width: 30, height: 14,
                rx: 5, fill: '#6b728033',
                stroke: '#6b728088',
                'stroke-width': 1,
            });
            const badgeTxt = _svgEl('text', {
                x: NODE_W - 23, y: 15,
                fill: '#9ca3af',
                'font-size': '8',
                'font-family': 'Inter,sans-serif',
                'font-weight': '700',
                'text-anchor': 'middle',
                'dominant-baseline': 'middle',
            });
            badgeTxt.textContent = 'END';
            g.appendChild(badge);
            g.appendChild(badgeTxt);
        }

        // Events
        g.addEventListener('click', (e) => { e.stopPropagation(); _selectNode(node); });
        g.addEventListener('mouseenter', (ev) => {
            body.setAttribute('fill', '#222540');
            body.setAttribute('stroke-width', isEntry ? '3' : '2.5');
            _showTooltip(ev, node);
        });
        g.addEventListener('mouseleave', () => {
            if (!(_selected && _selected.id === node.id)) {
                body.setAttribute('fill', isEnd ? '#1f2937' : '#1a1d2e');
                body.setAttribute('stroke-width', isEntry ? '2.5' : '1.5');
            }
            _hideTooltip();
        });
        g.addEventListener('mousemove', (ev) => _moveTooltip(ev));

        root.appendChild(g);
    }

    // ── Icon character mapping ────────────────────────────────────────────────
    const _iconMap = {
        initialize:                     '▶',
        select_task:                    '☰',
        analyze_task:                   '◉',
        breakdown_task:                 '⊕',
        aggregate_results:              '⊞',
        synthesize_research:            '✦',
        agentic_debate:                 '⇄',
        auto_synthesis:                 '⚡',
        handle_error:                   '⚠',
        human_review:                   '⚙',
        execute_task:                   '»',
        execute_pdf_task:               'P',
        execute_excel_task:             'X',
        execute_ocr_task:               'O',
        execute_web_search_task:        'W',
        execute_code_interpreter_task:  'C',
        execute_data_extraction_task:   'D',
        execute_problem_solver_task:    '?',
        execute_document_task:          '⊡',
        __end__:                        '⬛',
    };
    function _iconChar(node) { return _iconMap[node.id] || node.label[0].toUpperCase(); }

    // ── Selection ────────────────────────────────────────────────────────────
    function _selectNode(node) {
        _selected = node;

        // Reset all node styles
        document.querySelectorAll('.lg-node rect:nth-child(2)').forEach(r => {
            r.setAttribute('stroke-width', '1.5');
        });

        // Highlight selected
        const g = _root().querySelector(`[data-id="${CSS.escape(node.id)}"]`);
        if (g) {
            const body = g.querySelector('rect:nth-child(2)');
            if (body) {
                body.setAttribute('fill', '#222540');
                body.setAttribute('stroke-width', '3');
                body.setAttribute('filter', 'url(#glow)');
            }
        }

        // Highlight edges connected to this node
        document.querySelectorAll('.lg-edge path').forEach(p => {
            p.setAttribute('opacity', '0.2');
        });
        for (const e of _data.edges) {
            if (e.source === node.id || e.target === node.id) {
                const eg = _root().querySelector(`[data-edge="${e.id}"] path`);
                if (eg) eg.setAttribute('opacity', '1');
            }
        }

        _renderDetail(node);
    }

    function resetHighlight() {
        _selected = null;
        document.querySelectorAll('.lg-edge path').forEach(p => p.setAttribute('opacity', '0.75'));
        document.querySelectorAll('.lg-node rect').forEach(r => {
            r.removeAttribute('filter');
            r.setAttribute('stroke-width', '1.5');
        });
        _renderDetailEmpty();
    }

    // ── Detail panel ─────────────────────────────────────────────────────────
    function _renderDetail(node) {
        const panel = document.getElementById('lg-detail-content');
        if (!panel) return;

        const outEdges = _data.edges.filter(e => e.source === node.id);
        const inEdges  = _data.edges.filter(e => e.target === node.id);
        const chains   = (_data.chain_groups || []).filter(cg => cg.nodes.includes(node.id));

        const edgeRow = (e, dir) => {
            const other = dir === 'out' ? e.target : e.source;
            const otherNode = _data.nodes.find(n => n.id === other);
            const typeColor = e.type === 'chain' ? '#f97316' : e.type === 'conditional' ? '#9ca3af' : '#6b7280';
            return `<div class="lg-detail-edge" onclick="LangGraphViewer.highlightNode('${_esc(other)}')">
                <span class="lg-edge-badge" style="background:${typeColor}22;color:${typeColor};border-color:${typeColor}44">${e.type}</span>
                <span class="lg-edge-dir">${dir === 'out' ? '→' : '←'}</span>
                <span class="lg-edge-target">${_esc(otherNode?.label || other)}</span>
                ${e.label ? `<span class="lg-edge-label-txt">${_esc(e.label)}</span>` : ''}
            </div>`;
        };

        panel.innerHTML = `
            <div class="lg-detail-header" style="border-color:${node.color}">
                <span class="lg-detail-icon" style="background:${node.color}22;color:${node.color}">${_iconChar(node)}</span>
                <div>
                    <div class="lg-detail-name">${_esc(node.label)}</div>
                    <div class="lg-detail-category" style="color:${node.color}">${_esc(node.category)}</div>
                </div>
            </div>
            <div class="lg-detail-desc">${_esc(node.description)}</div>

            ${inEdges.length > 0 ? `
            <div class="lg-detail-section-title">Incoming (${inEdges.length})</div>
            <div class="lg-detail-edges">${inEdges.map(e => edgeRow(e, 'in')).join('')}</div>
            ` : ''}

            ${outEdges.length > 0 ? `
            <div class="lg-detail-section-title">Outgoing (${outEdges.length})</div>
            <div class="lg-detail-edges">${outEdges.map(e => edgeRow(e, 'out')).join('')}</div>
            ` : ''}

            ${chains.length > 0 ? `
            <div class="lg-detail-section-title">Chain Groups</div>
            ${chains.map(cg => `
                <div class="lg-detail-chain" style="border-color:${cg.color}55;background:${cg.color}0d">
                    <div style="color:${cg.color};font-weight:600;font-size:11px">${_esc(cg.label)}</div>
                    <div style="color:var(--text-muted);font-size:10.5px;margin-top:3px">${_esc(cg.description)}</div>
                </div>
            `).join('')}
            ` : ''}
        `;
        lucide.createIcons();
    }

    function _renderDetailEmpty() {
        const panel = document.getElementById('lg-detail-content');
        if (panel) panel.innerHTML = `
            <div class="empty-state small">
                <i data-lucide="mouse-pointer-click"></i>
                <p>Click a node to inspect it</p>
            </div>`;
        lucide.createIcons();
    }

    // ── Legend ───────────────────────────────────────────────────────────────
    function _buildLegend() {
        if (!_data) return;

        // Categories
        const catEl = document.getElementById('lg-legend-categories');
        if (catEl && _data.categories) {
            catEl.innerHTML = Object.entries(_data.categories).map(([k, v]) => `
                <div class="lg-legend-row">
                    <span class="lg-legend-dot" style="background:${v.color}"></span>
                    <span class="lg-legend-label">${_esc(v.label)}</span>
                </div>
            `).join('');
        }

        // Edge types
        const edgeEl = document.getElementById('lg-legend-edges');
        if (edgeEl && _data.edge_types) {
            edgeEl.innerHTML = Object.entries(_data.edge_types).map(([k, v]) => `
                <div class="lg-legend-row">
                    <svg width="28" height="10" style="flex-shrink:0">
                        <line x1="0" y1="5" x2="28" y2="5"
                              stroke="${v.color}" stroke-width="${k === 'chain' ? 2.5 : k === 'direct' ? 2 : 1.5}"
                              stroke-dasharray="${k === 'chain' ? '7 3' : k === 'conditional' ? '4 3' : 'none'}"/>
                    </svg>
                    <span class="lg-legend-label">${_esc(v.label)}</span>
                </div>
            `).join('');
        }

        // Chain groups
        const chainEl = document.getElementById('lg-legend-chains');
        if (chainEl && _data.chain_groups) {
            chainEl.innerHTML = _data.chain_groups.map(cg => `
                <div class="lg-legend-row lg-legend-chain-row" onclick="LangGraphViewer.highlightChain('${_esc(cg.id)}')"
                     title="${_esc(cg.description)}">
                    <span class="lg-legend-chain-swatch" style="background:${cg.color}22;border-color:${cg.color}66;color:${cg.color}">⊞</span>
                    <span class="lg-legend-label">${_esc(cg.label)}</span>
                </div>
            `).join('');
        }
    }

    // ── Stats ─────────────────────────────────────────────────────────────────
    function _buildStats() {
        const el = document.getElementById('lg-stats');
        if (!el || !_data) return;
        const nNodes = _data.nodes.length;
        const nEdges = _data.edges.length;
        const nChains = (_data.chain_groups || []).length;
        const nSubagents = _data.nodes.filter(n => n.category === 'subagent').length;
        el.innerHTML = `
            <div class="lg-stat"><span class="lg-stat-val">${nNodes}</span><span class="lg-stat-lbl">Nodes</span></div>
            <div class="lg-stat"><span class="lg-stat-val">${nEdges}</span><span class="lg-stat-lbl">Edges</span></div>
            <div class="lg-stat"><span class="lg-stat-val">${nSubagents}</span><span class="lg-stat-lbl">Sub-agents</span></div>
            <div class="lg-stat"><span class="lg-stat-val">${nChains}</span><span class="lg-stat-lbl">Chains</span></div>
        `;
    }

    // ── Tooltip ───────────────────────────────────────────────────────────────
    function _buildTooltip() {
        if (document.getElementById('lg-tooltip')) return;
        const el = document.createElement('div');
        el.id = 'lg-tooltip';
        el.className = 'lg-tooltip hidden';
        document.body.appendChild(el);
        _tooltip = el;
    }

    function _showTooltip(ev, node) {
        if (!_tooltip) return;
        _tooltip.innerHTML = `
            <div class="lg-tt-title" style="color:${node.color}">${_esc(node.label)}</div>
            <div class="lg-tt-body">${_esc(node.description)}</div>
        `;
        _tooltip.classList.remove('hidden');
        _moveTooltip(ev);
    }

    function _moveTooltip(ev) {
        if (!_tooltip) return;
        const x = ev.clientX + 14;
        const y = ev.clientY - 10;
        const tw = _tooltip.offsetWidth;
        const th = _tooltip.offsetHeight;
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        _tooltip.style.left = (x + tw > vw ? x - tw - 28 : x) + 'px';
        _tooltip.style.top  = (y + th > vh ? y - th + 10 : y) + 'px';
    }

    function _hideTooltip() {
        if (_tooltip) _tooltip.classList.add('hidden');
    }

    // ── Pan / Zoom ────────────────────────────────────────────────────────────
    function _applyTransform() {
        const root = _root();
        root.setAttribute('transform', `translate(${_panX},${_panY}) scale(${_zoom})`);
        const lbl = document.getElementById('lg-zoom-level');
        if (lbl) lbl.textContent = Math.round(_zoom * 100) + '%';
    }

    function _setupSvgInteraction() {
        const container = _container();
        if (!container) return;

        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            const newZoom = Math.max(0.2, Math.min(3, _zoom + delta));
            const rect = container.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            _panX = mx - (mx - _panX) * (newZoom / _zoom);
            _panY = my - (my - _panY) * (newZoom / _zoom);
            _zoom = newZoom;
            _applyTransform();
        }, { passive: false });

        container.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            _dragging = true;
            _lastMouse = { x: e.clientX, y: e.clientY };
            container.style.cursor = 'grabbing';
        });
        window.addEventListener('mousemove', (e) => {
            if (!_dragging) return;
            _panX += e.clientX - _lastMouse.x;
            _panY += e.clientY - _lastMouse.y;
            _lastMouse = { x: e.clientX, y: e.clientY };
            _applyTransform();
        });
        window.addEventListener('mouseup', () => {
            _dragging = false;
            if (container) container.style.cursor = 'default';
        });

        // Click on empty SVG deselects
        const svg = _svg();
        if (svg) {
            svg.addEventListener('click', () => resetHighlight());
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────
    function zoomIn() {
        _zoom = Math.min(3, _zoom + 0.2);
        _applyTransform();
    }

    function zoomOut() {
        _zoom = Math.max(0.2, _zoom - 0.2);
        _applyTransform();
    }

    function fit() {
        if (!_data || _data.nodes.length === 0) return;
        const container = _container();
        if (!container) return;
        const svg = _svg();
        const svgW = parseFloat(svg.getAttribute('width') || 800);
        const svgH = parseFloat(svg.getAttribute('height') || 600);
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        const scale = Math.min(cw / svgW, ch / svgH, 1.5) * 0.92;
        _zoom = scale;
        _panX = (cw - svgW * scale) / 2;
        _panY = (ch - svgH * scale) / 2;
        _applyTransform();
    }

    function toggleChains() {
        _showChains = !_showChains;
        const label = document.getElementById('lg-chain-label');
        if (label) label.textContent = _showChains ? 'Hide Chains' : 'Show Chains';
        if (_data) _render();
    }

    function highlightNode(nodeId) {
        const node = _data && _data.nodes.find(n => n.id === nodeId);
        if (node) _selectNode(node);
    }

    function highlightChain(chainId) {
        const cg = _data && _data.chain_groups && _data.chain_groups.find(c => c.id === chainId);
        if (!cg) return;

        // Dim all, then highlight chain nodes
        document.querySelectorAll('.lg-node').forEach(g => g.setAttribute('opacity', '0.25'));
        document.querySelectorAll('.lg-edge path').forEach(p => p.setAttribute('opacity', '0.1'));

        for (const nid of cg.nodes) {
            const g = _root().querySelector(`[data-id="${CSS.escape(nid)}"]`);
            if (g) g.setAttribute('opacity', '1');
        }
        for (const e of _data.edges) {
            if (cg.nodes.includes(e.source) && cg.nodes.includes(e.target)) {
                const eg = _root().querySelector(`[data-edge="${e.id}"] path`);
                if (eg) { eg.setAttribute('opacity', '1'); eg.setAttribute('filter', 'url(#glow)'); }
            }
        }
    }

    // Called by App navigation when page becomes active
    function onActivate() {
        if (!_data) {
            init();
        } else {
            // Re-fit on re-activation in case container resized
            setTimeout(fit, 80);
        }
    }

    // ── Parse-Code Tab ───────────────────────────────────────────────────────

    let _mode = 'live';   // 'live' | 'parse'

    function switchTab(tab) {
        _mode = tab;
        const parsePanel  = document.getElementById('lg-parse-panel');
        const mainLayout  = document.getElementById('lg-main-layout');
        const tabLive     = document.getElementById('lg-tab-live');
        const tabParse    = document.getElementById('lg-tab-parse');
        const graphActs   = document.getElementById('lg-graph-actions');
        const parseActs   = document.getElementById('lg-parse-actions');
        const subtitle    = document.getElementById('lg-subtitle');

        if (tab === 'parse') {
            parsePanel  && parsePanel.classList.remove('hidden');
            mainLayout  && mainLayout.classList.add('hidden');
            tabLive     && tabLive.classList.remove('active');
            tabParse    && tabParse.classList.add('active');
            graphActs   && graphActs.classList.add('hidden');
            parseActs   && parseActs.classList.remove('hidden');
            if (subtitle) subtitle.textContent = 'Paste or upload any LangGraph Python code to generate a visual diagram';
        } else {
            parsePanel  && parsePanel.classList.add('hidden');
            mainLayout  && mainLayout.classList.remove('hidden');
            tabLive     && tabLive.classList.add('active');
            tabParse    && tabParse.classList.remove('active');
            graphActs   && graphActs.classList.remove('hidden');
            parseActs   && parseActs.classList.add('hidden');
            if (subtitle) subtitle.textContent = 'Live topology of agents and interactions read from source code';
            // If the graph wasn't rendered yet, load live data
            if (!_data) init();
            else setTimeout(fit, 80);
        }
    }

    function _setParseStatus(type, html) {
        const el = document.getElementById('lg-parse-status');
        if (!el) return;
        el.className = `lg-parse-status lg-parse-status--${type}`;
        el.innerHTML = html;
        el.classList.remove('hidden');
    }

    function _clearParseStatus() {
        const el = document.getElementById('lg-parse-status');
        if (el) { el.classList.add('hidden'); el.innerHTML = ''; }
    }

    function _showSourceBadge(text) {
        const badge = document.getElementById('lg-source-badge');
        const span  = document.getElementById('lg-source-badge-text');
        if (badge) badge.classList.remove('hidden');
        if (span)  span.textContent = text;
    }

    function _hideSourceBadge() {
        const badge = document.getElementById('lg-source-badge');
        if (badge) badge.classList.add('hidden');
    }

    async function parseFromInput() {
        const textarea = document.getElementById('lg-code-input');
        if (!textarea) return;
        const code = textarea.value.trim();
        if (!code) {
            _setParseStatus('error', '<i>⚠</i> Please enter some LangGraph Python code first.');
            return;
        }
        _setParseStatus('loading',
            '<span class="lg-parse-spinner"></span> Parsing code and building graph…');
        try {
            const res = await App.api('/api/langgraph/parse', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code }),
            });
            _loadParsed(res);
        } catch (err) {
            let msg = err.message || String(err);
            // Try to extract server detail
            try {
                const detail = JSON.parse(msg);
                if (detail.detail) msg = detail.detail;
            } catch (_) { /* ignore */ }
            _setParseStatus('error',
                `<b>Parse error:</b> ${_esc(msg)}`);
        }
    }

    function loadFile(input) {
        const file = input && input.files && input.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            const ta = document.getElementById('lg-code-input');
            if (ta) ta.value = e.target.result;
            _clearParseStatus();
            // Automatically switch to parse tab and trigger parse
            if (_mode !== 'parse') switchTab('parse');
        };
        reader.onerror = () => {
            _setParseStatus('error', 'Failed to read file.');
        };
        reader.readAsText(file);
    }

    function clearInput() {
        const ta = document.getElementById('lg-code-input');
        if (ta) ta.value = '';
        _clearParseStatus();
    }

    function _loadParsed(data) {
        const summary = data.parse_summary || {};
        _data = data;
        _buildLayout();
        _render();
        _buildLegend();
        _buildStats();

        // Switch view to graph
        const parsePanel  = document.getElementById('lg-parse-panel');
        const mainLayout  = document.getElementById('lg-main-layout');
        if (parsePanel) parsePanel.classList.add('hidden');
        if (mainLayout) mainLayout.classList.remove('hidden');

        setTimeout(fit, 80);

        // Show success status and source badge
        const filename = summary.filename || '(pasted code)';
        const nodeCount = summary.nodes_found || data.nodes.length;
        const edgeCount = summary.edges_found || data.edges.length;
        _setParseStatus('success',
            `✓ Parsed <b>${filename}</b> — ${nodeCount} nodes, ${edgeCount} edges` +
            (summary.entry_point ? ` · entry: <code>${summary.entry_point}</code>` : ''));
        _showSourceBadge(filename);
    }

    // ── Sample code snippets ─────────────────────────────────────────────────

    const _SAMPLES = {
        simple: `from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    result: str

workflow = StateGraph(State)

workflow.add_node("start", start_handler)
workflow.add_node("process", process_handler)
workflow.add_node("format_output", format_handler)
workflow.add_node("save_result", save_handler)

workflow.set_entry_point("start")
workflow.add_edge("start", "process")
workflow.add_edge("process", "format_output")
workflow.add_edge("format_output", "save_result")
workflow.add_edge("save_result", END)
`,

        conditional: `from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    task: str
    task_type: str
    result: str

def route_task(state):
    t = state.get("task_type", "")
    if t == "search":   return "web_search"
    if t == "document": return "document_agent"
    if t == "code":     return "code_interpreter"
    return "default_handler"

workflow = StateGraph(AgentState)
workflow.add_node("initialize",       initialize_fn)
workflow.add_node("classify_task",    classify_fn)
workflow.add_node("web_search",       web_search_fn)
workflow.add_node("document_agent",   doc_agent_fn)
workflow.add_node("code_interpreter", code_agent_fn)
workflow.add_node("default_handler",  default_fn)
workflow.add_node("aggregate",        aggregate_fn)

workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "classify_task")
workflow.add_conditional_edges(
    "classify_task",
    route_task,
    {
        "web_search":       "web_search",
        "document_agent":   "document_agent",
        "code_interpreter": "code_interpreter",
        "default_handler":  "default_handler",
    }
)
workflow.add_edge("web_search",       "aggregate")
workflow.add_edge("document_agent",   "aggregate")
workflow.add_edge("code_interpreter", "aggregate")
workflow.add_edge("default_handler",  "aggregate")
workflow.add_edge("aggregate",        END)
`,

        multi: `from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class MultiAgentState(TypedDict):
    query: str
    subtasks: List[str]
    results: List[str]
    synthesis: str
    errors: List[str]

def route_after_planning(state):
    subtasks = state.get("subtasks", [])
    if len(subtasks) == 0: return "handle_error"
    return "dispatch_agents"

def route_after_synthesis(state):
    if state.get("errors"): return "review_errors"
    return "finalize"

graph = StateGraph(MultiAgentState)
graph.add_node("intake",          intake_fn)
graph.add_node("plan_subtasks",   planner_fn)
graph.add_node("dispatch_agents", dispatcher_fn)
graph.add_node("research_agent",  research_fn)
graph.add_node("analysis_agent",  analysis_fn)
graph.add_node("writing_agent",   writing_fn)
graph.add_node("synthesize",      synthesize_fn)
graph.add_node("review_errors",   review_fn)
graph.add_node("finalize",        finalize_fn)
graph.add_node("handle_error",    error_fn)

graph.set_entry_point("intake")
graph.add_edge("intake", "plan_subtasks")
graph.add_conditional_edges(
    "plan_subtasks",
    route_after_planning,
    {
        "dispatch_agents": "dispatch_agents",
        "handle_error":    "handle_error",
    }
)
graph.add_edge("dispatch_agents", "research_agent")
graph.add_edge("dispatch_agents", "analysis_agent")
graph.add_edge("dispatch_agents", "writing_agent")
graph.add_edge("research_agent",  "synthesize")
graph.add_edge("analysis_agent",  "synthesize")
graph.add_edge("writing_agent",   "synthesize")
graph.add_conditional_edges(
    "synthesize",
    route_after_synthesis,
    {
        "review_errors": "review_errors",
        "finalize":      "finalize",
    }
)
graph.add_edge("review_errors", "finalize")
graph.add_edge("finalize",      END)
graph.add_edge("handle_error",  END)
`,
    };

    async function loadSample(name) {
        const ta = document.getElementById('lg-code-input');
        if (!ta) return;
        _clearParseStatus();

        if (name === 'live') {
            // Fetch the live workflow.py source from the server
            _setParseStatus('loading', '<span class="lg-parse-spinner"></span> Loading workflow.py…');
            try {
                const res = await fetch('/api/langgraph/workflow-source');
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const { source } = await res.json();
                ta.value = source;
                _clearParseStatus();
            } catch (e) {
                // Fall back to the live workflow API data display
                _setParseStatus('error',
                    'Could not fetch workflow.py source. ' +
                    'Use the "Live Workflow" tab to view the built-in graph.');
            }
            return;
        }

        const code = _SAMPLES[name];
        if (code) {
            ta.value = code;
        }
    }

    return {
        init,
        fit,
        zoomIn,
        zoomOut,
        toggleChains,
        resetHighlight,
        highlightNode,
        highlightChain,
        onActivate,
        switchTab,
        parseFromInput,
        loadFile,
        clearInput,
        loadSample,
    };
})();
