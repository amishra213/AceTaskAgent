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
            _buildInfoPanel();
        } catch (e) {
            _root().innerHTML = `
                <text x="50%" y="50%" fill="#ef4444" text-anchor="middle"
                      font-family="Inter,sans-serif" font-size="14">
                    Failed to load workflow: ${_esc(e.message)}
                </text>`;
            _buildInfoPanel(); // show empty state
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

        // For decision → target edges, start from the diamond tip (bottom vertex)
        const srcNode = _data.nodes.find(n => n.id === edge.source);
        const tgtNode = _data.nodes.find(n => n.id === edge.target);
        const fromDiamond = srcNode && srcNode.category === 'condition';
        const toDiamond   = tgtNode && tgtNode.category === 'condition';

        // Determine style
        let stroke, dash, marker, width;
        if (edge.type === 'chain') {
            stroke = '#f97316'; dash = '8 4'; marker = 'url(#arrow-chain)'; width = 2.5;
        } else if (edge.type === 'conditional') {
            stroke = '#f59e0b'; dash = '6 3'; marker = 'url(#arrow-conditional)'; width = 1.8;
        } else {
            stroke = '#6b7280'; dash = 'none'; marker = 'url(#arrow-direct)'; width = 2;
        }

        const g = _svgEl('g', { class: `lg-edge lg-edge-${edge.type}`, 'data-edge': edge.id });

        // Adjust start/end points for diamond nodes (connect to the tip)
        let ax = sx, ay = sy, bx = tx, by = ty;
        if (fromDiamond) {
            // bottom tip of diamond
            ax = src.x + NODE_W / 2 + offset;
            ay = src.y + NODE_H / 2 + (NODE_H / 2 - 2);
        }
        if (toDiamond) {
            // top tip of diamond
            bx = tgt.x + NODE_W / 2 + offset;
            by = tgt.y + 2;
        }

        // Bezier control points — wider spread for conditional fan-out
        const dy = by - ay;
        const tension = fromDiamond ? 0.6 : 0.45;
        const cy1 = ay + dy * tension;
        const cy2 = by - dy * tension;
        const d = `M ${ax},${ay} C ${ax},${cy1} ${bx},${cy2} ${bx},${by}`;

        const path = _svgEl('path', {
            d, fill: 'none', stroke, 'stroke-width': width,
            'stroke-dasharray': dash,
            'marker-end': marker,
            opacity: edge.type === 'conditional' ? '0.85' : '0.75',
        });
        g.appendChild(path);

        // Invisible wider hit area for hover
        const hitPath = _svgEl('path', {
            d, fill: 'none', stroke: 'transparent', 'stroke-width': 12,
        });
        g.appendChild(hitPath);

        // Label on edge — show both branch key and condition expression if present
        const labelText = edge.condition
            ? `${edge.label} [${edge.condition}]`
            : edge.label || '';

        if (labelText) {
            const midX = (ax + bx) / 2 + (bx - ax) * 0.05;
            const midY = (ay + by) / 2;

            // Estimate pill width
            const textLen = Math.min(labelText.length, 28);
            const pillW = Math.max(48, textLen * 5.8 + 10);
            const pillH = 14;

            const bg = _svgEl('rect', {
                x: midX - pillW / 2, y: midY - pillH / 2,
                width: pillW, height: pillH,
                rx: 4,
                fill: edge.type === 'conditional' ? '#f59e0b22' : '#0f1117cc',
                stroke: edge.type === 'conditional' ? '#f59e0b55' : 'none',
                'stroke-width': 1,
            });
            const displayLabel = labelText.length > 28 ? labelText.slice(0, 27) + '…' : labelText;
            const txt = _svgEl('text', {
                x: midX, y: midY + 1,
                fill: edge.type === 'conditional' ? '#fbbf24' : stroke,
                'font-size': '8',
                'font-family': 'Inter,sans-serif',
                'text-anchor': 'middle',
                'dominant-baseline': 'middle',
                opacity: '0.95',
            });
            txt.textContent = displayLabel;
            g.appendChild(bg);
            g.appendChild(txt);
        }

        g.addEventListener('mouseenter', () => {
            path.setAttribute('opacity', '1');
            path.setAttribute('stroke-width', String(width + 1.5));
        });
        g.addEventListener('mouseleave', () => {
            path.setAttribute('opacity', edge.type === 'conditional' ? '0.85' : '0.75');
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
        const isDecision = node.category === 'condition';

        const g = _svgEl('g', {
            class: `lg-node lg-node-${node.category}`,
            'data-id': node.id,
            transform: `translate(${x},${y})`,
            style: 'cursor:pointer',
        });

        if (isDecision) {
            // ── Diamond shape for condition/router nodes ──────────────────
            const cx = NODE_W / 2;
            const cy = NODE_H / 2;
            const hw = NODE_W / 2 - 4;   // half-width of diamond
            const hh = NODE_H / 2 - 2;   // half-height of diamond
            const pts = `${cx},${cy - hh} ${cx + hw},${cy} ${cx},${cy + hh} ${cx - hw},${cy}`;

            // Shadow
            const shadow = _svgEl('polygon', {
                points: `${cx + 3},${cy - hh + 4} ${cx + hw + 3},${cy + 4} ${cx + 3},${cy + hh + 4} ${cx - hw + 3},${cy + 4}`,
                fill: 'rgba(0,0,0,0.45)',
            });
            g.appendChild(shadow);

            // Body diamond
            const body = _svgEl('polygon', {
                points: pts,
                fill: '#1a1d2e',
                stroke: color,
                'stroke-width': '2',
            });
            g.appendChild(body);

            // Inner highlight ring
            const hw2 = hw - 5; const hh2 = hh - 5;
            const pts2 = `${cx},${cy - hh2} ${cx + hw2},${cy} ${cx},${cy + hh2} ${cx - hw2},${cy}`;
            const inner = _svgEl('polygon', {
                points: pts2,
                fill: 'none',
                stroke: color + '44',
                'stroke-width': '1',
            });
            g.appendChild(inner);

            // Icon ⋄ symbol
            const iconTxt = _svgEl('text', {
                x: cx, y: cy - 8,
                fill: color,
                'font-size': '13',
                'font-family': 'Inter,sans-serif',
                'font-weight': '700',
                'text-anchor': 'middle',
                'dominant-baseline': 'middle',
            });
            iconTxt.textContent = '⋄';
            g.appendChild(iconTxt);

            // Label
            const labelEl = _svgEl('text', {
                x: cx, y: cy + 7,
                fill: '#e4e6f0',
                'font-size': '10',
                'font-family': 'Inter,sans-serif',
                'font-weight': '600',
                'text-anchor': 'middle',
                'dominant-baseline': 'middle',
            });
            const raw = node.label || '';
            labelEl.textContent = raw.length > 14 ? raw.slice(0, 13) + '…' : raw;
            g.appendChild(labelEl);

            // Events
            g.addEventListener('click', (e) => { e.stopPropagation(); _selectNode(node); });
            g.addEventListener('mouseenter', (ev) => {
                body.setAttribute('fill', '#222540');
                body.setAttribute('stroke-width', '3');
                _showTooltip(ev, node);
            });
            g.addEventListener('mouseleave', () => {
                if (!(_selected && _selected.id === node.id)) {
                    body.setAttribute('fill', '#1a1d2e');
                    body.setAttribute('stroke-width', '2');
                }
                _hideTooltip();
            });
            g.addEventListener('mousemove', (ev) => _moveTooltip(ev));
            root.appendChild(g);
            return;
        }

        // ── Standard rectangle node ───────────────────────────────────────
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

        const letter = _svgEl('text', {
            x: 26, y: NODE_H / 2 + 3,
            fill: color,
            'font-size': '13',
            'font-family': 'Inter,sans-serif',
            'font-weight': '700',
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
        });
        letter.textContent = _iconChar(node);
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
        document.querySelectorAll('.lg-node rect:nth-child(2)').forEach(r => r.setAttribute('stroke-width', '1.5'));
        document.querySelectorAll('.lg-node-condition polygon:nth-child(2)').forEach(p => p.setAttribute('stroke-width', '2'));

        // Highlight selected node
        const g = _root().querySelector(`[data-id="${CSS.escape(node.id)}"]`);
        if (g) {
            if (node.category === 'condition') {
                const body = g.querySelector('polygon:nth-child(2)');
                if (body) {
                    body.setAttribute('fill', '#222540');
                    body.setAttribute('stroke-width', '3');
                    body.setAttribute('filter', 'url(#glow)');
                }
            } else {
                const body = g.querySelector('rect:nth-child(2)');
                if (body) {
                    body.setAttribute('fill', '#222540');
                    body.setAttribute('stroke-width', '3');
                    body.setAttribute('filter', 'url(#glow)');
                }
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
        document.querySelectorAll('.lg-node-condition polygon').forEach(p => {
            p.removeAttribute('filter');
            p.setAttribute('stroke-width', '2');
        });
        _renderDetailEmpty();
    }

    // ── Right panel: Agent / Node info list ──────────────────────────────────
    function _buildInfoPanel() {
        const container = document.getElementById('lg-info-content');
        const countEl   = document.getElementById('lg-info-count');
        if (!container) return;

        if (!_data || !_data.nodes || _data.nodes.length === 0) {
            container.innerHTML = `<div class="empty-state small">
                <i data-lucide="loader"></i><p>Loading workflow…</p></div>`;
            if (countEl) countEl.classList.add('hidden');
            lucide.createIcons();
            return;
        }

        // All real nodes (no synthetic __decision__ nodes — show them only when
        // a real node card is expanded)
        const realNodes = _data.nodes.filter(n => !n.id.startsWith('__'));

        if (countEl) {
            countEl.textContent = realNodes.length;
            countEl.classList.remove('hidden');
        }

        container.innerHTML = realNodes.map(node => {
            const isDecision = node.category === 'condition';
            const color      = node.color || '#6b7280';
            const inEdges    = _data.edges.filter(e => e.target === node.id);
            const outEdges   = _data.edges.filter(e => e.source === node.id);
            const chains     = (_data.chain_groups || []).filter(cg => cg.nodes.includes(node.id));

            // Router node for this node (if any)
            const routerNode = _data.nodes.find(n => n.id === `__decision__${node.id}`);

            // Quick props list
            const props = [];
            if (node.handler)     props.push(['handler',     node.handler,      true]);
            if (node.chain)       props.push(['chain',       node.chain,        true]);
            if (node.description && node.description !== '(no description)')
                                  props.push(['description', node.description,  false]);
            if (isDecision && node.router_name)
                                  props.push(['router fn',   node.router_name,  true]);
            if (node.lambda_body) props.push(['condition',   node.lambda_body,  true]);
            if (chains.length)    props.push(['group',       chains.map(c => c.label).join(', '), false]);

            const propsHtml = props.map(([k, v, mono]) =>
                `<div class="lg-ic-prop">
                    <span class="lg-ic-key">${k}</span>
                    ${mono ? `<code class="lg-ic-val-code">${_esc(v)}</code>`
                           : `<span class="lg-ic-val">${_esc(v)}</span>`}
                </div>`
            ).join('');

            // Router branches (if this node has a decision router attached)
            let routingHtml = '';
            if (routerNode && routerNode.router_branches && routerNode.router_branches.length) {
                routingHtml = `<div class="lg-ic-section-title">Routing branches</div>
                <div class="lg-ic-branches">${routerNode.router_branches.map((b, i) =>
                    `<div class="lg-ic-branch">
                        <span class="lg-ic-branch-num">${i + 1}</span>
                        <span class="lg-ic-branch-cond">${_esc(b.condition)}</span>
                        <span class="lg-ic-branch-arrow">→</span>
                        <span class="lg-ic-branch-ret">${_esc(b.returns)}</span>
                    </div>`
                ).join('')}</div>`;
            }

            // Edges section (collapsed until card is active)
            const edgesHtml = (inEdges.length + outEdges.length) > 0 ? `
                <div class="lg-ic-edges">
                    ${inEdges.length ? `<div class="lg-ic-section-title">← Incoming (${inEdges.length})</div>
                    ${inEdges.map(e => {
                        const src = _data.nodes.find(n => n.id === e.source);
                        const typeColor = e.type === 'chain' ? '#f97316' : e.type === 'conditional' ? '#f59e0b' : '#6b7280';
                        return `<div class="lg-ic-edge" onclick="LangGraphViewer.highlightNode('${_esc(e.source)}')">
                            <span class="lg-ic-edge-badge" style="color:${typeColor}">${e.type}</span>
                            <span class="lg-ic-edge-node">${_esc(src?.label || e.source)}</span>
                            ${e.condition ? `<span class="lg-ic-edge-cond">${_esc(e.condition)}</span>` : ''}
                        </div>`;
                    }).join('')}` : ''}
                    ${outEdges.length ? `<div class="lg-ic-section-title">→ Outgoing (${outEdges.length})</div>
                    ${outEdges.map(e => {
                        const tgt = _data.nodes.find(n => n.id === e.target);
                        const typeColor = e.type === 'chain' ? '#f97316' : e.type === 'conditional' ? '#f59e0b' : '#6b7280';
                        return `<div class="lg-ic-edge" onclick="LangGraphViewer.highlightNode('${_esc(e.target)}')">
                            <span class="lg-ic-edge-badge" style="color:${typeColor}">${e.type}</span>
                            <span class="lg-ic-edge-node">${_esc(tgt?.label || e.target)}</span>
                            ${e.condition ? `<span class="lg-ic-edge-cond">${_esc(e.condition)}</span>` : ''}
                        </div>`;
                    }).join('')}` : ''}
                </div>` : '';

            return `<div class="lg-ic-card" id="lg-ic-${_esc(node.id)}" style="--node-color:${color}"
                         onclick="LangGraphViewer.highlightNode('${_esc(node.id)}')">
                <div class="lg-ic-header">
                    <span class="lg-ic-icon" style="background:${color}22;color:${color}">
                        ${isDecision ? '⋄' : _iconChar(node)}
                    </span>
                    <div class="lg-ic-title-block">
                        <span class="lg-ic-name">${_esc(node.label || node.id)}</span>
                        <span class="lg-ic-badge" style="background:${color}20;color:${color}">${_esc(node.category)}</span>
                    </div>
                    <div class="lg-ic-edge-pips">
                        <span class="lg-ic-pip lg-ic-pip-in" title="${inEdges.length} incoming">${inEdges.length}</span>
                        <span class="lg-ic-pip lg-ic-pip-out" title="${outEdges.length} outgoing">${outEdges.length}</span>
                    </div>
                </div>
                <div class="lg-ic-body">
                    ${propsHtml}
                    ${routingHtml}
                    ${edgesHtml}
                </div>
            </div>`;
        }).join('');

        lucide.createIcons();
    }

    function _highlightInfoCard(nodeId) {
        // Remove previous active state
        document.querySelectorAll('.lg-ic-card.active').forEach(c => c.classList.remove('active'));
        if (!nodeId) return;
        const card = document.getElementById(`lg-ic-${nodeId}`);
        if (!card) return;
        card.classList.add('active');
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Legacy detail panel functions — now delegate to card highlight
    function _renderDetail(node) {
        _highlightInfoCard(node.id);
    }

    function _renderDetailEmpty() {
        _highlightInfoCard(null);
    }

    // ── Legend ───────────────────────────────────────────────────────────────
    function _buildLegend() {
        if (!_data) return;

        // Categories
        const catEl = document.getElementById('lg-legend-categories');
        if (catEl && _data.categories) {
            catEl.innerHTML = Object.entries(_data.categories).map(([k, v]) => `
                <div class="lg-legend-row">
                    ${k === 'condition'
                        ? `<svg width="14" height="14" style="flex-shrink:0"><polygon points="7,1 13,7 7,13 1,7" fill="none" stroke="${v.color}" stroke-width="1.5"/></svg>`
                        : `<span class="lg-legend-dot" style="background:${v.color}"></span>`
                    }
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
                              stroke="${v.color}" stroke-width="${k === 'chain' ? 2.5 : k === 'direct' ? 2 : 1.8}"
                              stroke-dasharray="${k === 'chain' ? '7 3' : k === 'conditional' ? '5 2' : 'none'}"/>
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
        const nNodes     = _data.nodes.filter(n => n.category !== 'condition').length;
        const nEdges     = _data.edges.length;
        const nChains    = (_data.chain_groups || []).length;
        const nSubagents = _data.nodes.filter(n => n.category === 'subagent').length;
        const nDecisions = _data.nodes.filter(n => n.category === 'condition').length;
        el.innerHTML = `
            <div class="lg-stat"><span class="lg-stat-val">${nNodes}</span><span class="lg-stat-lbl">Nodes</span></div>
            <div class="lg-stat"><span class="lg-stat-val">${nEdges}</span><span class="lg-stat-lbl">Edges</span></div>
            <div class="lg-stat"><span class="lg-stat-val">${nSubagents}</span><span class="lg-stat-lbl">Sub-agents</span></div>
            <div class="lg-stat"><span class="lg-stat-val">${nDecisions}</span><span class="lg-stat-lbl">Decisions</span></div>
            ${nChains > 0 ? `<div class="lg-stat"><span class="lg-stat-val">${nChains}</span><span class="lg-stat-lbl">Chains</span></div>` : ''}
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
        // Update all zoom percentage labels (main toolbar + diagram toolbar)
        const pct = Math.round(_zoom * 100) + '%';
        document.querySelectorAll('#lg-zoom-level, .lg-zoom-display').forEach(el => {
            el.textContent = pct;
        });
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
            setTimeout(fit, 80);
        }
    }

    // ── Tab mode ─────────────────────────────────────────────────────────────

    let _mode = 'workflow';   // 'workflow' | 'summary'
    let _isCustomCode = false; // true when user has parsed custom code (not live data)

    function switchTab(tab) {
        _mode = tab;
        const summaryPanel = document.getElementById('lg-summary-panel');
        const mainLayout   = document.getElementById('lg-main-layout');
        const tabWorkflow  = document.getElementById('lg-tab-workflow');
        const tabSummary   = document.getElementById('lg-tab-summary');
        const graphActs    = document.getElementById('lg-graph-actions');
        const subtitle     = document.getElementById('lg-subtitle');

        // ── Reset all panels / tabs ───────────────────────────────────────
        summaryPanel && summaryPanel.classList.add('hidden');
        mainLayout   && mainLayout.classList.remove('hidden');

        tabWorkflow && tabWorkflow.classList.remove('active');
        tabSummary  && tabSummary.classList.remove('active');

        graphActs   && graphActs.classList.remove('hidden');

        if (tab === 'summary') {
            // ── Code Summary tab ──────────────────────────────────────────
            mainLayout   && mainLayout.classList.add('hidden');
            summaryPanel && summaryPanel.classList.remove('hidden');
            tabSummary   && tabSummary.classList.add('active');
            graphActs    && graphActs.classList.add('hidden');
            if (subtitle) subtitle.textContent = 'Code summary — agents, sub-agents, edges and routing';

        } else {
            // ── Workflow tab (default) ────────────────────────────────────
            tabWorkflow && tabWorkflow.classList.add('active');
            if (subtitle) {
                subtitle.textContent = _isCustomCode
                    ? 'Showing parsed custom code — click "Live" in the toolbar to reload the live workflow'
                    : 'Live topology of agents and interactions read from source code';
            }
            if (!_data) init();
            else setTimeout(fit, 80);
        }
    }

    // ── Code drawer ──────────────────────────────────────────────────────────

    let _drawerOpen = false;

    function toggleCodeDrawer() {
        const drawer     = document.getElementById('lg-code-drawer');
        const toggleBtn  = document.getElementById('lg-btn-code-toggle');
        if (!drawer) return;
        _drawerOpen = !_drawerOpen;
        drawer.classList.toggle('hidden', !_drawerOpen);
        if (toggleBtn) toggleBtn.classList.toggle('active', _drawerOpen);
    }

    async function reloadLive() {
        _isCustomCode = false;
        _clearParseStatus();
        // Hide Summary tab and source badge
        const tabSummary = document.getElementById('lg-tab-summary');
        const reloadBtn  = document.getElementById('lg-btn-reload-live');
        if (tabSummary) tabSummary.classList.add('hidden');
        if (reloadBtn)  reloadBtn.classList.add('hidden');
        _hideSourceBadge();
        // Close the code drawer if open
        const drawer = document.getElementById('lg-code-drawer');
        if (drawer && !drawer.classList.contains('hidden')) toggleCodeDrawer();
        // Re-run the live data fetch
        switchTab('workflow');
        await init();
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
            try {
                const detail = JSON.parse(msg);
                if (detail.detail) msg = detail.detail;
            } catch (_) { /* ignore */ }
            _setParseStatus('error', `<b>Parse error:</b> ${_esc(msg)}`);
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
            // Open the drawer so the user can review + click Parse
            if (!_drawerOpen) toggleCodeDrawer();
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
        _isCustomCode = true;
        _buildLayout();
        _render();
        _buildLegend();
        _buildStats();

        // Render the Code Summary panel
        _renderSummaryPanel(data);

        // Rebuild the right-panel agent info list
        _buildInfoPanel();

        // Show Summary tab button and the "Live" reset button
        const tabSummary = document.getElementById('lg-tab-summary');
        const reloadBtn  = document.getElementById('lg-btn-reload-live');
        if (tabSummary) tabSummary.classList.remove('hidden');
        if (reloadBtn)  reloadBtn.classList.remove('hidden');

        // Close the drawer (graph is now showing)
        if (_drawerOpen) toggleCodeDrawer();

        // Switch to Code Summary to show parsed results
        switchTab('summary');

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
    if t == "search":
        return "web_search"
    elif t == "document":
        return "document_agent"
    elif t == "code":
        return "code_interpreter"
    else:
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
    if len(subtasks) == 0:
        return "handle_error"
    return "dispatch_agents"

def route_after_synthesis(state):
    if state.get("errors"):
        return "review_errors"
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

    // ── Agent Info Panel ─────────────────────────────────────────────────────
    // ── Code Summary Panel ───────────────────────────────────────────────────
    function _renderSummaryPanel(data) {
        const summary   = data.parse_summary || {};
        const nodes     = data.nodes  || [];
        const edges     = data.edges  || [];

        // ─ Stats bar ─────────────────────────────────────────────────────
        const statsEl   = document.getElementById('lg-summary-stats');
        const fileEl    = document.getElementById('lg-summary-filename');
        const filename  = summary.filename || '(pasted code)';
        if (fileEl) fileEl.textContent = filename;

        const agentNodes = nodes.filter(n => !n.id.startsWith('__'));
        const condNodes  = nodes.filter(n => n.category === 'condition');
        const directEdges      = edges.filter(e => e.type === 'direct');
        const conditionalEdges = edges.filter(e => e.type === 'conditional');

        if (statsEl) {
            statsEl.innerHTML = [
                _statChip('cpu',       agentNodes.length,      'Agents / Nodes'),
                _statChip('git-branch',condNodes.length,       'Condition Routers'),
                _statChip('arrow-right',directEdges.length,    'Direct Edges'),
                _statChip('git-merge', conditionalEdges.length,'Conditional Edges'),
                summary.entry_point ? _statChip('play', summary.entry_point, 'Entry Point') : '',
                summary.finish_point ? _statChip('square', summary.finish_point, 'Exit Point') : '',
            ].join('');
        }

        // ─ Nodes count badge ─────────────────────────────────────────────
        const nodeCount = document.getElementById('lg-summary-node-count');
        if (nodeCount) nodeCount.textContent = agentNodes.length;

        // ─ Edge count badge ───────────────────────────────────────────────
        const edgeCount = document.getElementById('lg-summary-edge-count');
        if (edgeCount) edgeCount.textContent = edges.length;

        // ─ Agent / node cards ─────────────────────────────────────────────
        const nodesEl = document.getElementById('lg-summary-nodes');
        if (nodesEl) {
            if (agentNodes.length === 0) {
                nodesEl.innerHTML = '<p class="lg-summary-empty">No agents found.</p>';
            } else {
                nodesEl.innerHTML = agentNodes.map(n => {
                    const color    = n.color || '#6b7280';
                    const category = n.category || 'node';
                    const incoming = edges.filter(e => e.target === n.id || e.to === n.id).length;
                    const outgoing = edges.filter(e => e.source === n.id || e.from === n.id).length;
                    // Collect condition branches if this is a router source
                    const routedBy = nodes.find(c => c.category === 'condition' &&
                        c.id === `__decision__${n.id}`);
                    const branches = routedBy ? (routedBy.router_branches || []) : [];

                    return `
                    <div class="lg-snode-card" style="border-left-color:${color}">
                        <div class="lg-snode-header">
                            <span class="lg-snode-name">${_esc(n.id)}</span>
                            <span class="lg-snode-badge" style="background:${color}20;color:${color}">${_esc(category)}</span>
                        </div>
                        <div class="lg-snode-props">
                            ${n.handler ? `<div class="lg-snode-prop"><span class="lg-prop-key">handler</span><code class="lg-prop-val">${_esc(n.handler)}</code></div>` : ''}
                            ${n.description && n.description !== '(no description)' ? `<div class="lg-snode-prop"><span class="lg-prop-key">description</span><span class="lg-prop-val">${_esc(n.description)}</span></div>` : ''}
                            ${n.label && n.label !== n.id ? `<div class="lg-snode-prop"><span class="lg-prop-key">label</span><span class="lg-prop-val">${_esc(n.label)}</span></div>` : ''}
                            ${n.chain ? `<div class="lg-snode-prop"><span class="lg-prop-key">chain</span><code class="lg-prop-val">${_esc(n.chain)}</code></div>` : ''}
                            ${n.router_name ? `<div class="lg-snode-prop"><span class="lg-prop-key">router</span><code class="lg-prop-val">${_esc(n.router_name)}</code></div>` : ''}
                            ${n.lambda_body ? `<div class="lg-snode-prop"><span class="lg-prop-key">condition</span><code class="lg-prop-val lg-prop-code">${_esc(n.lambda_body)}</code></div>` : ''}
                            <div class="lg-snode-prop">
                                <span class="lg-prop-key">edges</span>
                                <span class="lg-prop-val">
                                    <span class="lg-edge-pip lg-pip-in">${incoming} in</span>
                                    <span class="lg-edge-pip lg-pip-out">${outgoing} out</span>
                                </span>
                            </div>
                            ${branches.length ? `<div class="lg-snode-prop"><span class="lg-prop-key">branches</span><span class="lg-prop-val">${branches.map(b => `<span class="lg-branch-tag">${_esc(b)}</span>`).join(' ')}</span></div>` : ''}
                        </div>
                    </div>`;
                }).join('');
            }
        }

        // ─ Edge / routing table ───────────────────────────────────────────
        const edgesEl = document.getElementById('lg-summary-edges');
        if (edgesEl) {
            if (edges.length === 0) {
                edgesEl.innerHTML = '<p class="lg-summary-empty">No edges found.</p>';
            } else {
                // Group: direct first, then conditional
                const sorted = [...edges].sort((a, b) => {
                    const ta = a.type || 'direct', tb = b.type || 'direct';
                    if (ta === tb) return 0;
                    return ta === 'direct' ? -1 : 1;
                });
                let html = '<table class="lg-edge-table"><thead><tr>'
                    + '<th>From</th><th></th><th>To</th><th>Type</th><th>Condition</th>'
                    + '</tr></thead><tbody>';
                for (const e of sorted) {
                    const src  = e.source || e.from || '?';
                    const tgt  = e.target || e.to   || '?';
                    const type = e.type   || 'direct';
                    const cond = e.condition || e.label || '';
                    const arrow = type === 'conditional' ? '⟶' : '→';
                    const cls   = type === 'conditional' ? 'lg-etype-cond' : 'lg-etype-dir';
                    html += `<tr>
                        <td><code class="lg-edge-node">${_esc(src)}</code></td>
                        <td class="lg-edge-arrow">${arrow}</td>
                        <td><code class="lg-edge-node">${_esc(tgt)}</code></td>
                        <td><span class="lg-edge-type ${cls}">${_esc(type)}</span></td>
                        <td class="lg-edge-cond">${cond ? _esc(cond) : ''}</td>
                    </tr>`;
                }
                html += '</tbody></table>';
                edgesEl.innerHTML = html;
            }
        }

        // Re-initialize lucide icons
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
    }

    function _statChip(icon, value, label) {
        return `<div class="lg-stat-chip">
            <i data-lucide="${icon}"></i>
            <span class="lg-stat-chip-val">${_esc(String(value))}</span>
            <span class="lg-stat-chip-label">${label}</span>
        </div>`;
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
        toggleCodeDrawer,
        reloadLive,
        parseFromInput,
        loadFile,
        clearInput,
        loadSample,
    };
})();
