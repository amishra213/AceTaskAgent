/**
 * Workflow Designer - Visual node-based workflow editor.
 * Uses HTML5 Canvas for rendering nodes and edges.
 */

const Designer = {
    canvas: null,
    ctx: null,
    nodes: [],
    edges: [],
    agents: [],
    currentWorkflow: null,
    
    // Canvas state
    zoom: 1,
    panX: 0,
    panY: 0,
    dragging: null,
    connecting: null,
    selected: null,
    isPanning: false,
    lastMouse: { x: 0, y: 0 },

    // Node dimensions
    NODE_W: 160,
    NODE_H: 64,
    PORT_R: 6,

    // Colors per agent type
    agentColors: {},

    async init() {
        this.canvas = document.getElementById('workflow-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();
        this.setupEvents();
        await this.loadAgents();
        this.render();
    },

    resizeCanvas() {
        const wrapper = this.canvas.parentElement;
        this.canvas.width = wrapper.clientWidth;
        this.canvas.height = wrapper.clientHeight;
    },

    // ---- Load Agents Palette ----
    async loadAgents() {
        try {
            const res = await App.api('/api/agents');
            this.agents = res.agents || [];
            this.renderPalette();
        } catch (e) {
            console.error('Failed to load agents:', e);
            // Fallback agent list
            this.agents = [
                { type: 'web_search', label: 'Web Search', color: '#3b82f6', icon: 'search' },
                { type: 'pdf', label: 'PDF Agent', color: '#ef4444', icon: 'file-text' },
                { type: 'excel', label: 'Excel Agent', color: '#22c55e', icon: 'table' },
                { type: 'ocr', label: 'OCR Agent', color: '#f59e0b', icon: 'image' },
                { type: 'code_interpreter', label: 'Code Interpreter', color: '#8b5cf6', icon: 'code' },
                { type: 'data_extraction', label: 'Data Extraction', color: '#06b6d4', icon: 'database' },
                { type: 'problem_solver', label: 'Problem Solver', color: '#ec4899', icon: 'lightbulb' },
                { type: 'document', label: 'Document', color: '#14b8a6', icon: 'file-output' },
            ];
            this.renderPalette();
        }

        this.agents.forEach(a => { this.agentColors[a.type] = a.color; });
    },

    renderPalette() {
        const list = document.getElementById('agent-list');
        list.innerHTML = this.agents.map(a => `
            <div class="palette-item" draggable="true" data-type="${a.type}"
                 ondragstart="Designer.onDragStart(event, '${a.type}')">
                <div class="palette-item-icon" style="background:${a.color}22; color:${a.color}">
                    <i data-lucide="${a.icon || 'box'}"></i>
                </div>
                <span class="palette-item-label">${a.label}</span>
            </div>
        `).join('');
        lucide.createIcons();
    },

    // ---- Drag & Drop from Palette ----
    onDragStart(e, type) {
        e.dataTransfer.setData('agent-type', type);
        e.dataTransfer.effectAllowed = 'copy';
    },

    setupEvents() {
        const c = this.canvas;
        window.addEventListener('resize', () => { this.resizeCanvas(); this.render(); });

        // Drop on canvas
        c.addEventListener('dragover', (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
        c.addEventListener('drop', (e) => {
            e.preventDefault();
            const type = e.dataTransfer.getData('agent-type');
            if (!type) return;
            const rect = c.getBoundingClientRect();
            const x = (e.clientX - rect.left - this.panX) / this.zoom;
            const y = (e.clientY - rect.top - this.panY) / this.zoom;
            this.addNode(type, x - this.NODE_W / 2, y - this.NODE_H / 2);
        });

        // Mouse events
        c.addEventListener('mousedown', (e) => this.onMouseDown(e));
        c.addEventListener('mousemove', (e) => this.onMouseMove(e));
        c.addEventListener('mouseup', (e) => this.onMouseUp(e));
        c.addEventListener('dblclick', (e) => this.onDblClick(e));
        c.addEventListener('wheel', (e) => { e.preventDefault(); this.onWheel(e); }, { passive: false });

        // Keyboard
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Delete' && this.selected && App.currentPage === 'designer') {
                this.deleteSelected();
            }
            if (e.key === 'Escape') {
                this.connecting = null;
                this.render();
            }
        });
    },

    // ---- Node Management ----
    addNode(type, x, y) {
        const agent = this.agents.find(a => a.type === type);
        const id = 'n' + Date.now().toString(36) + Math.random().toString(36).substr(2, 4);
        const node = {
            id,
            type,
            label: agent ? agent.label : type,
            x: Math.round(x),
            y: Math.round(y),
            config: {},
            description: agent ? agent.description || '' : '',
            instructions: '',  // Task-specific instructions for this node
        };
        this.nodes.push(node);
        this.selectNode(node);
        this.render();
    },

    deleteSelected() {
        if (!this.selected) return;
        if (this.selected._isEdge) {
            this.edges = this.edges.filter(e => e !== this.selected);
        } else {
            const nodeId = this.selected.id;
            this.nodes = this.nodes.filter(n => n.id !== nodeId);
            this.edges = this.edges.filter(e => e.source !== nodeId && e.target !== nodeId);
        }
        this.selected = null;
        this.renderProperties(null);
        this.render();
    },

    selectNode(node) {
        this.selected = node;
        this.renderProperties(node);
        this.render();
    },

    // ---- Canvas Mouse Handlers ----
    getCanvasPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left - this.panX) / this.zoom,
            y: (e.clientY - rect.top - this.panY) / this.zoom,
        };
    },

    hitTest(x, y) {
        // Check nodes (reverse for z-order)
        for (let i = this.nodes.length - 1; i >= 0; i--) {
            const n = this.nodes[i];
            if (x >= n.x && x <= n.x + this.NODE_W && y >= n.y && y <= n.y + this.NODE_H) {
                return { type: 'node', node: n };
            }
        }
        // Check output ports
        for (const n of this.nodes) {
            const px = n.x + this.NODE_W;
            const py = n.y + this.NODE_H / 2;
            if (Math.hypot(x - px, y - py) < this.PORT_R + 4) {
                return { type: 'output-port', node: n };
            }
        }
        // Check input ports
        for (const n of this.nodes) {
            const px = n.x;
            const py = n.y + this.NODE_H / 2;
            if (Math.hypot(x - px, y - py) < this.PORT_R + 4) {
                return { type: 'input-port', node: n };
            }
        }
        return null;
    },

    onMouseDown(e) {
        const pos = this.getCanvasPos(e);
        const hit = this.hitTest(pos.x, pos.y);

        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            // Middle click or Alt+click = pan
            this.isPanning = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.canvas.style.cursor = 'grabbing';
            return;
        }

        if (hit) {
            if (hit.type === 'output-port') {
                // Start edge connection
                this.connecting = { source: hit.node, mx: pos.x, my: pos.y };
            } else if (hit.type === 'node') {
                this.dragging = { node: hit.node, offsetX: pos.x - hit.node.x, offsetY: pos.y - hit.node.y };
                this.selectNode(hit.node);
            }
        } else {
            // Click on empty space — start panning
            this.isPanning = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.canvas.style.cursor = 'grabbing';
            this.selected = null;
            this.renderProperties(null);
            this.render();
        }
    },

    onMouseMove(e) {
        const pos = this.getCanvasPos(e);

        if (this.isPanning) {
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            this.panX += dx;
            this.panY += dy;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.render();
            return;
        }

        if (this.dragging) {
            this.dragging.node.x = Math.round(pos.x - this.dragging.offsetX);
            this.dragging.node.y = Math.round(pos.y - this.dragging.offsetY);
            this.render();
            return;
        }

        if (this.connecting) {
            this.connecting.mx = pos.x;
            this.connecting.my = pos.y;
            this.render();
            return;
        }

        // Hover cursor
        const hit = this.hitTest(pos.x, pos.y);
        if (hit && (hit.type === 'output-port' || hit.type === 'input-port')) {
            this.canvas.style.cursor = 'crosshair';
        } else if (hit && hit.type === 'node') {
            this.canvas.style.cursor = 'move';
        } else {
            this.canvas.style.cursor = 'default';
        }
    },

    onMouseUp(e) {
        if (this.isPanning) {
            this.isPanning = false;
            this.canvas.style.cursor = 'default';
            return;
        }

        if (this.connecting) {
            const pos = this.getCanvasPos(e);
            // Check if dropped on input port
            for (const n of this.nodes) {
                const px = n.x;
                const py = n.y + this.NODE_H / 2;
                if (Math.hypot(pos.x - px, pos.y - py) < this.PORT_R + 8) {
                    if (n.id !== this.connecting.source.id) {
                        // Check no duplicate
                        const exists = this.edges.some(e =>
                            e.source === this.connecting.source.id && e.target === n.id);
                        if (!exists) {
                            this.edges.push({
                                id: 'e' + Date.now().toString(36),
                                source: this.connecting.source.id,
                                target: n.id,
                                label: '',
                                condition: '',
                            });
                        }
                    }
                    break;
                }
            }
            this.connecting = null;
            this.render();
            return;
        }

        this.dragging = null;
    },

    onDblClick(e) {
        const pos = this.getCanvasPos(e);
        const hit = this.hitTest(pos.x, pos.y);
        if (hit && hit.type === 'node') {
            // Focus on label editing in properties
            this.selectNode(hit.node);
            const labelInput = document.getElementById('prop-label');
            if (labelInput) labelInput.focus();
        }
    },

    onWheel(e) {
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        const newZoom = Math.max(0.3, Math.min(3, this.zoom + delta));

        // Zoom toward mouse
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
        this.panY = my - (my - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;

        document.getElementById('zoom-level').textContent = Math.round(this.zoom * 100) + '%';
        this.render();
    },

    // ---- Rendering ----
    render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        ctx.clearRect(0, 0, w, h);
        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);

        // Draw grid
        this.drawGrid(ctx);

        // Draw edges
        for (const edge of this.edges) {
            this.drawEdge(ctx, edge);
        }

        // Draw connecting line
        if (this.connecting) {
            const src = this.connecting.source;
            ctx.beginPath();
            ctx.moveTo(src.x + this.NODE_W, src.y + this.NODE_H / 2);
            ctx.lineTo(this.connecting.mx, this.connecting.my);
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw nodes
        for (const node of this.nodes) {
            this.drawNode(ctx, node);
        }

        ctx.restore();
    },

    drawGrid(ctx) {
        const gridSize = 30;
        const vw = this.canvas.width / this.zoom;
        const vh = this.canvas.height / this.zoom;
        const startX = Math.floor(-this.panX / this.zoom / gridSize) * gridSize;
        const startY = Math.floor(-this.panY / this.zoom / gridSize) * gridSize;

        ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        ctx.lineWidth = 1;

        for (let x = startX; x < startX + vw + gridSize; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, startY);
            ctx.lineTo(x, startY + vh + gridSize);
            ctx.stroke();
        }
        for (let y = startY; y < startY + vh + gridSize; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(startX, y);
            ctx.lineTo(startX + vw + gridSize, y);
            ctx.stroke();
        }
    },

    drawNode(ctx, node) {
        const x = node.x, y = node.y;
        const w = this.NODE_W, h = this.NODE_H;
        const color = this.agentColors[node.type] || '#6b7099';
        const isSelected = this.selected && this.selected.id === node.id;

        // Shadow
        ctx.shadowColor = 'rgba(0,0,0,0.3)';
        ctx.shadowBlur = 8;
        ctx.shadowOffsetY = 2;

        // Body
        ctx.fillStyle = '#1a1d2e';
        this.roundRect(ctx, x, y, w, h, 8);
        ctx.fill();
        ctx.shadowColor = 'transparent';

        // Top accent bar
        ctx.fillStyle = color;
        this.roundRectTop(ctx, x, y, w, 4, 8);
        ctx.fill();

        // Border
        ctx.strokeStyle = isSelected ? '#3b82f6' : '#2a2d3e';
        ctx.lineWidth = isSelected ? 2 : 1;
        this.roundRect(ctx, x, y, w, h, 8);
        ctx.stroke();

        // Icon circle
        ctx.fillStyle = color + '22';
        ctx.beginPath();
        ctx.arc(x + 24, y + h / 2 + 2, 14, 0, Math.PI * 2);
        ctx.fill();

        // Type letter
        ctx.fillStyle = color;
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.type[0].toUpperCase(), x + 24, y + h / 2 + 2);

        // Label
        ctx.fillStyle = '#e4e6f0';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const labelText = node.label.length > 14 ? node.label.substr(0, 13) + '…' : node.label;
        ctx.fillText(labelText, x + 46, y + h / 2 - 4);

        // Type sub-label
        ctx.fillStyle = '#6b7099';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText(node.type, x + 46, y + h / 2 + 10);

        // Output port (right)
        ctx.beginPath();
        ctx.arc(x + w, y + h / 2, this.PORT_R, 0, Math.PI * 2);
        ctx.fillStyle = '#2a2d3e';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Input port (left)
        ctx.beginPath();
        ctx.arc(x, y + h / 2, this.PORT_R, 0, Math.PI * 2);
        ctx.fillStyle = '#2a2d3e';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
    },

    drawEdge(ctx, edge) {
        const srcNode = this.nodes.find(n => n.id === edge.source);
        const tgtNode = this.nodes.find(n => n.id === edge.target);
        if (!srcNode || !tgtNode) return;

        const sx = srcNode.x + this.NODE_W;
        const sy = srcNode.y + this.NODE_H / 2;
        const tx = tgtNode.x;
        const ty = tgtNode.y + this.NODE_H / 2;

        const isSelected = this.selected && this.selected._isEdge && this.selected.id === edge.id;
        const midX = (sx + tx) / 2;

        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.bezierCurveTo(midX, sy, midX, ty, tx, ty);
        ctx.strokeStyle = isSelected ? '#3b82f6' : '#4a4f6e';
        ctx.lineWidth = isSelected ? 2.5 : 2;
        ctx.stroke();

        // Arrow
        const angle = Math.atan2(ty - sy, tx - sx);
        const arrowLen = 8;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(tx - arrowLen * Math.cos(angle - 0.4), ty - arrowLen * Math.sin(angle - 0.4));
        ctx.lineTo(tx - arrowLen * Math.cos(angle + 0.4), ty - arrowLen * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fillStyle = isSelected ? '#3b82f6' : '#4a4f6e';
        ctx.fill();
    },

    roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    },

    roundRectTop(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    },

    // ---- Properties Panel ----
    renderProperties(node) {
        const el = document.getElementById('properties-content');

        if (!node || node._isEdge) {
            el.innerHTML = `
                <div class="empty-state small">
                    <i data-lucide="mouse-pointer-click"></i>
                    <p>Select a node to edit</p>
                </div>
            `;
            lucide.createIcons();
            return;
        }

        const agent = this.agents.find(a => a.type === node.type);
        const color = this.agentColors[node.type] || '#6b7099';

        let configHTML = '';
        if (agent && agent.config_schema) {
            configHTML = '<div class="prop-section"><div class="prop-section-title">Configuration</div>';
            for (const [key, schema] of Object.entries(agent.config_schema)) {
                const val = node.config[key] !== undefined ? node.config[key] : (schema.default || '');
                if (schema.type === 'select') {
                    configHTML += `
                        <div class="prop-row">
                            <label>${key}</label>
                            <select class="form-input" onchange="Designer.updateConfig('${node.id}','${key}',this.value)">
                                ${(schema.options || []).map(o => `<option value="${o}" ${val === o ? 'selected' : ''}>${o}</option>`).join('')}
                            </select>
                        </div>
                    `;
                } else if (schema.type === 'number') {
                    configHTML += `
                        <div class="prop-row">
                            <label>${key}</label>
                            <input type="number" value="${val}" min="${schema.min || 0}" max="${schema.max || 999}"
                                   onchange="Designer.updateConfig('${node.id}','${key}',parseInt(this.value))">
                        </div>
                    `;
                } else if (schema.type === 'boolean') {
                    configHTML += `
                        <div class="prop-row">
                            <label>${key}</label>
                            <select class="form-input" onchange="Designer.updateConfig('${node.id}','${key}',this.value==='true')">
                                <option value="true" ${val ? 'selected' : ''}>Yes</option>
                                <option value="false" ${!val ? 'selected' : ''}>No</option>
                            </select>
                        </div>
                    `;
                } else {
                    configHTML += `
                        <div class="prop-row">
                            <label>${key}</label>
                            <input type="text" value="${val}" onchange="Designer.updateConfig('${node.id}','${key}',this.value)">
                        </div>
                    `;
                }
            }
            configHTML += '</div>';
        }

        el.innerHTML = `
            <div class="prop-section">
                <div class="prop-section-title">Node</div>
                <div class="prop-row">
                    <label>Label</label>
                    <input type="text" id="prop-label" value="${App.esc(node.label)}"
                           onchange="Designer.updateLabel('${node.id}', this.value)">
                </div>
                <div class="prop-row">
                    <label>Type</label>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="width:10px;height:10px;border-radius:50%;background:${color};display:inline-block"></span>
                        <span style="font-size:12.5px;color:var(--text-secondary)">${node.type}</span>
                    </div>
                </div>
                <div class="prop-row">
                    <label>Position</label>
                    <span style="font-size:12px;color:var(--text-muted)">x: ${node.x}, y: ${node.y}</span>
                </div>
            </div>
            <div class="prop-section">
                <div class="prop-section-title">Task Instructions</div>
                <div class="prop-row">
                    <textarea id="prop-instructions" class="form-input" rows="4"
                              placeholder="Enter specific instructions for this task...\nE.g.: Search for recent AI trends in healthcare"
                              onchange="Designer.updateInstructions('${node.id}', this.value)"
                              style="width:100%;font-size:12px;resize:vertical;min-height:80px;">${App.esc(node.instructions || '')}</textarea>
                </div>
                <p style="font-size:10px;color:var(--text-muted);margin-top:4px;">These instructions guide what this task does. Output is automatically passed to connected downstream tasks.</p>
            </div>
            ${configHTML}
            <div class="prop-section">
                <button class="btn btn-danger btn-sm delete-node-btn" onclick="Designer.deleteSelected()">
                    <i data-lucide="trash-2"></i> Delete Node
                </button>
            </div>
        `;
        lucide.createIcons();
    },

    updateLabel(nodeId, label) {
        const node = this.nodes.find(n => n.id === nodeId);
        if (node) { node.label = label; this.render(); }
    },

    updateInstructions(nodeId, instructions) {
        const node = this.nodes.find(n => n.id === nodeId);
        if (node) { node.instructions = instructions; }
    },

    updateConfig(nodeId, key, value) {
        const node = this.nodes.find(n => n.id === nodeId);
        if (node) { node.config[key] = value; }
    },

    // ---- Zoom Controls ----
    zoomIn() {
        this.zoom = Math.min(3, this.zoom + 0.2);
        document.getElementById('zoom-level').textContent = Math.round(this.zoom * 100) + '%';
        this.render();
    },

    zoomOut() {
        this.zoom = Math.max(0.3, this.zoom - 0.2);
        document.getElementById('zoom-level').textContent = Math.round(this.zoom * 100) + '%';
        this.render();
    },

    zoomFit() {
        if (this.nodes.length === 0) {
            this.zoom = 1; this.panX = 0; this.panY = 0;
        } else {
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            for (const n of this.nodes) {
                minX = Math.min(minX, n.x);
                minY = Math.min(minY, n.y);
                maxX = Math.max(maxX, n.x + this.NODE_W);
                maxY = Math.max(maxY, n.y + this.NODE_H);
            }
            const padding = 60;
            const bw = maxX - minX + padding * 2;
            const bh = maxY - minY + padding * 2;
            this.zoom = Math.min(this.canvas.width / bw, this.canvas.height / bh, 2);
            this.panX = (this.canvas.width - bw * this.zoom) / 2 - minX * this.zoom + padding * this.zoom;
            this.panY = (this.canvas.height - bh * this.zoom) / 2 - minY * this.zoom + padding * this.zoom;
        }
        document.getElementById('zoom-level').textContent = Math.round(this.zoom * 100) + '%';
        this.render();
    },

    clearCanvas() {
        if (this.nodes.length === 0) return;
        if (!confirm('Clear all nodes and edges?')) return;
        this.nodes = [];
        this.edges = [];
        this.selected = null;
        this.renderProperties(null);
        this.render();
    },

    // ---- Workflow CRUD ----
    newWorkflow() {
        this.currentWorkflow = null;
        this.nodes = [];
        this.edges = [];
        this.selected = null;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        document.getElementById('designer-subtitle').textContent = 'New Workflow';
        this.renderProperties(null);
        this.render();
    },

    async saveWorkflow() {
        if (this.nodes.length === 0) {
            App.showToast('warning', 'Empty Workflow', 'Add some nodes before saving.');
            return;
        }

        const name = this.currentWorkflow?.name || prompt('Workflow name:', 'My Workflow');
        if (!name) return;

        try {
            const payload = {
                name,
                description: '',
                nodes: this.nodes,
                edges: this.edges,
                tags: [],
            };

            let wf;
            if (this.currentWorkflow) {
                wf = await App.api(`/api/workflows/${this.currentWorkflow.id}`, {
                    method: 'PUT',
                    body: JSON.stringify(payload),
                });
            } else {
                wf = await App.api('/api/workflows', {
                    method: 'POST',
                    body: JSON.stringify(payload),
                });
            }

            this.currentWorkflow = wf;
            document.getElementById('designer-subtitle').textContent = wf.name;
            App.showToast('success', 'Saved', `Workflow "${wf.name}" saved successfully.`);
        } catch (e) {
            App.showToast('error', 'Save Failed', e.message);
        }
    },

    async showTemplates() {
        try {
            const [templatesRes, workflowsRes] = await Promise.all([
                App.api('/api/workflows/templates'),
                App.api('/api/workflows'),
            ]);

            const templates = templatesRes.templates || [];
            const workflows = workflowsRes.workflows || [];

            document.getElementById('side-sheet-title').textContent = 'Load Workflow';
            const body = document.getElementById('side-sheet-body');

            let html = '<h4 style="font-size:12px;color:var(--text-muted);margin-bottom:8px;text-transform:uppercase">Templates</h4>';
            html += templates.map(t => `
                <div class="workflow-item" onclick="Designer.loadTemplate('${t.id}')">
                    <div class="workflow-item-name">${App.esc(t.name)}</div>
                    <div class="workflow-item-desc">${App.esc(t.description)}</div>
                    <div class="workflow-item-meta">
                        <span>${t.nodes.length} nodes</span>
                        <span>${t.edges.length} edges</span>
                    </div>
                </div>
            `).join('');

            if (workflows.length > 0) {
                html += '<h4 style="font-size:12px;color:var(--text-muted);margin:16px 0 8px;text-transform:uppercase">Saved Workflows</h4>';
                html += workflows.map(w => `
                    <div class="workflow-item" onclick="Designer.loadWorkflow('${w.id}')">
                        <div class="workflow-item-name">${App.esc(w.name)}</div>
                        <div class="workflow-item-desc">${App.esc(w.description || 'No description')}</div>
                        <div class="workflow-item-meta">
                            <span>${w.nodes?.length || 0} nodes</span>
                            <span>v${w.version}</span>
                            <span>${App.timeAgo(w.updated_at)}</span>
                        </div>
                        <div class="workflow-item-actions">
                            <button class="btn btn-sm btn-outline" onclick="event.stopPropagation();Designer.deleteWorkflow('${w.id}')">
                                <i data-lucide="trash-2"></i>
                            </button>
                        </div>
                    </div>
                `).join('');
            }

            body.innerHTML = html;
            document.getElementById('workflow-list-overlay').classList.remove('hidden');
            lucide.createIcons();
        } catch (e) {
            App.showToast('error', 'Load Failed', e.message);
        }
    },

    async loadWorkflow(id) {
        try {
            const wf = await App.api(`/api/workflows/${id}`);
            this.currentWorkflow = wf;
            this.nodes = wf.nodes || [];
            this.edges = wf.edges || [];
            document.getElementById('designer-subtitle').textContent = wf.name;
            this.closeOverlay();
            this.selected = null;
            this.renderProperties(null);
            this.zoomFit();
            App.showToast('info', 'Loaded', `Workflow "${wf.name}" loaded.`);
        } catch (e) {
            App.showToast('error', 'Load Failed', e.message);
        }
    },

    async loadTemplate(id) {
        try {
            const res = await App.api('/api/workflows/templates');
            const template = (res.templates || []).find(t => t.id === id);
            if (!template) throw new Error('Template not found');

            this.currentWorkflow = null;
            this.nodes = template.nodes || [];
            this.edges = template.edges || [];
            document.getElementById('designer-subtitle').textContent = template.name + ' (Template)';
            this.closeOverlay();
            this.selected = null;
            this.renderProperties(null);
            this.zoomFit();
            App.showToast('info', 'Template Loaded', template.name);
        } catch (e) {
            App.showToast('error', 'Load Failed', e.message);
        }
    },

    async deleteWorkflow(id) {
        if (!confirm('Delete this workflow?')) return;
        try {
            await App.api(`/api/workflows/${id}`, { method: 'DELETE' });
            if (this.currentWorkflow && this.currentWorkflow.id === id) {
                this.newWorkflow();
            }
            this.showTemplates(); // Refresh list
            App.showToast('info', 'Deleted', 'Workflow deleted.');
        } catch (e) {
            App.showToast('error', 'Delete Failed', e.message);
        }
    },

    closeOverlay() {
        document.getElementById('workflow-list-overlay').classList.add('hidden');
    },

    // ---- Run Workflow ----
    runWorkflow() {
        if (this.nodes.length === 0) {
            App.showToast('warning', 'Empty Workflow', 'Add nodes before running.');
            return;
        }
        // Save first if not saved
        if (!this.currentWorkflow) {
            App.showToast('warning', 'Save First', 'Please save the workflow before running.');
            return;
        }
        document.getElementById('run-modal').classList.remove('hidden');
        lucide.createIcons();
    },

    closeRunModal() {
        document.getElementById('run-modal').classList.add('hidden');
    },

    async executeWorkflow() {
        const objective = document.getElementById('run-objective').value.trim();
        if (!objective) {
            App.showToast('warning', 'Missing Objective', 'Please enter an objective.');
            return;
        }

        try {
            const result = await App.api('/api/executions', {
                method: 'POST',
                body: JSON.stringify({
                    workflow_id: this.currentWorkflow.id,
                    objective: objective,
                    config: {},
                }),
            });

            this.closeRunModal();
            document.getElementById('run-objective').value = '';
            App.showToast('success', 'Execution Started', `Workflow "${this.currentWorkflow.name}" is now running.`);
            App.navigate('executions');
        } catch (e) {
            App.showToast('error', 'Start Failed', e.message);
        }
    },

    // ---- LangGraph Code Generation ----
    _generatedCode: '',

    async generateCode() {
        if (this.nodes.length === 0) {
            App.showToast('warning', 'Empty Workflow', 'Add nodes before generating code.');
            return;
        }
        if (!this.currentWorkflow) {
            App.showToast('warning', 'Save First', 'Please save the workflow before generating code.');
            return;
        }

        try {
            const res = await App.api(`/api/workflows/${this.currentWorkflow.id}/generate-code`, {
                method: 'POST',
            });
            this._generatedCode = res.code || '';
            document.getElementById('generated-code').textContent = this._generatedCode;
            document.getElementById('code-modal').classList.remove('hidden');
            lucide.createIcons();
            App.showToast('success', 'Code Generated', 'LangGraph Python code is ready.');
        } catch (e) {
            App.showToast('error', 'Generation Failed', e.message);
        }
    },

    closeCodeModal() {
        document.getElementById('code-modal').classList.add('hidden');
    },

    copyGeneratedCode() {
        if (!this._generatedCode) return;
        navigator.clipboard.writeText(this._generatedCode).then(() => {
            App.showToast('success', 'Copied', 'Code copied to clipboard.');
        }).catch(() => {
            // Fallback
            const ta = document.createElement('textarea');
            ta.value = this._generatedCode;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            App.showToast('success', 'Copied', 'Code copied to clipboard.');
        });
    },

    downloadGeneratedCode() {
        if (!this._generatedCode) return;
        const name = (this.currentWorkflow?.name || 'workflow').toLowerCase().replace(/\s+/g, '_');
        const blob = new Blob([this._generatedCode], { type: 'text/x-python' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${name}_langgraph.py`;
        a.click();
        URL.revokeObjectURL(url);
        App.showToast('info', 'Downloaded', `${name}_langgraph.py saved.`);
    },
};
