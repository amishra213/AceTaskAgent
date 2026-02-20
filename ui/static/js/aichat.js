/**
 * AceTaskAgent UI - AI Chat Assistant
 * Floating chat widget powered by DeepSeek via /api/aichat
 */

const AIChat = {
    isOpen: false,
    history: [],
    isLoading: false,

    toggle() {
        this.isOpen = !this.isOpen;
        const panel = document.getElementById('ai-chat-panel');
        const toggle = document.getElementById('ai-chat-toggle');

        if (this.isOpen) {
            panel.classList.add('open');
            toggle.style.display = 'none';
            document.getElementById('chat-input').focus();
        } else {
            panel.classList.remove('open');
            toggle.style.display = 'flex';
        }
        lucide.createIcons();
    },

    handleKey(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.send();
        }
    },

    async send() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        if (!message || this.isLoading) return;

        // Add user message to UI
        this._addMessage('user', message);
        this.history.push({ role: 'user', content: message });
        input.value = '';
        this._autoResize(input);

        // Show typing indicator
        this.isLoading = true;
        this._showTyping(true);
        document.getElementById('chat-send-btn').disabled = true;

        try {
            const res = await fetch('/api/aichat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    history: this.history.slice(-20),
                    include_logs: true,
                }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            const reply = data.response || 'No response received.';

            this.history.push({ role: 'assistant', content: reply });
            this._addMessage('assistant', reply, data.duration_ms, data.tokens_used);

        } catch (err) {
            this._addMessage('system', `Error: ${err.message}`);
        } finally {
            this.isLoading = false;
            this._showTyping(false);
            document.getElementById('chat-send-btn').disabled = false;
        }
    },

    clearHistory() {
        this.history = [];
        const container = document.getElementById('chat-messages');
        container.innerHTML = `
            <div class="chat-msg system">
                Chat cleared. Ask me anything about AceTaskAgent!
            </div>
        `;
        lucide.createIcons();
    },

    // ---- Internal Helpers ----

    _addMessage(role, content, durationMs, tokens) {
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = `chat-msg ${role}`;

        // Format content: escape HTML but preserve line breaks
        let html = this._escapeHtml(content);

        // Basic markdown-like formatting for assistant messages
        if (role === 'assistant') {
            html = this._formatMarkdown(html);
        }

        let metaHtml = '';
        if (role === 'assistant' && (durationMs || tokens)) {
            const parts = [];
            if (durationMs) parts.push(`${(durationMs / 1000).toFixed(1)}s`);
            if (tokens) parts.push(`${tokens} tokens`);
            metaHtml = `<div class="msg-meta">${parts.join(' · ')}</div>`;
        }

        div.innerHTML = html + metaHtml;
        container.appendChild(div);

        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    },

    _showTyping(show) {
        const container = document.getElementById('chat-messages');
        const existing = container.querySelector('.chat-typing');
        if (existing) existing.remove();

        if (show) {
            const div = document.createElement('div');
            div.className = 'chat-typing';
            div.innerHTML = '<span></span><span></span><span></span>';
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
    },

    _escapeHtml(str) {
        if (!str) return '';
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    },

    _formatMarkdown(html) {
        // Bold: **text**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:1px 4px;border-radius:3px;font-size:12px;">$1</code>');
        // Code blocks: ```...```
        html = html.replace(/```([\s\S]*?)```/g, '<pre style="background:rgba(0,0,0,0.3);padding:8px 10px;border-radius:6px;overflow-x:auto;font-size:12px;margin:6px 0;">$1</pre>');
        return html;
    },

    _autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
    },
};

// Auto-resize textarea on input
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('input', () => AIChat._autoResize(chatInput));
    }
});
