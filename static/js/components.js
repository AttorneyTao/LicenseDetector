/**
 * UI component rendering functions
 */
const Components = {
    /**
     * Update health badge status
     */
    updateHealthBadge(status) {
        const badge = document.getElementById('healthBadge');
        const text = badge.querySelector('.health-text');
        badge.classList.remove('ok', 'error');

        if (status === 'ok') {
            badge.classList.add('ok');
            text.textContent = '服务正常';
        } else if (status === 'error') {
            badge.classList.add('error');
            text.textContent = '服务异常';
        } else {
            text.textContent = '检查中...';
        }
    },

    /**
     * Setup drag-and-drop zone
     */
    setupDropZone(zoneEl, fileInputEl, onFile) {
        const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };

        zoneEl.addEventListener('click', () => fileInputEl.click());

        zoneEl.addEventListener('dragenter', (e) => {
            prevent(e);
            zoneEl.classList.add('dragover');
        });
        zoneEl.addEventListener('dragover', (e) => {
            prevent(e);
            zoneEl.classList.add('dragover');
        });
        zoneEl.addEventListener('dragleave', (e) => {
            prevent(e);
            zoneEl.classList.remove('dragover');
        });
        zoneEl.addEventListener('drop', (e) => {
            prevent(e);
            zoneEl.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) onFile(files[0]);
        });

        fileInputEl.addEventListener('change', () => {
            if (fileInputEl.files.length > 0) {
                onFile(fileInputEl.files[0]);
            }
        });
    },

    /**
     * Show file info
     */
    showFileInfo(file) {
        const infoEl = document.getElementById('fileInfo');
        const nameEl = document.getElementById('fileName');
        const sizeEl = document.getElementById('fileSize');

        nameEl.textContent = file.name;
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        sizeEl.textContent = `(${sizeMB} MB)`;
        infoEl.hidden = false;
        document.getElementById('dropZone').style.display = 'none';
    },

    /**
     * Reset file display
     */
    resetFileDisplay() {
        document.getElementById('fileInfo').hidden = true;
        document.getElementById('dropZone').style.display = '';
        document.getElementById('filePreview').hidden = true;
        document.getElementById('previewTableWrap').innerHTML = '';
    },

    /**
     * Render file preview table (first 5 rows)
     */
    renderFilePreview(data) {
        const previewEl = document.getElementById('filePreview');
        const wrapEl = document.getElementById('previewTableWrap');

        if (!data || !data.length) {
            previewEl.hidden = true;
            return;
        }

        const headers = Object.keys(data[0]);
        const rows = data.slice(0, 5);

        let html = '<table><thead><tr>';
        headers.forEach(h => { html += `<th>${this._escapeHtml(h)}</th>`; });
        html += '</tr></thead><tbody>';
        rows.forEach(row => {
            html += '<tr>';
            headers.forEach(h => {
                const val = row[h] != null ? String(row[h]) : '';
                html += `<td title="${this._escapeHtml(val)}">${this._escapeHtml(val)}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table>';

        wrapEl.innerHTML = html;
        previewEl.hidden = false;
    },

    /**
     * Update progress bar
     */
    updateProgressBar(percent) {
        const fill = document.getElementById('progressFill');
        const text = document.getElementById('progressText');
        fill.classList.remove('indeterminate');
        fill.style.width = `${Math.min(percent, 100)}%`;
        text.textContent = `${Math.round(percent)}%`;
    },

    /**
     * Set progress bar to indeterminate
     */
    setProgressIndeterminate() {
        const fill = document.getElementById('progressFill');
        fill.classList.add('indeterminate');
        fill.style.width = '';
        document.getElementById('progressText').textContent = '...';
    },

    /**
     * Set status text
     */
    setStatusText(text) {
        document.getElementById('statusText').textContent = text;
    },

    /**
     * Append a log line to the viewer
     * Returns the created element
     */
    appendLogLine(parsed) {
        const viewer = document.getElementById('logViewer');
        const div = document.createElement('div');
        div.className = 'log-line ' + StreamParser.getLogClass(parsed.type);
        div.dataset.type = parsed.type;
        div.textContent = parsed.message;

        // Cap DOM at 2000 lines
        if (viewer.children.length >= 2000) {
            viewer.removeChild(viewer.firstChild);
        }

        viewer.appendChild(div);
        return div;
    },

    /**
     * Auto-scroll log viewer if user is near bottom
     */
    autoScrollLog() {
        const viewer = document.getElementById('logViewer');
        const isNearBottom = viewer.scrollHeight - viewer.scrollTop - viewer.clientHeight < 60;
        if (isNearBottom) {
            viewer.scrollTop = viewer.scrollHeight;
        }
    },

    /**
     * Clear log viewer
     */
    clearLog() {
        document.getElementById('logViewer').innerHTML = '';
    },

    /**
     * Apply filter to log lines
     */
    filterLogs(filterType) {
        const viewer = document.getElementById('logViewer');
        const lines = viewer.querySelectorAll('.log-line');
        lines.forEach(line => {
            if (filterType === 'all') {
                line.classList.remove('filtered');
            } else {
                const match = line.dataset.type === filterType;
                line.classList.toggle('filtered', !match);
            }
        });
    },

    /**
     * Search and highlight in log lines
     */
    searchLogs(query) {
        const viewer = document.getElementById('logViewer');
        const lines = viewer.querySelectorAll('.log-line');
        lines.forEach(line => {
            // Remove old marks
            const text = line.textContent;
            if (!query) {
                line.innerHTML = '';
                line.textContent = text;
                return;
            }
            const lowerText = text.toLowerCase();
            const lowerQuery = query.toLowerCase();
            const idx = lowerText.indexOf(lowerQuery);
            if (idx === -1) {
                line.innerHTML = '';
                line.textContent = text;
            } else {
                const before = this._escapeHtml(text.substring(0, idx));
                const match = this._escapeHtml(text.substring(idx, idx + query.length));
                const after = this._escapeHtml(text.substring(idx + query.length));
                line.innerHTML = `${before}<mark>${match}</mark>${after}`;
            }
        });
    },

    /**
     * Render history list
     */
    renderHistory(entries) {
        const listEl = document.getElementById('historyList');

        if (!entries || entries.length === 0) {
            listEl.innerHTML = '<p class="empty-hint">暂无分析记录</p>';
            return;
        }

        listEl.innerHTML = entries.map(entry => {
            const date = new Date(entry.timestamp);
            const timeStr = date.toLocaleString('zh-CN', {
                month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit'
            });
            const duration = entry.durationMs
                ? `${Math.round(entry.durationMs / 1000)}秒`
                : '-';
            const modeStr = entry.mode === 'email' ? `邮件: ${entry.email}` : '下载';

            return `
                <div class="history-item">
                    <div class="history-item-left">
                        <span class="history-status ${entry.status}"></span>
                        <span class="history-filename" title="${this._escapeHtml(entry.filename)}">${this._escapeHtml(entry.filename)}</span>
                    </div>
                    <div class="history-item-right">
                        <span>${entry.rowCount || '?'} 行</span>
                        <span>${modeStr}</span>
                        <span>${duration}</span>
                        <span>${timeStr}</span>
                    </div>
                </div>
            `;
        }).join('');
    },

    /**
     * Show a toast notification
     */
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('fadeout');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    },

    /**
     * Trigger file download from blob
     */
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },

    // ===== Monitor Panel =====

    setMonitorStatus(status) {
        const dot = document.getElementById('monitorStatusDot');
        const text = document.getElementById('monitorStatusText');
        dot.className = 'monitor-status-dot ' + status;
        const labels = { ok: '已连接', error: '连接断开', connecting: '连接中...' };
        text.textContent = labels[status] || '未知';
    },

    renderMonitorJobs(jobs) {
        const container = document.getElementById('monitorJobs');
        const entries = Object.entries(jobs);
        if (entries.length === 0) {
            container.innerHTML = '<span class="monitor-no-jobs">暂无活动作业</span>';
            return;
        }
        container.innerHTML = entries.map(([jobId, job]) => {
            const p = job.lastProgress;
            const progressText = p ? `${p.current}/${p.total} (${p.percent}%)` : '处理中...';
            const statusClass = job.status === 'done' ? 'success'
                : job.status === 'error' ? 'error' : 'running';
            return `<div class="monitor-job-chip ${statusClass}">` +
                `<span class="monitor-job-id">${this._escapeHtml(jobId)}</span>` +
                `<span class="monitor-job-progress">${this._escapeHtml(progressText)}</span>` +
                `</div>`;
        }).join('');
    },

    appendMonitorLine(parsed, jobId, timestamp) {
        const viewer = document.getElementById('monitorLogViewer');
        const div = document.createElement('div');
        div.className = 'log-line ' + StreamParser.getLogClass(parsed.type);
        div.dataset.type = parsed.type;
        div.dataset.jobId = jobId || '';

        const time = timestamp ? new Date(timestamp).toLocaleTimeString('zh-CN', {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        }) : '';
        div.textContent = `[${time}][${jobId}] ${parsed.message}`;

        if (viewer.children.length >= 2000) {
            viewer.removeChild(viewer.firstChild);
        }
        viewer.appendChild(div);

        const isNearBottom = viewer.scrollHeight - viewer.scrollTop - viewer.clientHeight < 60;
        if (isNearBottom) viewer.scrollTop = viewer.scrollHeight;
    },

    filterMonitorLogs(filterType) {
        const viewer = document.getElementById('monitorLogViewer');
        viewer.querySelectorAll('.log-line').forEach(line => {
            if (filterType === 'all') {
                line.classList.remove('filtered');
            } else {
                line.classList.toggle('filtered', line.dataset.type !== filterType);
            }
        });
    },

    searchMonitorLogs(query) {
        const viewer = document.getElementById('monitorLogViewer');
        viewer.querySelectorAll('.log-line').forEach(line => {
            if (!query) {
                line.style.display = '';
                return;
            }
            const matches = line.textContent.toLowerCase().includes(query.toLowerCase())
                || (line.dataset.jobId || '').includes(query);
            line.style.display = matches ? '' : 'none';
        });
    }
};
