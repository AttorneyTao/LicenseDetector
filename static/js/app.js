/**
 * Main application controller
 */
(function () {
    'use strict';

    // ===== State =====
    const state = {
        file: null,         // Selected File object
        fileBuffer: null,   // ArrayBuffer of the file (for reuse)
        mode: 'download',   // 'download' | 'email'
        email: '',
        analyzing: false,
        logs: [],           // All raw log lines (for export)
        progress: 0,
        currentFilter: 'all',
        historyId: null,    // Current analysis history entry ID
        startTime: null,
        abortController: null
    };

    // ===== Initialization =====
    document.addEventListener('DOMContentLoaded', () => {
        initTheme();
        initHealthCheck();
        initDropZone();
        initModeSelection();
        initButtons();
        initLogToolbar();
        initHistory();
        initBeforeUnload();
        loadSheetJS();
    });

    // ===== Theme =====
    function initTheme() {
        const theme = Storage.getTheme();
        document.documentElement.setAttribute('data-theme', theme);

        document.getElementById('themeToggle').addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            Storage.setTheme(next);
        });
    }

    // ===== Health Check =====
    function initHealthCheck() {
        checkHealth();
        setInterval(checkHealth, 30000);
    }

    async function checkHealth() {
        try {
            const data = await ApiClient.healthCheck();
            Components.updateHealthBadge(data.status === 'ok' ? 'ok' : 'error');
        } catch {
            Components.updateHealthBadge('error');
        }
    }

    // ===== Drop Zone =====
    function initDropZone() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        Components.setupDropZone(dropZone, fileInput, handleFile);

        document.getElementById('removeFile').addEventListener('click', () => {
            state.file = null;
            state.fileBuffer = null;
            fileInput.value = '';
            Components.resetFileDisplay();
            updateStartBtn();
        });
    }

    function handleFile(file) {
        // Validate
        if (!file.name.toLowerCase().endsWith('.xlsx')) {
            Components.showToast('请选择 .xlsx 格式的文件', 'error');
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            Components.showToast('文件大小不能超过 10MB', 'error');
            return;
        }

        state.file = file;
        Components.showFileInfo(file);
        updateStartBtn();

        // Read file buffer for reuse and preview
        const reader = new FileReader();
        reader.onload = (e) => {
            state.fileBuffer = e.target.result;
            tryPreviewFile(e.target.result, file.name);
        };
        reader.readAsArrayBuffer(file);
    }

    function tryPreviewFile(buffer, filename) {
        if (typeof XLSX === 'undefined') return;
        try {
            const wb = XLSX.read(buffer, { type: 'array' });
            const ws = wb.Sheets[wb.SheetNames[0]];
            const data = XLSX.utils.sheet_to_json(ws, { defval: '' });
            Components.renderFilePreview(data);
        } catch {
            // SheetJS not available or parse error — skip preview
        }
    }

    // ===== Mode Selection =====
    function initModeSelection() {
        const radios = document.querySelectorAll('input[name="mode"]');
        const emailSection = document.getElementById('emailSection');
        const emailInput = document.getElementById('emailInput');

        radios.forEach(radio => {
            radio.addEventListener('change', () => {
                state.mode = radio.value;
                emailSection.hidden = radio.value !== 'email';
                updateStartBtn();
            });
        });

        emailInput.addEventListener('input', () => {
            state.email = emailInput.value.trim();
            updateStartBtn();
        });
    }

    // ===== Start Button =====
    function updateStartBtn() {
        const btn = document.getElementById('startBtn');
        const valid = state.file &&
            !state.analyzing &&
            (state.mode !== 'email' || isValidEmail(state.email));
        btn.disabled = !valid;
    }

    function isValidEmail(email) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    // ===== Buttons =====
    function initButtons() {
        document.getElementById('startBtn').addEventListener('click', startAnalysis);
        document.getElementById('downloadResult').addEventListener('click', downloadResult);
        document.getElementById('retryBtn').addEventListener('click', retryAnalysis);
    }

    // ===== Log Toolbar =====
    function initLogToolbar() {
        // Filters
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.currentFilter = btn.dataset.filter;
                Components.filterLogs(btn.dataset.filter);
            });
        });

        // Search
        let searchTimeout;
        document.getElementById('logSearch').addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                Components.searchLogs(e.target.value.trim());
            }, 200);
        });

        // Copy
        document.getElementById('copyLogs').addEventListener('click', () => {
            if (state.logs.length === 0) {
                Components.showToast('没有日志可复制', 'info');
                return;
            }
            navigator.clipboard.writeText(state.logs.join('\n'))
                .then(() => Components.showToast('日志已复制到剪贴板', 'success'))
                .catch(() => Components.showToast('复制失败', 'error'));
        });

        // Export
        document.getElementById('exportLogs').addEventListener('click', () => {
            if (state.logs.length === 0) {
                Components.showToast('没有日志可导出', 'info');
                return;
            }
            const blob = new Blob([state.logs.join('\n')], { type: 'text/plain;charset=utf-8' });
            const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            Components.downloadBlob(blob, `analysis-log-${ts}.txt`);
        });
    }

    // ===== History =====
    function initHistory() {
        Components.renderHistory(Storage.getAll());

        document.getElementById('clearHistory').addEventListener('click', () => {
            Storage.clearAll();
            Components.renderHistory([]);
            Components.showToast('历史记录已清除', 'info');
        });
    }

    // ===== Before Unload =====
    function initBeforeUnload() {
        window.addEventListener('beforeunload', (e) => {
            if (state.analyzing) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    }

    // ===== SheetJS Loader =====
    function loadSheetJS() {
        const script = document.createElement('script');
        script.src = 'https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js';
        script.async = true;
        script.onerror = () => { /* graceful degradation */ };
        document.head.appendChild(script);
    }

    // ===== Analysis Flow =====
    async function startAnalysis() {
        if (state.analyzing || !state.file) return;

        state.analyzing = true;
        state.logs = [];
        state.progress = 0;
        state.startTime = Date.now();
        state.abortController = new AbortController();

        const startBtn = document.getElementById('startBtn');
        startBtn.disabled = true;
        startBtn.classList.add('loading');

        // Show progress panel
        const progressCard = document.getElementById('progressCard');
        progressCard.hidden = false;
        document.getElementById('resultActions').hidden = true;
        Components.clearLog();
        Components.setProgressIndeterminate();
        Components.setStatusText('正在上传文件并开始分析...');

        // Add history entry
        const historyEntry = Storage.add({
            filename: state.file.name,
            rowCount: 0,
            status: 'running',
            mode: state.mode,
            email: state.mode === 'email' ? state.email : null
        });
        state.historyId = historyEntry.id;
        Components.renderHistory(Storage.getAll());

        // Scroll to progress
        progressCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

        let rowCount = 0;
        let success = false;

        try {
            // Recreate File from buffer for the stream request
            const streamFile = new File(
                [state.fileBuffer],
                state.file.name,
                { type: state.file.type }
            );

            const onLine = (line) => {
                state.logs.push(line);
                const parsed = StreamParser.parseLogLine(line);
                if (!parsed) return;

                Components.appendLogLine(parsed);
                Components.autoScrollLog();

                // Apply current filter
                if (state.currentFilter !== 'all') {
                    Components.filterLogs(state.currentFilter);
                }

                // Extract progress
                const progress = StreamParser.extractProgress(line);
                if (progress) {
                    state.progress = progress.percent;
                    rowCount = progress.total;
                    Components.updateProgressBar(progress.percent);
                    Components.setStatusText(`正在分析... 已完成 ${progress.current}/${progress.total}`);
                }

                // Check completion
                if (StreamParser.isComplete(line)) {
                    Components.updateProgressBar(100);
                    Components.setStatusText('分析完成');
                }
            };

            if (state.mode === 'email') {
                await ApiClient.analyzeStreamEmail(
                    streamFile, state.email, onLine, state.abortController.signal
                );
                Components.updateProgressBar(100);
                Components.setStatusText('分析完成，结果已发送到邮箱');
                Components.showToast('分析完成，结果已发送到邮箱', 'success');
                success = true;
            } else {
                await ApiClient.analyzeStream(
                    streamFile, onLine, state.abortController.signal
                );
                Components.updateProgressBar(100);
                Components.setStatusText('分析完成，正在准备下载...');
                success = true;
            }

        } catch (err) {
            if (err.name === 'AbortError') {
                Components.setStatusText('分析已取消');
                Components.showToast('分析已取消', 'info');
            } else {
                Components.setStatusText(`分析失败: ${err.message}`);
                Components.showToast(`分析失败: ${err.message}`, 'error');
            }
        }

        // Update history
        const durationMs = Date.now() - state.startTime;
        Storage.update(state.historyId, {
            status: success ? 'success' : 'error',
            rowCount: rowCount,
            durationMs: durationMs
        });
        Components.renderHistory(Storage.getAll());

        // Show result actions
        if (success) {
            const resultActions = document.getElementById('resultActions');
            resultActions.hidden = false;
            // Show download button only in download mode
            document.getElementById('downloadResult').style.display =
                state.mode === 'download' ? '' : 'none';
        }

        // Reset state
        state.analyzing = false;
        startBtn.classList.remove('loading');
        updateStartBtn();
    }

    async function downloadResult() {
        const btn = document.getElementById('downloadResult');
        btn.disabled = true;
        btn.textContent = '正在下载...';

        try {
            const downloadFile = new File(
                [state.fileBuffer],
                state.file.name,
                { type: state.file.type }
            );
            const blob = await ApiClient.analyzeAndDownload(downloadFile);
            const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            Components.downloadBlob(blob, `output_${ts}.xlsx`);
            Components.showToast('结果文件下载成功', 'success');
            Components.setStatusText('分析完成，结果已下载');
        } catch (err) {
            Components.showToast(`下载失败: ${err.message}`, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                下载结果文件
            `;
        }
    }

    function retryAnalysis() {
        document.getElementById('resultActions').hidden = true;
        document.getElementById('progressCard').hidden = true;
        if (state.file) {
            startAnalysis();
        }
    }
})();
