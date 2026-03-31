/**
 * API client for the GitHub License Analyzer backend
 */
const ApiClient = {
    baseUrl: '',  // Same origin

    /**
     * Health check
     */
    async healthCheck() {
        const resp = await fetch(`${this.baseUrl}/health`, { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
    },

    /**
     * Stream analysis logs from /api/v1/analyze-stream
     * @param {File} file - Excel file to analyze
     * @param {function} onLine - Callback for each log line
     * @param {AbortSignal} signal - Optional abort signal
     */
    async analyzeStream(file, onLine, signal) {
        const form = new FormData();
        form.append('file', file);

        const resp = await fetch(`${this.baseUrl}/api/v1/analyze-stream`, {
            method: 'POST',
            body: form,
            signal,
            cache: 'no-store'
        });

        if (!resp.ok) {
            const text = await resp.text();
            let detail;
            try { detail = JSON.parse(text).detail; } catch { detail = text; }
            throw new Error(detail || `HTTP ${resp.status}`);
        }

        await this._consumeStream(resp, onLine);
    },

    /**
     * Stream analysis logs + send email
     * @param {File} file - Excel file
     * @param {string} email - Recipient email
     * @param {function} onLine - Callback for each log line
     * @param {AbortSignal} signal - Optional abort signal
     */
    async analyzeStreamEmail(file, email, onLine, signal) {
        const form = new FormData();
        form.append('file', file);
        form.append('email', email);

        const resp = await fetch(`${this.baseUrl}/api/v1/analyze-stream-email`, {
            method: 'POST',
            body: form,
            signal,
            cache: 'no-store'
        });

        if (!resp.ok) {
            const text = await resp.text();
            let detail;
            try { detail = JSON.parse(text).detail; } catch { detail = text; }
            throw new Error(detail || `HTTP ${resp.status}`);
        }

        await this._consumeStream(resp, onLine);
    },

    /**
     * Download analysis result as Excel file
     * @param {File} file - Excel file
     * @param {AbortSignal} signal - Optional abort signal
     * @returns {Promise<Blob>} Excel file blob
     */
    async analyzeAndDownload(file, signal) {
        const form = new FormData();
        form.append('file', file);

        const resp = await fetch(`${this.baseUrl}/api/v1/analyze-and-download`, {
            method: 'POST',
            body: form,
            signal,
            cache: 'no-store'
        });

        if (!resp.ok) {
            const text = await resp.text();
            let detail;
            try { detail = JSON.parse(text).detail; } catch { detail = text; }
            throw new Error(detail || `HTTP ${resp.status}`);
        }

        return resp.blob();
    },

    /**
     * Subscribe to the live log SSE stream (server monitor panel).
     * @param {function} onMessage - Called with parsed JSON object per event
     * @param {function} [onError]  - Called on connection error
     * @returns {EventSource} The EventSource instance (caller can close it)
     */
    subscribeLiveLogs(onMessage, onError) {
        const es = new EventSource(`${this.baseUrl}/api/v1/logs/live`);
        es.onmessage = (event) => {
            try { onMessage(JSON.parse(event.data)); } catch { /* malformed JSON — skip */ }
        };
        es.onerror = (err) => { if (onError) onError(err); };
        return es;
    },

    /**
     * Consume a streaming text response line by line
     */
    async _consumeStream(resp, onLine) {
        const reader = resp.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (value) {
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                // Keep the last (possibly incomplete) line in buffer
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (line.trim()) {
                        onLine(line);
                    }
                }
            }

            if (done) {
                // Flush remaining buffer
                if (buffer.trim()) {
                    onLine(buffer);
                }
                break;
            }
        }
    }
};
