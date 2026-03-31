/**
 * localStorage history manager
 */
const Storage = {
    KEY: 'license-analyzer-history',
    MAX_ENTRIES: 20,

    _read() {
        try {
            const data = localStorage.getItem(this.KEY);
            return data ? JSON.parse(data) : [];
        } catch {
            return [];
        }
    },

    _write(entries) {
        try {
            localStorage.setItem(this.KEY, JSON.stringify(entries));
        } catch { /* storage full or unavailable */ }
    },

    getAll() {
        return this._read();
    },

    add(entry) {
        const entries = this._read();
        const record = {
            id: crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2),
            filename: entry.filename || 'unknown.xlsx',
            rowCount: entry.rowCount || 0,
            status: entry.status || 'running', // running | success | error
            mode: entry.mode || 'download',
            email: entry.email || null,
            timestamp: new Date().toISOString(),
            durationMs: null
        };
        entries.unshift(record);
        if (entries.length > this.MAX_ENTRIES) {
            entries.length = this.MAX_ENTRIES;
        }
        this._write(entries);
        return record;
    },

    update(id, updates) {
        const entries = this._read();
        const idx = entries.findIndex(e => e.id === id);
        if (idx !== -1) {
            Object.assign(entries[idx], updates);
            this._write(entries);
        }
    },

    clearAll() {
        this._write([]);
    },

    // Theme persistence
    getTheme() {
        try {
            return localStorage.getItem('license-analyzer-theme') || 'light';
        } catch {
            return 'light';
        }
    },

    setTheme(theme) {
        try {
            localStorage.setItem('license-analyzer-theme', theme);
        } catch { /* ignore */ }
    }
};
