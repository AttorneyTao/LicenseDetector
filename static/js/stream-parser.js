/**
 * Stream log line parser and progress extractor
 */
const StreamParser = {
    /**
     * Parse a single log line and determine its type
     */
    parseLogLine(line) {
        const trimmed = line.trim();
        if (!trimmed) return null;

        let type = 'unknown';

        if (/\[DOWNLOAD_PROGRESS\]/i.test(trimmed)) {
            // 归档下载进度：单独归类为 progress，不参与总体进度条计算
            type = 'progress';
        } else if (/\[START\]/i.test(trimmed) || /开始处理|开始分析|API启动/.test(trimmed)) {
            type = 'start';
        } else if (/\[SUCCESS\]/i.test(trimmed) || /成功|完成/.test(trimmed)) {
            type = 'success';
        } else if (/\[ERROR\]/i.test(trimmed) || /ERROR/i.test(trimmed) || /失败|错误|异常/.test(trimmed)) {
            type = 'error';
        } else if (/\[PROGRESS\]/i.test(trimmed) || /已完成\s*\d+\/\d+/.test(trimmed)) {
            type = 'progress';
        } else if (/\[WARNING\]/i.test(trimmed) || /WARNING/i.test(trimmed) || /警告/.test(trimmed)) {
            type = 'warning';
        } else if (/\[INFO\]/i.test(trimmed) || /INFO/i.test(trimmed)) {
            type = 'info';
        } else {
            type = 'info';
        }

        return { type, message: trimmed, raw: line };
    },

    /**
     * Extract progress info from a log line
     * Returns { current, total, percent } or null
     */
    extractProgress(line) {
        // 归档下载进度是单个文件的下载百分比，不代表整体分析进度
        if (/\[DOWNLOAD_PROGRESS\]/i.test(line)) return null;

        // Match: [PROGRESS] 已完成 X/Y (Z.Z%)
        const match = line.match(/已完成\s*(\d+)\s*\/\s*(\d+)\s*\((\d+\.?\d*)%\)/);
        if (match) {
            return {
                current: parseInt(match[1], 10),
                total: parseInt(match[2], 10),
                percent: parseFloat(match[3])
            };
        }

        // Match generic: X/Y or X of Y
        const altMatch = line.match(/(\d+)\s*\/\s*(\d+)/);
        if (altMatch && /progress|完成|处理/i.test(line)) {
            const current = parseInt(altMatch[1], 10);
            const total = parseInt(altMatch[2], 10);
            if (total > 0) {
                return {
                    current,
                    total,
                    percent: Math.round((current / total) * 1000) / 10
                };
            }
        }

        return null;
    },

    /**
     * Check if a line indicates completion
     */
    isComplete(line) {
        return /\[SUCCESS\]/i.test(line) || /分析完成|处理完成|全部完成/.test(line);
    },

    /**
     * Check if a line indicates an error/failure
     */
    isFailure(line) {
        return /处理失败|分析失败|处理请求时出错/.test(line);
    },

    /**
     * Get CSS class for a log type
     */
    getLogClass(type) {
        const classMap = {
            start: 'start',
            info: 'info',
            error: 'error',
            success: 'success',
            progress: 'progress',
            warning: 'warning',
            unknown: ''
        };
        return classMap[type] || '';
    }
};
