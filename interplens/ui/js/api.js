// InterpLens Studio Centralized REST API Client

var API = window.API || {
    async getSystemHealth() {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
        return await res.json();
    },

    async runModelForwardPass(prompt) {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Forward pass failed');
        }
        return await res.json();
    },

    async getLogitLensMatrix(sessionId, topK = 5, applyLn = true, position = null) {
        let url = `/api/analysis/logit-lens?session_id=${sessionId}&top_k=${topK}&apply_ln=${applyLn}`;
        if (position !== null) url += `&position=${position}`;
        
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Logit Lens extraction failed');
        }
        return await res.json();
    },

    async getGpuProfiler(sessionId = '') {
        const url = `/api/hardware/gpu-profiler${sessionId ? '?session_id=' + sessionId : ''}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`GPU Profiler fetch failed: ${res.statusText}`);
        return await res.json();
    },

    async deleteSession(sessionId) {
        const res = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(`Session eviction failed: ${res.statusText}`);
        return await res.json();
    },

    async getResidualStreamMetrics(sessionId, position = null) {
        let url = `/api/analysis/residual-stream?session_id=${sessionId}`;
        if (position !== null) url += `&position=${position}`;
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Residual stream fetch failed');
        }
        return await res.json();
    },

    async steerResidualStream(prompt, targetLayer = 0, multiplier = 1.0, steeringVector = null) {
        const res = await fetch('/api/analysis/residual-stream/steer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, target_layer: targetLayer, multiplier, steering_vector: steeringVector })
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Steering execution failed');
        }
        return await res.json();
    },

    async getModelTopology() {
        const res = await fetch('/api/model/topology');
        if (!res.ok) throw new Error(`Model topology fetch failed: ${res.statusText}`);
        return await res.json();
    }
};

window.API = API;
