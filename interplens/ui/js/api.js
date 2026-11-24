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
    },

    async getAttentionHeads(sessionId, layer = 0, head = 0, threshold = 0.02) {
        const url = `/api/analysis/attention?session_id=${sessionId}&layer=${layer}&head=${head}&threshold=${threshold}`;
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Attention heads fetch failed');
        }
        return await res.json();
    },

    async getNeuronActivations(sessionId, layer = 0, position = null, topK = 10, neuronIdx = null) {
        let url = `/api/analysis/neurons?session_id=${sessionId}&layer=${layer}&top_k=${topK}`;
        if (position !== null) url += `&position=${position}`;
        if (neuronIdx !== null) url += `&neuron_idx=${neuronIdx}`;
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Neuron activations fetch failed');
        }
        return await res.json();
    },

    async getTokenAttribution(sessionId, position = null, method = 'attention_rollout') {
        let url = `/api/analysis/attribution?session_id=${sessionId}&method=${method}`;
        if (position !== null) url += `&position=${position}`;
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Token attribution fetch failed');
        }
        return await res.json();
    },

    async runCausalPatching(cleanPrompt, corruptPrompt, targetToken = null) {
        const res = await fetch('/api/analysis/causal-patching', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                clean_prompt: cleanPrompt,
                corrupt_prompt: corruptPrompt,
                target_token: targetToken,
            })
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Causal patching execution failed');
        }
        return await res.json();
    },

    async getInductionHeads(sequenceLength = 20, threshold = 0.15, topK = 10) {
        const url = `/api/analysis/induction-heads?sequence_length=${sequenceLength}&threshold=${threshold}&top_k=${topK}`;
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Induction head detection sweep failed');
        }
        return await res.json();
    },

    async getDirectLogitAttribution(sessionId, position = null, targetToken = null, foilToken = null) {
        let url = `/api/analysis/dla?session_id=${encodeURIComponent(sessionId)}`;
        if (position !== null) url += `&position=${position}`;
        if (targetToken) url += `&target_token=${encodeURIComponent(targetToken)}`;
        if (foilToken) url += `&foil_token=${encodeURIComponent(foilToken)}`;
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Direct Logit Attribution failed');
        }
        return await res.json();
    },

    async runAttributionPatching(cleanPrompt, corruptPrompt, targetToken = null, foilToken = null) {
        const res = await fetch('/api/analysis/attribution-patching', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                clean_prompt: cleanPrompt,
                corrupt_prompt: corruptPrompt,
                target_token: targetToken,
                foil_token: foilToken,
            })
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Attribution Patching failed');
        }
        return await res.json();
    },

    async computeSteeringVector(positivePrompts, negativePrompts, targetLayer = 4, name = 'Contrastive Vector', normalize = false) {
        const res = await fetch('/api/analysis/steering/compute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                positive_prompts: positivePrompts,
                negative_prompts: negativePrompts,
                target_layer: targetLayer,
                name,
                normalize,
            })
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Steering vector computation failed');
        }
        return await res.json();
    },

    async generateSteeredComparison(prompt, vectorId, targetLayer = null, multiplier = 2.0, maxNewTokens = 20, temperature = 0.0) {
        const res = await fetch('/api/analysis/steering/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                vector_id: vectorId,
                target_layer: targetLayer,
                multiplier,
                max_new_tokens: maxNewTokens,
                temperature,
            })
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Steered generation failed');
        }
        return await res.json();
    }
};

window.API = API;
