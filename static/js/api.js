/**
 * REST API client
 */

const API = {
    baseUrl: '',

    /**
     * Send a POST request
     */
    async post(endpoint, data = {}) {
        try {
            const response = await fetch(this.baseUrl + endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return response;
        } catch (error) {
            console.error(`API POST ${endpoint} failed:`, error);
            throw error;
        }
    },

    /**
     * Send a GET request
     */
    async get(endpoint) {
        try {
            const response = await fetch(this.baseUrl + endpoint);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return response.json();
        } catch (error) {
            console.error(`API GET ${endpoint} failed:`, error);
            throw error;
        }
    },

    // --- Simulation Control ---

    async pause() {
        await this.post('/api/sim/pause');
        AppState.setSimState('Paused');
    },

    async resume() {
        await this.post('/api/sim/resume');
        AppState.setSimState('Running');
    },

    async step() {
        await this.post('/api/sim/step');
    },

    async reset() {
        await this.post('/api/sim/reset');
        AppState.setSimState('Paused');
        AppState.clearHistory();
    },

    async setSpeed(speed) {
        await this.post('/api/sim/speed', { speed });
    },

    async selectOrganism(id) {
        await this.post('/api/sim/select', { id });
        AppState.setSelectedId(id);
    },

    // --- Settings ---

    async getSettings() {
        const settings = await this.get('/api/settings');
        AppState.setSettings(settings);
        return settings;
    },

    async updateSettings(settings) {
        await this.post('/api/settings', settings);
        AppState.setSettings(settings);
        AppState.setSimState('Paused');
        AppState.clearHistory();
    },

    // --- State ---

    async getState() {
        const { state } = await this.get('/api/state');
        AppState.setSimState(state);
        return state;
    }
};

// Make globally available
window.API = API;
