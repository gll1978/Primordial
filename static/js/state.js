/**
 * State management with pub/sub pattern
 */

const AppState = {
    // Current simulation state
    simState: 'Paused',  // 'Running', 'Paused', 'Stopped'

    // Latest snapshot from server
    snapshot: null,

    // Selected organism ID
    selectedId: null,

    // Connection status
    connected: false,

    // Current settings
    settings: {
        max_population: 5000,
        max_steps: 0,
        initial_population: 200,
        grid_size: 80,
        mutation_rate: 0.05,
        mutation_strength: 0.3,
        food_regen_rate: 0.3,
        reproduction_threshold: 50.0,
        predation_enabled: true,
        seasons_enabled: true,
        terrain_enabled: true,
        learning_enabled: false,
        learning_rate: 0.001,
        diversity_enabled: false,
        database_enabled: false,
        cognitive_gate_enabled: false,
        food_patches_enabled: false,
        enhanced_senses: false,
        n_inputs: 75
    },

    // Canvas options
    showFood: true,
    showGrid: false,
    zoom: 1.0,

    // Population history for chart
    populationHistory: [],
    // Brain complexity history for chart
    brainHistory: [],
    maxHistory: 200,

    // Subscribers
    _subscribers: {},

    /**
     * Subscribe to state changes
     */
    subscribe(event, callback) {
        if (!this._subscribers[event]) {
            this._subscribers[event] = [];
        }
        this._subscribers[event].push(callback);
        return () => {
            this._subscribers[event] = this._subscribers[event].filter(cb => cb !== callback);
        };
    },

    /**
     * Notify subscribers of state change
     */
    notify(event, data) {
        if (this._subscribers[event]) {
            this._subscribers[event].forEach(cb => cb(data));
        }
    },

    /**
     * Update simulation state
     */
    setSimState(state) {
        this.simState = state;
        this.notify('simStateChange', state);
    },

    /**
     * Update snapshot
     */
    setSnapshot(snapshot) {
        this.snapshot = snapshot;

        // Update history for charts
        if (snapshot && snapshot.stats) {
            this.populationHistory.push({
                time: snapshot.time,
                population: snapshot.stats.population
            });

            this.brainHistory.push({
                time: snapshot.time,
                brain_mean: snapshot.stats.brain_mean
            });

            // Trim history
            if (this.populationHistory.length > this.maxHistory) {
                this.populationHistory.shift();
            }
            if (this.brainHistory.length > this.maxHistory) {
                this.brainHistory.shift();
            }
        }

        this.notify('snapshotUpdate', snapshot);
    },

    /**
     * Update connection status
     */
    setConnected(connected) {
        this.connected = connected;
        this.notify('connectionChange', connected);
    },

    /**
     * Update selected organism
     */
    setSelectedId(id) {
        this.selectedId = id;
        this.notify('selectionChange', id);
    },

    /**
     * Update settings
     */
    setSettings(settings) {
        this.settings = { ...this.settings, ...settings };
        this.notify('settingsChange', this.settings);
    },

    /**
     * Clear history (on reset)
     */
    clearHistory() {
        this.populationHistory = [];
        this.brainHistory = [];
        this.notify('historyCleared');
    }
};

// Make globally available
window.AppState = AppState;
