/**
 * UI bindings and event handlers
 */

const UI = {
    /**
     * Initialize UI bindings
     */
    init() {
        this.bindControls();
        this.bindSettings();
        this.bindCanvasControls();
        this.bindCollapsibles();
        this.subscribeToState();
    },

    /**
     * Bind control buttons
     */
    bindControls() {
        // Play/Pause
        document.getElementById('btn-play').addEventListener('click', async () => {
            if (AppState.simState === 'Running') {
                await API.pause();
            } else {
                await API.resume();
            }
        });

        // Step
        document.getElementById('btn-step').addEventListener('click', async () => {
            await API.step();
        });

        // Reset
        document.getElementById('btn-reset').addEventListener('click', async () => {
            await API.reset();
        });

        // Speed slider
        const speedSlider = document.getElementById('speed-slider');
        const speedValue = document.getElementById('speed-value');

        speedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            speedValue.textContent = speed.toFixed(1) + 'x';
        });

        speedSlider.addEventListener('change', async (e) => {
            const speed = parseFloat(e.target.value);
            await API.setSpeed(speed);
        });
    },

    /**
     * Bind settings form
     */
    bindSettings() {
        document.getElementById('btn-apply').addEventListener('click', async () => {
            const settings = {
                grid_size: parseInt(document.getElementById('grid-size').value),
                initial_population: parseInt(document.getElementById('initial-pop').value),
                max_population: parseInt(document.getElementById('max-pop').value),
                mutation_rate: parseFloat(document.getElementById('mutation-rate').value),
                food_regen_rate: parseFloat(document.getElementById('food-regen').value),
                predation_enabled: document.getElementById('predation-enabled').checked,
                seasons_enabled: document.getElementById('seasons-enabled').checked,
                terrain_enabled: document.getElementById('terrain-enabled').checked,
                // Keep other settings from current state
                ...AppState.settings,
                grid_size: parseInt(document.getElementById('grid-size').value),
                initial_population: parseInt(document.getElementById('initial-pop').value),
                max_population: parseInt(document.getElementById('max-pop').value),
                mutation_rate: parseFloat(document.getElementById('mutation-rate').value),
                food_regen_rate: parseFloat(document.getElementById('food-regen').value),
                predation_enabled: document.getElementById('predation-enabled').checked,
                seasons_enabled: document.getElementById('seasons-enabled').checked,
                terrain_enabled: document.getElementById('terrain-enabled').checked,
            };

            await API.updateSettings(settings);
        });
    },

    /**
     * Bind canvas controls
     */
    bindCanvasControls() {
        document.getElementById('show-food').addEventListener('change', (e) => {
            WorldCanvas.setShowFood(e.target.checked);
        });

        document.getElementById('show-grid').addEventListener('change', (e) => {
            WorldCanvas.setShowGrid(e.target.checked);
        });

        document.getElementById('zoom-slider').addEventListener('input', (e) => {
            WorldCanvas.setZoom(parseFloat(e.target.value));
        });
    },

    /**
     * Bind collapsible panels
     */
    bindCollapsibles() {
        document.querySelectorAll('.panel.collapsible .panel-header').forEach(header => {
            header.addEventListener('click', () => {
                const panel = header.closest('.panel');
                panel.classList.toggle('collapsed');
            });
        });
    },

    /**
     * Subscribe to state changes
     */
    subscribeToState() {
        // Connection status
        AppState.subscribe('connectionChange', (connected) => {
            const statusEl = document.getElementById('connection-status');
            if (connected) {
                statusEl.textContent = 'Connected';
                statusEl.classList.remove('disconnected');
                statusEl.classList.add('connected');
            } else {
                statusEl.textContent = 'Disconnected';
                statusEl.classList.remove('connected');
                statusEl.classList.add('disconnected');
            }
        });

        // Simulation state
        AppState.subscribe('simStateChange', (state) => {
            this.updatePlayButton(state);
            this.updateStepButton(state);
            document.getElementById('status-state').textContent = `State: ${state}`;
        });

        // Snapshot updates
        AppState.subscribe('snapshotUpdate', (snapshot) => {
            if (snapshot) {
                this.updateStats(snapshot);
                this.updateStatusBar(snapshot);
                this.updateOrganismPanel(snapshot);
            }
        });

        // Settings changes
        AppState.subscribe('settingsChange', (settings) => {
            this.updateSettingsForm(settings);
        });

        // Selection changes
        AppState.subscribe('selectionChange', (id) => {
            const panel = document.getElementById('organism-panel');
            if (id === null) {
                panel.style.display = 'none';
            }
        });
    },

    /**
     * Update play button state
     */
    updatePlayButton(state) {
        const btn = document.getElementById('btn-play');
        const icon = document.getElementById('play-icon');

        if (state === 'Running') {
            icon.innerHTML = '&#10074;&#10074;'; // Pause icon
            btn.title = 'Pause';
        } else {
            icon.innerHTML = '&#9658;'; // Play icon
            btn.title = 'Play';
        }

        btn.disabled = state === 'Stopped';
    },

    /**
     * Update step button state
     */
    updateStepButton(state) {
        const btn = document.getElementById('btn-step');
        btn.disabled = state !== 'Paused';
    },

    /**
     * Update statistics display
     */
    updateStats(snapshot) {
        document.getElementById('stat-time').textContent = snapshot.time;
        document.getElementById('stat-population').textContent = snapshot.stats.population;
        document.getElementById('stat-generation').textContent = snapshot.stats.generation_max;
        document.getElementById('stat-energy').textContent = snapshot.stats.energy_mean.toFixed(1);
        document.getElementById('stat-lineages').textContent = snapshot.stats.lineage_count;
        document.getElementById('stat-brain').textContent = snapshot.stats.brain_mean.toFixed(2);
        document.getElementById('stat-food').textContent = snapshot.stats.total_food.toFixed(0);
    },

    /**
     * Update status bar
     */
    updateStatusBar(snapshot) {
        document.getElementById('status-time').textContent = `Time: ${snapshot.time}`;
        document.getElementById('status-pop').textContent = `Pop: ${snapshot.stats.population}`;
        document.getElementById('status-gen').textContent = `Gen: ${snapshot.stats.generation_max}`;
    },

    /**
     * Update organism panel
     */
    updateOrganismPanel(snapshot) {
        const panel = document.getElementById('organism-panel');
        const details = document.getElementById('organism-details');

        if (snapshot.selected_organism) {
            const org = snapshot.selected_organism;
            panel.style.display = 'block';

            // Determine type
            let typeClass = 'normal';
            let typeName = 'Herbivore';
            if (org.is_predator) {
                typeClass = 'predator';
                typeName = 'Predator';
            } else if (org.is_aquatic) {
                typeClass = 'aquatic';
                typeName = 'Aquatic';
            }

            details.innerHTML = `
                <div class="organism-type ${typeClass}">${typeName} #${org.id}</div>
                <div class="detail-row"><span>Position:</span><span>(${org.x}, ${org.y})</span></div>
                <div class="detail-row"><span>Energy:</span><span>${org.energy.toFixed(1)}</span></div>
                <div class="detail-row"><span>Health:</span><span>${org.health.toFixed(1)}</span></div>
                <div class="detail-row"><span>Age:</span><span>${org.age}</span></div>
                <div class="detail-row"><span>Generation:</span><span>${org.generation}</span></div>
                <div class="detail-row"><span>Lineage:</span><span>${org.lineage_id}</span></div>
                <div class="detail-row"><span>Size:</span><span>${org.size.toFixed(2)}</span></div>
                <div class="detail-row"><span>Kills:</span><span>${org.kills}</span></div>
                <div class="detail-row"><span>Offspring:</span><span>${org.offspring_count}</span></div>
                <div class="detail-row"><span>Food Eaten:</span><span>${org.food_eaten}</span></div>
                <div class="detail-row"><span>Brain Layers:</span><span>${org.brain_layers.length}</span></div>
            `;
        } else if (AppState.selectedId === null) {
            panel.style.display = 'none';
        }
    },

    /**
     * Update settings form from state
     */
    updateSettingsForm(settings) {
        document.getElementById('grid-size').value = settings.grid_size;
        document.getElementById('initial-pop').value = settings.initial_population;
        document.getElementById('max-pop').value = settings.max_population;
        document.getElementById('mutation-rate').value = settings.mutation_rate;
        document.getElementById('food-regen').value = settings.food_regen_rate;
        document.getElementById('predation-enabled').checked = settings.predation_enabled;
        document.getElementById('seasons-enabled').checked = settings.seasons_enabled;
        document.getElementById('terrain-enabled').checked = settings.terrain_enabled;
    }
};

// Make globally available
window.UI = UI;
