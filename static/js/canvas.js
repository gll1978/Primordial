/**
 * Canvas renderer for the simulation world
 */

const WorldCanvas = {
    canvas: null,
    ctx: null,
    zoom: 1.0,
    showFood: true,
    showGrid: false,

    // Terrain colors (matching Rust)
    terrainColors: {
        0: '#90EE90',  // Plain - Light green
        1: '#228B22',  // Forest - Forest green
        2: '#8B8989',  // Mountain - Gray
        3: '#EED591',  // Desert - Sandy
        4: '#4169E1'   // Water - Royal blue
    },

    /**
     * Initialize the canvas
     */
    init() {
        this.canvas = document.getElementById('world-canvas');
        this.ctx = this.canvas.getContext('2d');

        // Handle resize
        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Handle clicks
        this.canvas.addEventListener('click', (e) => this.handleClick(e));

        // Handle keyboard
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.deselectOrganism();
            }
        });

        // Subscribe to state changes
        AppState.subscribe('snapshotUpdate', (snapshot) => {
            if (snapshot) {
                this.render(snapshot);
            }
        });
    },

    /**
     * Resize canvas to fit container
     */
    resize() {
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();

        // Account for canvas controls height
        const controlsHeight = document.getElementById('canvas-controls').offsetHeight;

        this.canvas.width = rect.width;
        this.canvas.height = rect.height - controlsHeight;

        // Re-render if we have a snapshot
        if (AppState.snapshot) {
            this.render(AppState.snapshot);
        }
    },

    /**
     * Handle click on canvas
     */
    handleClick(event) {
        const snapshot = AppState.snapshot;
        if (!snapshot) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const { cellSize, offsetX, offsetY } = this.calculateLayout(snapshot);

        const gridX = Math.floor((x - offsetX) / cellSize);
        const gridY = Math.floor((y - offsetY) / cellSize);

        if (gridX < 0 || gridY < 0 || gridX >= snapshot.grid_size || gridY >= snapshot.grid_size) {
            return;
        }

        // Find organism at this position
        const organism = snapshot.organisms.find(o =>
            o.x === gridX && o.y === gridY
        );

        if (organism) {
            API.selectOrganism(organism.id);
        }
    },

    /**
     * Deselect current organism
     */
    deselectOrganism() {
        API.selectOrganism(null);
    },

    /**
     * Calculate layout parameters
     */
    calculateLayout(snapshot) {
        const gridSize = snapshot.grid_size;
        const availableWidth = this.canvas.width;
        const availableHeight = this.canvas.height;

        const baseCellSize = Math.min(
            availableWidth / gridSize,
            availableHeight / gridSize
        );
        const cellSize = baseCellSize * this.zoom;
        const gridPixels = gridSize * cellSize;

        const offsetX = Math.max(0, (availableWidth - gridPixels) / 2);
        const offsetY = Math.max(0, (availableHeight - gridPixels) / 2);

        return { cellSize, gridPixels, offsetX, offsetY };
    },

    /**
     * Render the world
     */
    render(snapshot) {
        const ctx = this.ctx;
        const { cellSize, gridPixels, offsetX, offsetY } = this.calculateLayout(snapshot);

        // Clear canvas
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw terrain
        for (let y = 0; y < snapshot.grid_size; y++) {
            for (let x = 0; x < snapshot.grid_size; x++) {
                const idx = y * snapshot.grid_size + x;
                const terrain = snapshot.terrain_grid[idx];

                ctx.fillStyle = this.terrainColors[terrain] || this.terrainColors[0];
                ctx.fillRect(
                    offsetX + x * cellSize,
                    offsetY + y * cellSize,
                    cellSize,
                    cellSize
                );
            }
        }

        // Draw food overlay
        if (this.showFood) {
            for (let y = 0; y < snapshot.grid_size; y++) {
                for (let x = 0; x < snapshot.grid_size; x++) {
                    const idx = y * snapshot.grid_size + x;
                    const food = snapshot.food_grid[idx];

                    if (food > 0.1) {
                        const alpha = Math.min(food / 50 * 0.7, 0.7);
                        ctx.fillStyle = `rgba(0, 200, 0, ${alpha})`;
                        ctx.fillRect(
                            offsetX + x * cellSize,
                            offsetY + y * cellSize,
                            cellSize,
                            cellSize
                        );
                    }
                }
            }
        }

        // Draw grid lines
        if (this.showGrid) {
            ctx.strokeStyle = 'rgba(100, 100, 100, 0.3)';
            ctx.lineWidth = 1;

            for (let i = 0; i <= snapshot.grid_size; i++) {
                const pos = i * cellSize;

                ctx.beginPath();
                ctx.moveTo(offsetX + pos, offsetY);
                ctx.lineTo(offsetX + pos, offsetY + gridPixels);
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(offsetX, offsetY + pos);
                ctx.lineTo(offsetX + gridPixels, offsetY + pos);
                ctx.stroke();
            }
        }

        // Draw organisms
        for (const org of snapshot.organisms) {
            const centerX = offsetX + (org.x + 0.5) * cellSize;
            const centerY = offsetY + (org.y + 0.5) * cellSize;
            const radius = Math.max(cellSize * 0.4 * Math.sqrt(org.size), 2);

            // Get organism color
            const color = this.getOrganismColor(org);

            // Draw organism
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // Draw selection indicator
            if (AppState.selectedId === org.id) {
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius + 2, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
    },

    /**
     * Get color for an organism
     */
    getOrganismColor(org) {
        const intensity = Math.max(0.3, Math.min(org.energy / 100, 1.0));

        if (org.is_predator) {
            // Red for predators
            return `rgb(${Math.floor(255 * intensity)}, ${Math.floor(50 * intensity)}, ${Math.floor(50 * intensity)})`;
        } else if (org.is_aquatic) {
            // Cyan for aquatic
            return `rgb(${Math.floor(50 * intensity)}, ${Math.floor(200 * intensity)}, ${Math.floor(255 * intensity)})`;
        } else {
            // Green for normal
            return `rgb(${Math.floor(50 * intensity)}, ${Math.floor(200 * intensity)}, ${Math.floor(50 * intensity)})`;
        }
    },

    /**
     * Update zoom level
     */
    setZoom(zoom) {
        this.zoom = zoom;
        if (AppState.snapshot) {
            this.render(AppState.snapshot);
        }
    },

    /**
     * Toggle food display
     */
    setShowFood(show) {
        this.showFood = show;
        if (AppState.snapshot) {
            this.render(AppState.snapshot);
        }
    },

    /**
     * Toggle grid display
     */
    setShowGrid(show) {
        this.showGrid = show;
        if (AppState.snapshot) {
            this.render(AppState.snapshot);
        }
    }
};

// Make globally available
window.WorldCanvas = WorldCanvas;
