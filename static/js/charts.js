/**
 * Simple charts for simulation history
 */

const PopulationChart = {
    canvas: null,
    ctx: null,

    /**
     * Initialize the chart
     */
    init() {
        this.canvas = document.getElementById('population-chart');
        this.ctx = this.canvas.getContext('2d');

        // Set actual size
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = 100;

        // Subscribe to snapshot updates
        AppState.subscribe('snapshotUpdate', () => {
            this.render();
        });

        // Subscribe to history clear
        AppState.subscribe('historyCleared', () => {
            this.render();
        });
    },

    /**
     * Render the chart
     */
    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const history = AppState.populationHistory;
        const padding = 5;

        // Clear canvas
        ctx.fillStyle = '#0f3460';
        ctx.fillRect(0, 0, width, height);

        if (history.length < 2) {
            ctx.fillStyle = '#a0a0a0';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for data...', width / 2, height / 2);
            return;
        }

        // Calculate scale
        const maxPop = Math.max(...history.map(h => h.population), 1);
        const minPop = 0;
        const popRange = maxPop - minPop;

        const xStep = (width - padding * 2) / (history.length - 1);

        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;

        for (let i = 0; i <= 4; i++) {
            const y = padding + (height - padding * 2) * (i / 4);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }

        // Draw population line
        ctx.beginPath();
        ctx.strokeStyle = '#4caf50';
        ctx.lineWidth = 2;

        for (let i = 0; i < history.length; i++) {
            const x = padding + i * xStep;
            const normalizedPop = (history[i].population - minPop) / popRange;
            const y = height - padding - normalizedPop * (height - padding * 2);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw current value
        if (history.length > 0) {
            const current = history[history.length - 1].population;
            ctx.fillStyle = '#e6e6e6';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`Pop: ${current}`, width - padding, padding + 12);
            ctx.fillText(`Max: ${maxPop}`, width - padding, padding + 24);
        }
    }
};

/**
 * Brain complexity chart
 */
const BrainChart = {
    canvas: null,
    ctx: null,

    /**
     * Initialize the chart
     */
    init() {
        this.canvas = document.getElementById('brain-chart');
        this.ctx = this.canvas.getContext('2d');

        // Set actual size
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = 100;

        // Subscribe to snapshot updates
        AppState.subscribe('snapshotUpdate', () => {
            this.render();
        });

        // Subscribe to history clear
        AppState.subscribe('historyCleared', () => {
            this.render();
        });
    },

    /**
     * Render the chart
     */
    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const history = AppState.brainHistory;
        const padding = 5;

        // Clear canvas
        ctx.fillStyle = '#0f3460';
        ctx.fillRect(0, 0, width, height);

        if (history.length < 2) {
            ctx.fillStyle = '#a0a0a0';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for data...', width / 2, height / 2);
            return;
        }

        // Calculate scale
        const maxBrain = Math.max(...history.map(h => h.brain_mean), 1);
        const minBrain = Math.min(...history.map(h => h.brain_mean), 0);
        const brainRange = maxBrain - minBrain || 1;

        const xStep = (width - padding * 2) / (history.length - 1);

        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;

        for (let i = 0; i <= 4; i++) {
            const y = padding + (height - padding * 2) * (i / 4);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }

        // Draw brain complexity line (orange)
        ctx.beginPath();
        ctx.strokeStyle = '#ffa500';
        ctx.lineWidth = 2;

        for (let i = 0; i < history.length; i++) {
            const x = padding + i * xStep;
            const normalized = (history[i].brain_mean - minBrain) / brainRange;
            const y = height - padding - normalized * (height - padding * 2);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw current value
        if (history.length > 0) {
            const current = history[history.length - 1].brain_mean;
            ctx.fillStyle = '#e6e6e6';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`Brain: ${current.toFixed(2)}`, width - padding, padding + 12);
            ctx.fillText(`Max: ${maxBrain.toFixed(2)}`, width - padding, padding + 24);
        }
    }
};

// Make globally available
window.PopulationChart = PopulationChart;
window.BrainChart = BrainChart;
