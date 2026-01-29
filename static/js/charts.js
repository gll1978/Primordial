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
        this.canvas.height = 120;

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
        const leftPadding = 45;  // Space for Y axis labels
        const rightPadding = 10;
        const topPadding = 10;
        const bottomPadding = 25; // Space for X axis labels

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

        const chartWidth = width - leftPadding - rightPadding;
        const chartHeight = height - topPadding - bottomPadding;
        const xStep = chartWidth / (history.length - 1);

        // Draw grid lines and Y axis labels
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#a0a0a0';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';

        for (let i = 0; i <= 4; i++) {
            const y = topPadding + chartHeight * (i / 4);
            const value = Math.round(maxPop - (maxPop * i / 4));

            // Grid line
            ctx.beginPath();
            ctx.moveTo(leftPadding, y);
            ctx.lineTo(width - rightPadding, y);
            ctx.stroke();

            // Y axis label
            ctx.fillText(this.formatNumber(value), leftPadding - 5, y + 3);
        }

        // Draw X axis labels (time)
        ctx.textAlign = 'center';
        const timeStart = history[0].time;
        const timeEnd = history[history.length - 1].time;

        ctx.fillText(this.formatNumber(timeStart), leftPadding, height - 5);
        ctx.fillText(this.formatNumber(timeEnd), width - rightPadding, height - 5);
        ctx.fillText('Time', width / 2, height - 5);

        // Draw Y axis title
        ctx.save();
        ctx.translate(12, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('Population', 0, 0);
        ctx.restore();

        // Draw population line
        ctx.beginPath();
        ctx.strokeStyle = '#4caf50';
        ctx.lineWidth = 2;

        for (let i = 0; i < history.length; i++) {
            const x = leftPadding + i * xStep;
            const normalizedPop = (history[i].population - minPop) / popRange;
            const y = topPadding + chartHeight - normalizedPop * chartHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw current value badge
        if (history.length > 0) {
            const current = history[history.length - 1].population;
            ctx.fillStyle = '#4caf50';
            ctx.fillRect(width - rightPadding - 55, topPadding, 55, 18);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(this.formatNumber(current), width - rightPadding - 27, topPadding + 13);
        }
    },

    formatNumber(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return n.toString();
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
        this.canvas.height = 120;

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
        const leftPadding = 45;
        const rightPadding = 10;
        const topPadding = 10;
        const bottomPadding = 25;

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
        const minBrain = 0;
        const brainRange = maxBrain - minBrain || 1;

        const chartWidth = width - leftPadding - rightPadding;
        const chartHeight = height - topPadding - bottomPadding;
        const xStep = chartWidth / (history.length - 1);

        // Draw grid lines and Y axis labels
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#a0a0a0';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';

        for (let i = 0; i <= 4; i++) {
            const y = topPadding + chartHeight * (i / 4);
            const value = maxBrain - (maxBrain * i / 4);

            // Grid line
            ctx.beginPath();
            ctx.moveTo(leftPadding, y);
            ctx.lineTo(width - rightPadding, y);
            ctx.stroke();

            // Y axis label
            ctx.fillText(value.toFixed(1), leftPadding - 5, y + 3);
        }

        // Draw X axis labels (time)
        ctx.textAlign = 'center';
        const timeStart = history[0].time;
        const timeEnd = history[history.length - 1].time;

        ctx.fillText(PopulationChart.formatNumber(timeStart), leftPadding, height - 5);
        ctx.fillText(PopulationChart.formatNumber(timeEnd), width - rightPadding, height - 5);
        ctx.fillText('Time', width / 2, height - 5);

        // Draw Y axis title
        ctx.save();
        ctx.translate(12, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('Brain Layers', 0, 0);
        ctx.restore();

        // Draw brain complexity line (orange)
        ctx.beginPath();
        ctx.strokeStyle = '#ffa500';
        ctx.lineWidth = 2;

        for (let i = 0; i < history.length; i++) {
            const x = leftPadding + i * xStep;
            const normalized = (history[i].brain_mean - minBrain) / brainRange;
            const y = topPadding + chartHeight - normalized * chartHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();

        // Draw current value badge
        if (history.length > 0) {
            const current = history[history.length - 1].brain_mean;
            ctx.fillStyle = '#ffa500';
            ctx.fillRect(width - rightPadding - 55, topPadding, 55, 18);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(current.toFixed(2), width - rightPadding - 27, topPadding + 13);
        }
    }
};

/**
 * Species distribution pie chart
 */
const SpeciesChart = {
    canvas: null,
    ctx: null,
    data: { predators: 0, herbivores: 0, aquatic: 0 },

    /**
     * Initialize the chart
     */
    init() {
        this.canvas = document.getElementById('species-chart');
        if (!this.canvas) {
            console.warn('Species chart canvas not found');
            return;
        }

        this.ctx = this.canvas.getContext('2d');

        // Subscribe to snapshot updates
        AppState.subscribe('snapshotUpdate', (snapshot) => {
            if (snapshot && snapshot.organisms) {
                this.updateData(snapshot.organisms);
                this.render();
            }
        });

        // Subscribe to history clear
        AppState.subscribe('historyCleared', () => {
            this.data = { predators: 0, herbivores: 0, aquatic: 0 };
            this.render();
        });

        // Delay initial render to ensure DOM is ready
        setTimeout(() => this.render(), 100);
    },

    /**
     * Update data from organisms
     */
    updateData(organisms) {
        if (!organisms || organisms.length === 0) {
            this.data = { predators: 0, herbivores: 0, aquatic: 0 };
            return;
        }

        let predators = 0;
        let aquatic = 0;
        let herbivores = 0;

        for (const org of organisms) {
            if (org.is_predator) {
                predators++;
            } else if (org.is_aquatic) {
                aquatic++;
            } else {
                herbivores++;
            }
        }

        this.data = { predators, herbivores, aquatic };
    },

    /**
     * Render the pie chart
     */
    render() {
        if (!this.canvas || !this.ctx) return;

        // Update canvas size dynamically - get parent width or use minimum
        let containerWidth = this.canvas.parentElement?.clientWidth ||
                            this.canvas.offsetWidth ||
                            250;
        // Ensure minimum width
        containerWidth = Math.max(containerWidth, 200);

        this.canvas.width = containerWidth;
        this.canvas.height = 150;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        ctx.fillStyle = '#0f3460';
        ctx.fillRect(0, 0, width, height);

        const total = this.data.predators + this.data.herbivores + this.data.aquatic;

        if (total === 0) {
            ctx.fillStyle = '#a0a0a0';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No organisms', width / 2, height / 2);
            return;
        }

        const centerX = width / 3;
        const centerY = height / 2;
        const radius = Math.min(centerX, centerY) - 15;

        // Calculate percentages
        const predatorPct = (this.data.predators / total * 100).toFixed(1);
        const herbivorePct = (this.data.herbivores / total * 100).toFixed(1);
        const aquaticPct = (this.data.aquatic / total * 100).toFixed(1);

        // Colors
        const colors = {
            predators: '#e94560',
            herbivores: '#4caf50',
            aquatic: '#00bcd4'
        };

        // Draw pie slices
        let startAngle = -Math.PI / 2; // Start from top

        const slices = [
            { key: 'herbivores', value: this.data.herbivores, color: colors.herbivores },
            { key: 'predators', value: this.data.predators, color: colors.predators },
            { key: 'aquatic', value: this.data.aquatic, color: colors.aquatic }
        ];

        for (const slice of slices) {
            if (slice.value === 0) continue;

            const sliceAngle = (slice.value / total) * Math.PI * 2;
            const endAngle = startAngle + sliceAngle;

            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, startAngle, endAngle);
            ctx.closePath();
            ctx.fillStyle = slice.color;
            ctx.fill();

            // Draw white border between slices
            ctx.strokeStyle = '#0f3460';
            ctx.lineWidth = 2;
            ctx.stroke();

            startAngle = endAngle;
        }

        // Draw center hole (donut style)
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 0.5, 0, Math.PI * 2);
        ctx.fillStyle = '#0f3460';
        ctx.fill();

        // Draw total in center
        ctx.fillStyle = '#e6e6e6';
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(total.toString(), centerX, centerY - 6);
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#a0a0a0';
        ctx.fillText('Total', centerX, centerY + 8);

        // Draw legend
        const legendX = width / 2 + 20;
        const legendY = 25;
        const legendSpacing = 35;

        const legendItems = [
            { label: 'Herbivores', value: this.data.herbivores, pct: herbivorePct, color: colors.herbivores },
            { label: 'Predators', value: this.data.predators, pct: predatorPct, color: colors.predators },
            { label: 'Aquatic', value: this.data.aquatic, pct: aquaticPct, color: colors.aquatic }
        ];

        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';

        legendItems.forEach((item, i) => {
            const y = legendY + i * legendSpacing;

            // Color box
            ctx.fillStyle = item.color;
            ctx.fillRect(legendX, y - 6, 12, 12);

            // Label
            ctx.fillStyle = '#e6e6e6';
            ctx.font = '11px sans-serif';
            ctx.fillText(item.label, legendX + 18, y);

            // Value and percentage
            ctx.fillStyle = '#a0a0a0';
            ctx.font = '10px sans-serif';
            ctx.fillText(`${item.value} (${item.pct}%)`, legendX + 18, y + 14);
        });
    }
};

// Make globally available
window.PopulationChart = PopulationChart;
window.BrainChart = BrainChart;
window.SpeciesChart = SpeciesChart;
