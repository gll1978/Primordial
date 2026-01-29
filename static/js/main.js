/**
 * Main entry point - initialize all components
 */

document.addEventListener('DOMContentLoaded', async () => {
    console.log('PRIMORDIAL Web UI starting...');

    try {
        // Initialize UI bindings
        UI.init();

        // Initialize canvas renderer
        WorldCanvas.init();

        // Initialize charts
        PopulationChart.init();
        BrainChart.init();
        SpeciesChart.init();

        // Load initial settings from server
        try {
            await API.getSettings();
            await API.getState();
        } catch (error) {
            console.warn('Could not fetch initial state from server:', error);
        }

        // Connect WebSocket
        WebSocketClient.connect();

        console.log('PRIMORDIAL Web UI initialized');
    } catch (error) {
        console.error('Failed to initialize PRIMORDIAL Web UI:', error);
    }
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    WebSocketClient.disconnect();
});
