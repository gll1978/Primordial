//! Shared application state for the web server.

use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};

use crate::config::Config;
use crate::shared::{SimCommand, SimSettings, SimState, SimulationHandle, WorldSnapshot};

/// Maximum number of subscribers that can receive snapshots
const BROADCAST_CAPACITY: usize = 16;

/// Application state shared between all handlers
pub struct AppState {
    /// Simulation handle (protected by mutex for exclusive access)
    pub sim: Mutex<SimulationHandle>,
    /// Broadcast channel for snapshots to WebSocket clients
    pub snapshot_tx: broadcast::Sender<Arc<WorldSnapshot>>,
    /// Current simulation state
    pub state: RwLock<SimState>,
    /// Current settings
    pub settings: RwLock<SimSettings>,
    /// Current config (reserved for future use)
    #[allow(dead_code)]
    pub config: RwLock<Config>,
    /// Currently selected organism ID
    pub selected_id: RwLock<Option<u64>>,
}

impl AppState {
    /// Create new application state
    pub fn new(config: Config) -> Self {
        let settings = SimSettings::from_config(&config);
        let sim = SimulationHandle::spawn(config.clone());
        let (snapshot_tx, _) = broadcast::channel(BROADCAST_CAPACITY);

        Self {
            sim: Mutex::new(sim),
            snapshot_tx,
            state: RwLock::new(SimState::Paused),
            settings: RwLock::new(settings),
            config: RwLock::new(config),
            selected_id: RwLock::new(None),
        }
    }

    /// Send a command to the simulation
    pub async fn send_command(&self, command: SimCommand) {
        // Update local state based on command
        match &command {
            SimCommand::Pause => {
                *self.state.write().await = SimState::Paused;
            }
            SimCommand::Resume => {
                *self.state.write().await = SimState::Running;
            }
            SimCommand::Shutdown => {
                *self.state.write().await = SimState::Stopped;
            }
            SimCommand::Reset | SimCommand::ResetWithSettings(_) => {
                *self.state.write().await = SimState::Paused;
            }
            SimCommand::SelectOrganism(id) => {
                *self.selected_id.write().await = *id;
            }
            _ => {}
        }

        // Send to simulation thread
        let mut sim = self.sim.lock().await;
        sim.send(command);
    }

    /// Try to get the latest snapshot from the simulation
    pub async fn try_recv_snapshot(&self) -> Option<WorldSnapshot> {
        let sim = self.sim.lock().await;
        sim.try_recv_snapshot()
    }

    /// Update settings
    pub async fn update_settings(&self, settings: SimSettings) {
        *self.settings.write().await = settings;
    }

    /// Get current settings
    pub async fn get_settings(&self) -> SimSettings {
        self.settings.read().await.clone()
    }

    /// Get current state
    pub async fn get_state(&self) -> SimState {
        *self.state.read().await
    }

    /// Broadcast a snapshot to all WebSocket clients
    pub fn broadcast_snapshot(&self, snapshot: Arc<WorldSnapshot>) {
        // Ignore send errors (no receivers is ok)
        let _ = self.snapshot_tx.send(snapshot);
    }

    /// Subscribe to snapshot broadcasts
    pub fn subscribe_snapshots(&self) -> broadcast::Receiver<Arc<WorldSnapshot>> {
        self.snapshot_tx.subscribe()
    }
}

/// Spawns a task that relays snapshots from the simulation thread to the broadcast channel
pub fn spawn_snapshot_relay(state: Arc<AppState>) {
    tokio::spawn(async move {
        let poll_interval = std::time::Duration::from_millis(33); // ~30fps

        loop {
            // Try to get a new snapshot
            if let Some(snapshot) = state.try_recv_snapshot().await {
                let snapshot = Arc::new(snapshot);
                state.broadcast_snapshot(snapshot);
            }

            tokio::time::sleep(poll_interval).await;
        }
    });
}
