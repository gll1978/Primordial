//! Simulation thread that runs independently from the GUI/Web UI.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::checkpoint::{Checkpoint, CheckpointManager};
use crate::config::Config;
use crate::memory_monitor::{format_bytes, MemoryAction, MemoryMonitor, MemoryThresholds};
use crate::World;

#[cfg(feature = "database")]
use crate::database::Database;

use super::commands::{SimCommand, SimState};
use super::snapshot::WorldSnapshot;

/// Handle for controlling the simulation thread
pub struct SimulationHandle {
    /// Thread handle
    thread: Option<JoinHandle<()>>,
    /// Channel to send commands to simulation
    command_tx: Sender<SimCommand>,
    /// Channel to receive snapshots from simulation
    snapshot_rx: Receiver<WorldSnapshot>,
    /// Current state
    pub state: SimState,
}

impl SimulationHandle {
    /// Spawn a new simulation thread
    pub fn spawn(config: Config) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let (snapshot_tx, snapshot_rx) = mpsc::channel();

        let thread = thread::spawn(move || {
            run_simulation(config, command_rx, snapshot_tx);
        });

        Self {
            thread: Some(thread),
            command_tx,
            snapshot_rx,
            state: SimState::Paused,
        }
    }

    /// Send a command to the simulation
    pub fn send(&mut self, command: SimCommand) {
        match &command {
            SimCommand::Pause => self.state = SimState::Paused,
            SimCommand::Resume => self.state = SimState::Running,
            SimCommand::Shutdown => self.state = SimState::Stopped,
            SimCommand::Reset | SimCommand::ResetWithSettings(_) => self.state = SimState::Paused,
            _ => {}
        }
        let _ = self.command_tx.send(command);
    }

    /// Try to receive the latest snapshot (non-blocking)
    pub fn try_recv_snapshot(&self) -> Option<WorldSnapshot> {
        let mut latest = None;
        // Drain all available snapshots, keep only the latest
        loop {
            match self.snapshot_rx.try_recv() {
                Ok(snapshot) => latest = Some(snapshot),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        latest
    }

    /// Check if simulation is running
    pub fn is_running(&self) -> bool {
        self.state == SimState::Running
    }

    /// Shutdown the simulation thread
    pub fn shutdown(&mut self) {
        self.send(SimCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    /// Save checkpoint manually
    pub fn save_checkpoint(&mut self) {
        self.send(SimCommand::SaveCheckpoint);
    }

    /// Load checkpoint from file
    pub fn load_checkpoint(&mut self, path: &str) {
        self.send(SimCommand::LoadCheckpoint(path.to_string()));
    }

    /// Set checkpoint directory
    pub fn set_checkpoint_dir(&mut self, dir: &str) {
        self.send(SimCommand::SetCheckpointDir(dir.to_string()));
    }

    /// Get default checkpoint directory
    pub fn default_checkpoint_dir() -> std::path::PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("primordial")
            .join("checkpoints")
    }

    /// Find latest checkpoint in the default directory
    pub fn find_latest_checkpoint() -> Option<String> {
        let dir = Self::default_checkpoint_dir();
        std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("checkpoint_")
            })
            .max_by_key(|e| e.file_name())
            .map(|e| e.path().to_string_lossy().to_string())
    }
}

impl Drop for SimulationHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Main simulation loop running in separate thread
fn run_simulation(
    config: Config,
    command_rx: Receiver<SimCommand>,
    snapshot_tx: Sender<WorldSnapshot>,
) {
    let mut current_config = config.clone();
    let mut world = World::new(current_config.clone());
    let mut state = SimState::Paused;
    let mut speed = 1.0f32;
    let mut selected_id: Option<u64> = None;
    let mut max_steps: u64 = 0; // 0 = unlimited

    // Initialize checkpoint manager for auto-save
    let default_checkpoint_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("primordial")
        .join("checkpoints");
    let mut checkpoint_mgr = CheckpointManager::new(
        default_checkpoint_dir.to_string_lossy().to_string(),
        current_config.logging.checkpoint_interval,
        5, // Keep last 5 checkpoints
    );
    log::info!(
        "Checkpoint auto-save enabled: dir={}, interval={}",
        checkpoint_mgr.base_dir,
        checkpoint_mgr.interval
    );

    // Initialize memory monitor for safety
    let memory_thresholds = MemoryThresholds {
        warning_percent: current_config.safety.memory_warning_percent.unwrap_or(70.0),
        critical_percent: current_config.safety.memory_critical_percent.unwrap_or(85.0),
        max_memory_mb: current_config.safety.max_memory_mb.unwrap_or(0),
    };
    let memory_monitor = MemoryMonitor::new(memory_thresholds);
    let mut last_memory_check: u64 = 0;
    const MEMORY_CHECK_INTERVAL: u64 = 100; // Check every 100 steps

    log::info!(
        "Memory monitor enabled: warning={}%, critical={}%, max={}MB, total_system={}",
        memory_thresholds.warning_percent,
        memory_thresholds.critical_percent,
        memory_thresholds.max_memory_mb,
        format_bytes(memory_monitor.total_system_memory())
    );

    // Initialize database if enabled in config
    #[cfg(feature = "database")]
    #[allow(unused_assignments)]
    let mut _db: Option<Database> = if current_config.database.enabled {
        let config_json = serde_json::to_string(&current_config).unwrap_or_default();
        match Database::new(&current_config.database.url, &config_json, None) {
            Ok(database) => {
                log::info!("Database connected: run_id = {}", database.run_id);
                world.set_db_sender(database.sender_clone());
                Some(database)
            }
            Err(e) => {
                log::error!("Database connection failed: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Timing control
    let base_step_duration = Duration::from_micros(1000); // Base: 1000 steps/s max
    let mut last_step = Instant::now();
    let mut steps_since_snapshot = 0u32;
    let snapshot_interval = 3u32; // Send snapshot every N steps

    // Send initial snapshot
    let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));

    loop {
        // Process commands (non-blocking)
        match command_rx.try_recv() {
            Ok(cmd) => match cmd {
                SimCommand::Pause => state = SimState::Paused,
                SimCommand::Resume => state = SimState::Running,
                SimCommand::Step => {
                    world.step();
                    let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                }
                SimCommand::SetSpeed(s) => speed = s.clamp(0.1, 10.0),
                SimCommand::SelectOrganism(id) => {
                    selected_id = id;
                    let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                }
                SimCommand::Reset => {
                    log::info!(
                        "Reset: using current_config with grid_size={}",
                        current_config.world.grid_size
                    );
                    world = World::new(current_config.clone());
                    log::info!("World reset: population={}", world.organisms.len());
                    // Reconnect database for new run
                    #[cfg(feature = "database")]
                    if current_config.database.enabled {
                        let config_json =
                            serde_json::to_string(&current_config).unwrap_or_default();
                        if let Ok(new_db) =
                            Database::new(&current_config.database.url, &config_json, None)
                        {
                            log::info!("Database reconnected: run_id = {}", new_db.run_id);
                            world.set_db_sender(new_db.sender_clone());
                            _db = Some(new_db);
                        }
                    }
                    selected_id = None;
                    last_memory_check = 0;
                    let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                }
                SimCommand::ResetWithSettings(settings) => {
                    log::info!(
                        "ResetWithSettings: grid_size={}, population={}",
                        settings.grid_size,
                        settings.initial_population
                    );
                    max_steps = settings.max_steps;
                    settings.apply_to_config(&mut current_config);
                    log::info!("Config applied: grid_size={}", current_config.world.grid_size);
                    world = World::new(current_config.clone());
                    log::info!("World created: population={}", world.organisms.len());
                    // Reconnect database for new run
                    #[cfg(feature = "database")]
                    if current_config.database.enabled {
                        let config_json =
                            serde_json::to_string(&current_config).unwrap_or_default();
                        if let Ok(new_db) =
                            Database::new(&current_config.database.url, &config_json, None)
                        {
                            log::info!("Database reconnected: run_id = {}", new_db.run_id);
                            world.set_db_sender(new_db.sender_clone());
                            _db = Some(new_db);
                        }
                    }
                    selected_id = None;
                    last_memory_check = 0;
                    state = SimState::Paused;
                    log::debug!(
                        "[SIM] Creating snapshot for grid_size={}...",
                        current_config.world.grid_size
                    );
                    let snapshot = WorldSnapshot::from_world(&world, selected_id);
                    log::debug!(
                        "[SIM] Snapshot created: grid_size={}, organisms={}",
                        snapshot.grid_size,
                        snapshot.organisms.len()
                    );
                    let send_result = snapshot_tx.send(snapshot);
                    log::debug!("[SIM] Snapshot sent: {:?}", send_result.is_ok());
                }
                SimCommand::SaveCheckpoint => {
                    // Manual checkpoint save
                    let checkpoint = world.create_checkpoint();
                    match checkpoint_mgr.save(&checkpoint) {
                        Ok(path) => log::info!("Checkpoint saved: {}", path),
                        Err(e) => log::error!("Checkpoint save failed: {}", e),
                    }
                }
                SimCommand::LoadCheckpoint(path) => {
                    match Checkpoint::load(&path) {
                        Ok(checkpoint) => {
                            log::info!(
                                "Loading checkpoint from {}: time={}, organisms={}",
                                path,
                                checkpoint.time,
                                checkpoint.organisms.len()
                            );
                            current_config = checkpoint.config.clone();
                            world = World::from_checkpoint(checkpoint);
                            // Reconnect database for loaded run
                            #[cfg(feature = "database")]
                            if current_config.database.enabled {
                                let config_json =
                                    serde_json::to_string(&current_config).unwrap_or_default();
                                if let Ok(new_db) =
                                    Database::new(&current_config.database.url, &config_json, None)
                                {
                                    log::info!("Database reconnected: run_id = {}", new_db.run_id);
                                    world.set_db_sender(new_db.sender_clone());
                                    _db = Some(new_db);
                                }
                            }
                            selected_id = None;
                            state = SimState::Paused;
                            let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                            log::info!("Checkpoint loaded successfully");
                        }
                        Err(e) => log::error!("Checkpoint load failed: {}", e),
                    }
                }
                SimCommand::SetCheckpointDir(dir) => {
                    checkpoint_mgr = CheckpointManager::new(
                        dir.clone(),
                        current_config.logging.checkpoint_interval,
                        5,
                    );
                    log::info!("Checkpoint directory changed to: {}", dir);
                }
                SimCommand::Shutdown => {
                    // Save final checkpoint before shutdown
                    let checkpoint = world.create_checkpoint();
                    if let Err(e) = checkpoint_mgr.save(&checkpoint) {
                        log::error!("Final checkpoint save failed: {}", e);
                    } else {
                        log::info!("Final checkpoint saved before shutdown");
                    }
                    return;
                }
            },
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                return;
            }
        }

        // Check if we reached max steps
        let reached_max_steps = max_steps > 0 && world.time >= max_steps;

        // Run simulation step if not paused and not at limit
        if state == SimState::Running && !world.is_extinct() && !reached_max_steps {
            let step_duration =
                Duration::from_micros((base_step_duration.as_micros() as f32 / speed) as u64);

            if last_step.elapsed() >= step_duration {
                world.step();
                last_step = Instant::now();
                steps_since_snapshot += 1;

                // Auto-save checkpoint periodically
                if checkpoint_mgr.should_save(world.time) {
                    let checkpoint = world.create_checkpoint();
                    match checkpoint_mgr.save(&checkpoint) {
                        Ok(path) => log::debug!("Auto-checkpoint saved: {}", path),
                        Err(e) => log::warn!("Auto-checkpoint failed: {}", e),
                    }
                }

                // Memory check periodically
                if world.time.saturating_sub(last_memory_check) >= MEMORY_CHECK_INTERVAL {
                    last_memory_check = world.time;
                    match memory_monitor.check(world.time) {
                        MemoryAction::Critical(stats) => {
                            log::error!(
                                "CRITICAL: Memory usage at {:.1}% ({}) - auto-pausing and saving checkpoint",
                                stats.percent_used,
                                format_bytes(stats.rss_bytes)
                            );
                            // Save emergency checkpoint
                            let checkpoint = world.create_checkpoint();
                            if let Err(e) = checkpoint_mgr.save(&checkpoint) {
                                log::error!("Emergency checkpoint failed: {}", e);
                            } else {
                                log::info!("Emergency checkpoint saved");
                            }
                            // Auto-pause
                            state = SimState::Paused;
                            let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                        }
                        MemoryAction::Warning(stats) => {
                            log::warn!(
                                "Memory usage high: {:.1}% ({}) - consider saving checkpoint",
                                stats.percent_used,
                                format_bytes(stats.rss_bytes)
                            );
                        }
                        MemoryAction::Ok(_) => {}
                    }
                }

                // Send snapshot periodically
                if steps_since_snapshot >= snapshot_interval {
                    let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                    steps_since_snapshot = 0;
                }

                // Auto-pause when reaching max steps
                if max_steps > 0 && world.time >= max_steps {
                    state = SimState::Paused;
                    let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                }
            }
        }

        // Small sleep to avoid busy-waiting when paused
        if state == SimState::Paused || reached_max_steps {
            thread::sleep(Duration::from_millis(16)); // ~60fps polling
        } else {
            // Yield to allow other threads
            thread::yield_now();
        }
    }
}
