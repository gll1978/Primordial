//! Simulation thread that runs independently from the GUI.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::World;

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
}

impl Drop for SimulationHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Main simulation loop running in separate thread
fn run_simulation(config: Config, command_rx: Receiver<SimCommand>, snapshot_tx: Sender<WorldSnapshot>) {
    let mut current_config = config.clone();
    let mut world = World::new(current_config.clone());
    let mut state = SimState::Paused;
    let mut speed = 1.0f32;
    let mut selected_id: Option<u64> = None;
    let mut max_steps: u64 = 0; // 0 = unlimited

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
                    world = World::new(current_config.clone());
                    selected_id = None;
                    let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                }
                SimCommand::ResetWithSettings(settings) => {
                    max_steps = settings.max_steps;
                    settings.apply_to_config(&mut current_config);
                    world = World::new(current_config.clone());
                    selected_id = None;
                    state = SimState::Paused;
                    let _ = snapshot_tx.send(WorldSnapshot::from_world(&world, selected_id));
                }
                SimCommand::Shutdown => {
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
            let step_duration = Duration::from_micros((base_step_duration.as_micros() as f32 / speed) as u64);

            if last_step.elapsed() >= step_duration {
                world.step();
                last_step = Instant::now();
                steps_since_snapshot += 1;

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
