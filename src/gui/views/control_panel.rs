//! Control panel for simulation playback.

use egui::Ui;

use crate::shared::sim_thread::SimulationHandle;
use crate::shared::{SimCommand, SimState};

/// Control panel for simulation playback
pub struct ControlPanel {
    /// Current speed multiplier
    speed: f32,
    /// Last checkpoint path (for display)
    last_checkpoint: Option<String>,
    /// Checkpoint load path input
    load_path: String,
    /// Status message
    status_message: Option<(String, std::time::Instant)>,
}

impl Default for ControlPanel {
    fn default() -> Self {
        // Try to find latest checkpoint on startup
        let latest = SimulationHandle::find_latest_checkpoint();
        Self {
            speed: 1.0,
            last_checkpoint: latest.clone(),
            load_path: latest.unwrap_or_default(),
            status_message: None,
        }
    }
}

impl ControlPanel {
    /// Create a new control panel
    pub fn new() -> Self {
        Self::default()
    }

    /// Render the control panel and return any commands
    pub fn show(&mut self, ui: &mut Ui, state: SimState) -> Vec<SimCommand> {
        let mut commands = Vec::new();

        // Clear old status messages after 3 seconds
        if let Some((_, time)) = &self.status_message {
            if time.elapsed().as_secs() > 3 {
                self.status_message = None;
            }
        }

        ui.horizontal(|ui| {
            // Play/Pause button
            let play_text = match state {
                SimState::Running => "⏸ Pause",
                SimState::Paused => "▶ Play",
                SimState::Stopped => "⏹ Stopped",
            };

            if ui.button(play_text).clicked() {
                match state {
                    SimState::Running => commands.push(SimCommand::Pause),
                    SimState::Paused => commands.push(SimCommand::Resume),
                    SimState::Stopped => {}
                }
            }

            // Step button (only when paused)
            ui.add_enabled_ui(state == SimState::Paused, |ui| {
                if ui.button("⏭ Step").clicked() {
                    commands.push(SimCommand::Step);
                }
            });

            ui.separator();

            // Speed slider
            ui.label("Speed:");
            let old_speed = self.speed;
            ui.add(egui::Slider::new(&mut self.speed, 0.1..=10.0).logarithmic(true));
            if (self.speed - old_speed).abs() > 0.01 {
                commands.push(SimCommand::SetSpeed(self.speed));
            }

            ui.separator();

            // Reset button - resets with current config (no GUI changes applied)
            if ui.button("🔄 Restart").clicked() {
                commands.push(SimCommand::Reset);
            }

            ui.separator();

            // Checkpoint controls
            if ui.button("💾 Save").on_hover_text("Save checkpoint now").clicked() {
                commands.push(SimCommand::SaveCheckpoint);
                self.status_message = Some(("Checkpoint saved!".to_string(), std::time::Instant::now()));
                // Update last checkpoint path
                self.last_checkpoint = SimulationHandle::find_latest_checkpoint();
                if let Some(ref path) = self.last_checkpoint {
                    self.load_path = path.clone();
                }
            }

            // Load button (only when paused)
            ui.add_enabled_ui(state == SimState::Paused, |ui| {
                if ui.button("📂 Load").on_hover_text("Load checkpoint").clicked() {
                    if !self.load_path.is_empty() {
                        commands.push(SimCommand::LoadCheckpoint(self.load_path.clone()));
                        self.status_message = Some(("Checkpoint loaded!".to_string(), std::time::Instant::now()));
                    } else {
                        self.status_message = Some(("No checkpoint path specified".to_string(), std::time::Instant::now()));
                    }
                }
            });
        });

        // Second row: checkpoint path and status
        ui.horizontal(|ui| {
            ui.label("Checkpoint:");
            ui.add(egui::TextEdit::singleline(&mut self.load_path)
                .desired_width(300.0)
                .hint_text("Path to checkpoint file"));

            if ui.button("📁").on_hover_text("Find latest checkpoint").clicked() {
                if let Some(path) = SimulationHandle::find_latest_checkpoint() {
                    self.load_path = path;
                    self.status_message = Some(("Found latest checkpoint".to_string(), std::time::Instant::now()));
                } else {
                    self.status_message = Some(("No checkpoints found".to_string(), std::time::Instant::now()));
                }
            }

            // Show status message
            if let Some((msg, _)) = &self.status_message {
                ui.separator();
                ui.label(egui::RichText::new(msg).italics());
            }
        });

        commands
    }

    /// Get current speed
    #[allow(dead_code)]
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Set speed
    #[allow(dead_code)]
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.1, 10.0);
    }
}
