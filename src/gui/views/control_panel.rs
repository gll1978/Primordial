//! Control panel for simulation playback.

use egui::Ui;

use crate::gui::commands::{SimCommand, SimState};

/// Control panel for simulation playback
pub struct ControlPanel {
    /// Current speed multiplier
    speed: f32,
}

impl Default for ControlPanel {
    fn default() -> Self {
        Self { speed: 1.0 }
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

            // Reset button
            if ui.button("🔄 Reset").clicked() {
                commands.push(SimCommand::Reset);
            }
        });

        commands
    }

    /// Get current speed
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Set speed
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.1, 10.0);
    }
}
