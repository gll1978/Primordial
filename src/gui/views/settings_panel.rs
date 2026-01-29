//! Settings panel for simulation parameters.

use egui::Ui;

use crate::shared::SimSettings;

/// Settings panel for configuring simulation parameters
pub struct SettingsPanel {
    /// Current settings being edited
    settings: SimSettings,
    /// Whether settings have been modified
    modified: bool,
}

impl SettingsPanel {
    /// Create a new settings panel with given initial settings
    pub fn new(settings: SimSettings) -> Self {
        Self {
            settings,
            modified: false,
        }
    }

    /// Get current settings
    pub fn settings(&self) -> &SimSettings {
        &self.settings
    }

    /// Check if settings were modified
    #[allow(dead_code)]
    pub fn is_modified(&self) -> bool {
        self.modified
    }

    /// Reset modified flag
    #[allow(dead_code)]
    pub fn clear_modified(&mut self) {
        self.modified = false;
    }

    /// Render the settings panel, returns true if "Apply & Reset" was clicked
    pub fn show(&mut self, ui: &mut Ui) -> bool {
        let mut apply_clicked = false;

        egui::CollapsingHeader::new("Simulation Settings")
            .default_open(false)
            .show(ui, |ui| {
                ui.set_min_width(230.0);

                egui::Grid::new("settings_grid")
                    .num_columns(2)
                    .spacing([10.0, 6.0])
                    .show(ui, |ui| {
                        // Population limits
                        ui.label("Max Population:");
                        if ui.add(egui::DragValue::new(&mut self.settings.max_population)
                            .clamp_range(100..=50000)
                            .speed(100))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Initial Population:");
                        if ui.add(egui::DragValue::new(&mut self.settings.initial_population)
                            .clamp_range(10..=1000)
                            .speed(10))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Max Steps (0=inf):");
                        if ui.add(egui::DragValue::new(&mut self.settings.max_steps)
                            .clamp_range(0..=1000000)
                            .speed(1000))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.separator();
                        ui.separator();
                        ui.end_row();

                        // World settings
                        ui.label("Grid Size (max 255):");
                        if ui.add(egui::DragValue::new(&mut self.settings.grid_size)
                            .clamp_range(20..=255)
                            .speed(5))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Food Regen Rate:");
                        if ui.add(egui::DragValue::new(&mut self.settings.food_regen_rate)
                            .clamp_range(0.01..=2.0)
                            .speed(0.01)
                            .fixed_decimals(2))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.separator();
                        ui.separator();
                        ui.end_row();

                        // Evolution settings
                        ui.label("Mutation Rate:");
                        if ui.add(egui::DragValue::new(&mut self.settings.mutation_rate)
                            .clamp_range(0.0..=1.0)
                            .speed(0.01)
                            .fixed_decimals(2))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Mutation Strength:");
                        if ui.add(egui::DragValue::new(&mut self.settings.mutation_strength)
                            .clamp_range(0.0..=2.0)
                            .speed(0.01)
                            .fixed_decimals(2))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Repro. Threshold:");
                        if ui.add(egui::DragValue::new(&mut self.settings.reproduction_threshold)
                            .clamp_range(10.0..=200.0)
                            .speed(1.0)
                            .fixed_decimals(0))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.separator();
                        ui.separator();
                        ui.end_row();

                        // Feature toggles
                        ui.label("Predation:");
                        if ui.checkbox(&mut self.settings.predation_enabled, "").changed() {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Seasons:");
                        if ui.checkbox(&mut self.settings.seasons_enabled, "").changed() {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Terrain:");
                        if ui.checkbox(&mut self.settings.terrain_enabled, "").changed() {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.separator();
                        ui.separator();
                        ui.end_row();

                        // Learning settings
                        ui.label("Learning:");
                        if ui.checkbox(&mut self.settings.learning_enabled, "").changed() {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Learning Rate:");
                        if ui.add(egui::DragValue::new(&mut self.settings.learning_rate)
                            .clamp_range(0.0001..=0.1)
                            .speed(0.0001)
                            .fixed_decimals(4))
                            .changed()
                        {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.separator();
                        ui.separator();
                        ui.end_row();

                        // Advanced features
                        ui.label("Diversity:");
                        if ui.checkbox(&mut self.settings.diversity_enabled, "").changed() {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Database:");
                        if ui.checkbox(&mut self.settings.database_enabled, "").changed() {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Cognitive Gate:");
                        if ui.checkbox(&mut self.settings.cognitive_gate_enabled, "").changed() {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Food Patches:");
                        if ui.checkbox(&mut self.settings.food_patches_enabled, "").changed() {
                            self.modified = true;
                        }
                        ui.end_row();

                        ui.label("Enhanced Senses:");
                        if ui.checkbox(&mut self.settings.enhanced_senses, "").changed() {
                            self.modified = true;
                            // Sync n_inputs with enhanced_senses
                            self.settings.n_inputs = if self.settings.enhanced_senses { 95 } else { 75 };
                        }
                        ui.end_row();
                    });

                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    if ui.button("Apply & Reset").clicked() {
                        apply_clicked = true;
                        self.modified = false;
                    }

                    if self.modified {
                        ui.label(egui::RichText::new("(modified)").color(egui::Color32::YELLOW));
                    }
                });
            });

        apply_clicked
    }
}
