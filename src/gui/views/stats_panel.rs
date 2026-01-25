//! Statistics panel showing simulation metrics.

use egui::Ui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::gui::snapshot::{OrganismDetail, WorldSnapshot};

/// Stats panel showing simulation metrics and selected organism info
pub struct StatsPanel {
    /// History of population values for plotting
    population_history: Vec<[f64; 2]>,
    /// History of generation max values
    generation_history: Vec<[f64; 2]>,
    /// History of energy mean values
    energy_history: Vec<[f64; 2]>,
    /// Maximum history length
    max_history: usize,
}

impl Default for StatsPanel {
    fn default() -> Self {
        Self {
            population_history: Vec::new(),
            generation_history: Vec::new(),
            energy_history: Vec::new(),
            max_history: 500,
        }
    }
}

impl StatsPanel {
    /// Create a new stats panel
    pub fn new() -> Self {
        Self::default()
    }

    /// Update history from snapshot
    pub fn update(&mut self, snapshot: &WorldSnapshot) {
        let time = snapshot.time as f64;

        self.population_history
            .push([time, snapshot.stats.population as f64]);
        self.generation_history
            .push([time, snapshot.stats.generation_max as f64]);
        self.energy_history
            .push([time, snapshot.stats.energy_mean as f64]);

        // Trim to max history
        if self.population_history.len() > self.max_history {
            self.population_history.remove(0);
            self.generation_history.remove(0);
            self.energy_history.remove(0);
        }
    }

    /// Clear history (for reset)
    pub fn clear(&mut self) {
        self.population_history.clear();
        self.generation_history.clear();
        self.energy_history.clear();
    }

    /// Render the stats panel
    pub fn show(&mut self, ui: &mut Ui, snapshot: &WorldSnapshot) {
        // Current values
        egui::CollapsingHeader::new("Statistics")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("stats_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Time:");
                        ui.label(format!("{}", snapshot.time));
                        ui.end_row();

                        ui.label("Population:");
                        ui.label(format!("{}", snapshot.stats.population));
                        ui.end_row();

                        ui.label("Max Generation:");
                        ui.label(format!("{}", snapshot.stats.generation_max));
                        ui.end_row();

                        ui.label("Avg Energy:");
                        ui.label(format!("{:.1}", snapshot.stats.energy_mean));
                        ui.end_row();

                        ui.label("Avg Health:");
                        ui.label(format!("{:.1}", snapshot.stats.health_mean));
                        ui.end_row();

                        ui.label("Lineages:");
                        ui.label(format!("{}", snapshot.stats.lineage_count));
                        ui.end_row();

                        ui.label("Total Food:");
                        ui.label(format!("{:.0}", snapshot.stats.total_food));
                        ui.end_row();
                    });
            });

        ui.separator();

        // Population graph
        egui::CollapsingHeader::new("Population Graph")
            .default_open(true)
            .show(ui, |ui| {
                let points: PlotPoints = self.population_history.iter().copied().collect();
                let line = Line::new(points).color(egui::Color32::GREEN);

                Plot::new("population_plot")
                    .height(100.0)
                    .show_axes(true)
                    .show(ui, |plot_ui| {
                        plot_ui.line(line);
                    });
            });

        // Generation graph
        egui::CollapsingHeader::new("Generation Graph")
            .default_open(false)
            .show(ui, |ui| {
                let points: PlotPoints = self.generation_history.iter().copied().collect();
                let line = Line::new(points).color(egui::Color32::LIGHT_BLUE);

                Plot::new("generation_plot")
                    .height(100.0)
                    .show_axes(true)
                    .show(ui, |plot_ui| {
                        plot_ui.line(line);
                    });
            });

        ui.separator();

        // Selected organism details
        if let Some(ref detail) = snapshot.selected_organism {
            self.show_organism_detail(ui, detail);
        } else {
            ui.label("Click on an organism to see details");
        }
    }

    /// Show details of selected organism
    fn show_organism_detail(&self, ui: &mut Ui, org: &OrganismDetail) {
        egui::CollapsingHeader::new("Selected Organism")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("organism_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("ID:");
                        ui.label(format!("{}", org.id));
                        ui.end_row();

                        ui.label("Position:");
                        ui.label(format!("({}, {})", org.x, org.y));
                        ui.end_row();

                        ui.label("Energy:");
                        ui.label(format!("{:.1}", org.energy));
                        ui.end_row();

                        ui.label("Health:");
                        ui.label(format!("{:.1}", org.health));
                        ui.end_row();

                        ui.label("Age:");
                        ui.label(format!("{}", org.age));
                        ui.end_row();

                        ui.label("Generation:");
                        ui.label(format!("{}", org.generation));
                        ui.end_row();

                        ui.label("Lineage:");
                        ui.label(format!("{}", org.lineage_id));
                        ui.end_row();

                        ui.label("Size:");
                        ui.label(format!("{:.2}", org.size));
                        ui.end_row();

                        ui.label("Traits:");
                        let mut traits = Vec::new();
                        if org.is_predator {
                            traits.push("Predator");
                        }
                        if org.is_aquatic {
                            traits.push("Aquatic");
                        }
                        if traits.is_empty() {
                            traits.push("Normal");
                        }
                        ui.label(traits.join(", "));
                        ui.end_row();

                        ui.label("Kills:");
                        ui.label(format!("{}", org.kills));
                        ui.end_row();

                        ui.label("Offspring:");
                        ui.label(format!("{}", org.offspring_count));
                        ui.end_row();

                        ui.label("Food Eaten:");
                        ui.label(format!("{}", org.food_eaten));
                        ui.end_row();
                    });
            });

        // Brain info
        egui::CollapsingHeader::new("Brain")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(format!("Layers: {}", org.brain_layers.len()));
                for (i, layer) in org.brain_layers.iter().enumerate() {
                    ui.label(format!(
                        "  Layer {}: {}x{} ({} params)",
                        i,
                        layer.inputs,
                        layer.outputs,
                        layer.weights.len() + layer.biases.len()
                    ));
                }

                ui.separator();
                ui.label("Memory:");
                ui.horizontal(|ui| {
                    for (i, &val) in org.memory.iter().enumerate() {
                        ui.label(format!("[{}]: {:.2}", i, val));
                    }
                });
            });
    }
}
