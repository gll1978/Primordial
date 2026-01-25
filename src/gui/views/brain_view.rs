//! Brain (neural network) visualization.

use egui::{Color32, Pos2, Stroke, Ui, Vec2};

use crate::gui::snapshot::OrganismDetail;

/// Input neuron labels
const INPUT_LABELS: &[&str] = &[
    "Food N", "Food E", "Food S", "Food W",  // 0-3: food directions
    "Threat", "Mates",                        // 4-5: nearby organisms
    "Energy", "Health", "Size", "Age",        // 6-9: internal state
    "Mem 0", "Mem 1", "Mem 2", "Mem 3", "Mem 4", // 10-14: memory
    "Bias", "Time", "Food@", "Crowd", "Signal", // 15-19: misc
];

/// Output neuron labels
const OUTPUT_LABELS: &[&str] = &[
    "Move N", "Move E", "Move S", "Move W",  // 0-3: movement
    "Eat", "Repro", "Attack",                 // 4-6: actions
    "Signal", "Wait", "Rsv",                  // 7-9: other
];

/// Brain visualization component
pub struct BrainView {
    /// Show weights as connections
    show_weights: bool,
    /// Minimum weight to display (filter noise)
    min_weight: f32,
}

impl Default for BrainView {
    fn default() -> Self {
        Self {
            show_weights: true,
            min_weight: 0.1,
        }
    }
}

impl BrainView {
    /// Create a new brain view
    pub fn new() -> Self {
        Self::default()
    }

    /// Render brain visualization for the selected organism
    pub fn show(&mut self, ui: &mut Ui, detail: &OrganismDetail) {
        egui::CollapsingHeader::new("Brain Visualization")
            .default_open(true)
            .show(ui, |ui| {
                // Controls
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.show_weights, "Show weights");
                    ui.add(egui::Slider::new(&mut self.min_weight, 0.0..=1.0).text("Min weight"));
                });

                ui.separator();

                // Brain stats
                let total_neurons: usize = detail.brain_layers.iter()
                    .map(|l| l.outputs)
                    .sum();
                let total_weights: usize = detail.brain_layers.iter()
                    .map(|l| l.weights.len())
                    .sum();

                ui.horizontal(|ui| {
                    ui.label(format!("Layers: {}", detail.brain_layers.len()));
                    ui.separator();
                    ui.label(format!("Hidden neurons: {}", total_neurons.saturating_sub(10)));
                    ui.separator();
                    ui.label(format!("Connections: {}", total_weights));
                });

                ui.separator();

                // Draw neural network
                self.draw_network(ui, detail);
            });
    }

    /// Draw the neural network visualization
    fn draw_network(&self, ui: &mut Ui, detail: &OrganismDetail) {
        if detail.brain_layers.is_empty() {
            ui.label("No brain data available");
            return;
        }

        // Calculate layout
        let available_width = ui.available_width().min(400.0);
        let available_height = 300.0;

        let (response, painter) = ui.allocate_painter(
            Vec2::new(available_width, available_height),
            egui::Sense::hover(),
        );

        let rect = response.rect;
        let padding = 10.0;

        // Collect layer sizes
        let mut layer_sizes: Vec<usize> = Vec::new();

        // Input layer
        if let Some(first_layer) = detail.brain_layers.first() {
            layer_sizes.push(first_layer.inputs);
        }

        // Hidden and output layers
        for layer in &detail.brain_layers {
            layer_sizes.push(layer.outputs);
        }

        let num_layers = layer_sizes.len();
        if num_layers == 0 {
            return;
        }

        // Calculate neuron positions
        let layer_spacing = (available_width - 2.0 * padding) / (num_layers.max(1) as f32);

        let mut neuron_positions: Vec<Vec<Pos2>> = Vec::new();

        for (layer_idx, &layer_size) in layer_sizes.iter().enumerate() {
            let x = rect.min.x + padding + layer_idx as f32 * layer_spacing + layer_spacing / 2.0;

            // Limit displayed neurons for large layers
            let display_size = layer_size.min(12);
            let neuron_spacing = (available_height - 2.0 * padding) / (display_size.max(1) as f32);

            let mut positions = Vec::new();
            for i in 0..display_size {
                let y = rect.min.y + padding + i as f32 * neuron_spacing + neuron_spacing / 2.0;
                positions.push(Pos2::new(x, y));
            }
            neuron_positions.push(positions);
        }

        // Draw connections (weights)
        if self.show_weights && neuron_positions.len() >= 2 {
            for (layer_idx, layer) in detail.brain_layers.iter().enumerate() {
                let from_positions = &neuron_positions[layer_idx];
                let to_positions = &neuron_positions[layer_idx + 1];

                let from_size = from_positions.len();
                let to_size = to_positions.len();

                for (w_idx, &weight) in layer.weights.iter().enumerate() {
                    if weight.abs() < self.min_weight {
                        continue;
                    }

                    // Calculate from/to indices (row-major order)
                    let from_idx = w_idx / layer.outputs;
                    let to_idx = w_idx % layer.outputs;

                    if from_idx >= from_size || to_idx >= to_size {
                        continue;
                    }

                    let from_pos = from_positions[from_idx];
                    let to_pos = to_positions[to_idx];

                    // Color based on weight sign and magnitude
                    let intensity = (weight.abs() * 255.0).min(255.0) as u8;
                    let color = if weight > 0.0 {
                        Color32::from_rgba_unmultiplied(0, intensity, 0, 150)
                    } else {
                        Color32::from_rgba_unmultiplied(intensity, 0, 0, 150)
                    };

                    let thickness = (weight.abs() * 2.0).clamp(0.5, 3.0);

                    painter.line_segment(
                        [from_pos, to_pos],
                        Stroke::new(thickness, color),
                    );
                }
            }
        }

        // Draw neurons
        let neuron_radius = 6.0;

        for (layer_idx, positions) in neuron_positions.iter().enumerate() {
            for (neuron_idx, &pos) in positions.iter().enumerate() {
                // Determine neuron color based on layer
                let color = if layer_idx == 0 {
                    // Input layer - blue
                    Color32::from_rgb(100, 150, 255)
                } else if layer_idx == neuron_positions.len() - 1 {
                    // Output layer - orange
                    Color32::from_rgb(255, 180, 100)
                } else {
                    // Hidden layer - gray
                    Color32::from_rgb(180, 180, 180)
                };

                painter.circle_filled(pos, neuron_radius, color);
                painter.circle_stroke(pos, neuron_radius, Stroke::new(1.0, Color32::WHITE));

                // Draw labels for input/output layers
                if layer_idx == 0 && neuron_idx < INPUT_LABELS.len() {
                    painter.text(
                        Pos2::new(pos.x - 45.0, pos.y),
                        egui::Align2::RIGHT_CENTER,
                        INPUT_LABELS[neuron_idx],
                        egui::FontId::proportional(8.0),
                        Color32::LIGHT_GRAY,
                    );
                } else if layer_idx == neuron_positions.len() - 1 && neuron_idx < OUTPUT_LABELS.len() {
                    painter.text(
                        Pos2::new(pos.x + 10.0, pos.y),
                        egui::Align2::LEFT_CENTER,
                        OUTPUT_LABELS[neuron_idx],
                        egui::FontId::proportional(8.0),
                        Color32::LIGHT_GRAY,
                    );
                }
            }
        }

        // Draw layer labels
        for (layer_idx, _) in layer_sizes.iter().enumerate() {
            let x = rect.min.x + padding + layer_idx as f32 * layer_spacing + layer_spacing / 2.0;
            let label = if layer_idx == 0 {
                "Input"
            } else if layer_idx == layer_sizes.len() - 1 {
                "Output"
            } else {
                "Hidden"
            };

            painter.text(
                Pos2::new(x, rect.max.y - 5.0),
                egui::Align2::CENTER_BOTTOM,
                label,
                egui::FontId::proportional(10.0),
                Color32::GRAY,
            );
        }

        // Legend
        ui.horizontal(|ui| {
            ui.label("Legend:");
            ui.colored_label(Color32::from_rgb(0, 200, 0), "Green = positive weight");
            ui.colored_label(Color32::from_rgb(200, 0, 0), "Red = negative weight");
        });
    }
}
