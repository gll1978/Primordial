//! World grid visualization.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::gui::snapshot::{terrain_color, u8_to_terrain, OrganismView, WorldSnapshot};

/// World view for rendering the simulation grid
pub struct WorldView {
    /// Zoom multiplier (1.0 = fit to window)
    zoom: f32,
    /// Show food overlay
    show_food: bool,
    /// Show grid lines
    show_grid: bool,
}

impl Default for WorldView {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            show_food: true,
            show_grid: false,
        }
    }
}

impl WorldView {
    /// Create a new world view
    pub fn new() -> Self {
        Self::default()
    }

    /// Render the world and return clicked organism ID if any
    pub fn show(
        &mut self,
        ui: &mut Ui,
        snapshot: &WorldSnapshot,
        selected_id: Option<u64>,
    ) -> Option<u64> {
        let mut clicked_organism: Option<u64> = None;

        // Reserve space for controls at bottom
        let controls_height = 30.0;

        // Calculate available space for the grid
        let available_size = ui.available_size() - Vec2::new(0.0, controls_height);

        // Calculate cell size to fit the grid in available space
        let base_cell_size = (available_size.x.min(available_size.y) / snapshot.grid_size as f32).max(1.0);
        let cell_size = base_cell_size * self.zoom;

        // Calculate actual grid size in pixels
        let grid_pixels = snapshot.grid_size as f32 * cell_size;

        // Center the grid in available space
        let offset = Vec2::new(
            ((available_size.x - grid_pixels) / 2.0).max(0.0),
            ((available_size.y - grid_pixels) / 2.0).max(0.0),
        );

        // Use a scroll area if zoomed in beyond available space
        let use_scroll = grid_pixels > available_size.x || grid_pixels > available_size.y;

        let show_grid_content = |ui: &mut Ui| -> (egui::Response, Option<u64>) {
            // Allocate space for the grid
            let (response, painter) = ui.allocate_painter(
                Vec2::new(grid_pixels.max(available_size.x), grid_pixels.max(available_size.y)),
                Sense::click(),
            );

            let origin = if use_scroll {
                response.rect.min
            } else {
                response.rect.min + offset
            };

            self.draw_grid(&painter, snapshot, selected_id, origin, cell_size, grid_pixels);

            // Handle click to select organism
            let clicked = if response.clicked() {
                if let Some(click_pos) = response.interact_pointer_pos() {
                    let grid_x = ((click_pos.x - origin.x) / cell_size) as u8;
                    let grid_y = ((click_pos.y - origin.y) / cell_size) as u8;

                    snapshot
                        .organisms
                        .iter()
                        .find(|o| o.x == grid_x && o.y == grid_y)
                        .map(|o| o.id)
                } else {
                    None
                }
            } else {
                None
            };

            (response, clicked)
        };

        if use_scroll {
            egui::ScrollArea::both()
                .max_height(available_size.y)
                .show(ui, |ui| {
                    let (_, clicked) = show_grid_content(ui);
                    clicked_organism = clicked;
                });
        } else {
            let (_, clicked) = show_grid_content(ui);
            clicked_organism = clicked;
        }

        // Options panel
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_food, "Food");
            ui.checkbox(&mut self.show_grid, "Grid");
            ui.add(egui::Slider::new(&mut self.zoom, 0.5..=3.0).text("Zoom"));
            if ui.button("Reset").clicked() {
                self.zoom = 1.0;
            }
        });

        clicked_organism
    }

    /// Draw the grid content
    fn draw_grid(
        &self,
        painter: &egui::Painter,
        snapshot: &WorldSnapshot,
        selected_id: Option<u64>,
        origin: Pos2,
        cell_size: f32,
        grid_pixels: f32,
    ) {
        // Draw terrain
        for y in 0..snapshot.grid_size {
            for x in 0..snapshot.grid_size {
                let idx = y * snapshot.grid_size + x;
                let terrain = u8_to_terrain(snapshot.terrain_grid[idx]);
                let (r, g, b) = terrain_color(terrain);

                let cell_rect = Rect::from_min_size(
                    Pos2::new(
                        origin.x + x as f32 * cell_size,
                        origin.y + y as f32 * cell_size,
                    ),
                    Vec2::splat(cell_size),
                );

                painter.rect_filled(cell_rect, 0.0, Color32::from_rgb(r, g, b));
            }
        }

        // Draw food overlay
        if self.show_food {
            for y in 0..snapshot.grid_size {
                for x in 0..snapshot.grid_size {
                    let idx = y * snapshot.grid_size + x;
                    let food = snapshot.food_grid[idx];

                    if food > 0.1 {
                        let alpha = ((food / 50.0) * 180.0).min(180.0) as u8;
                        let cell_rect = Rect::from_min_size(
                            Pos2::new(
                                origin.x + x as f32 * cell_size,
                                origin.y + y as f32 * cell_size,
                            ),
                            Vec2::splat(cell_size),
                        );

                        painter.rect_filled(
                            cell_rect,
                            0.0,
                            Color32::from_rgba_unmultiplied(0, 200, 0, alpha),
                        );
                    }
                }
            }
        }

        // Draw grid lines
        if self.show_grid {
            let grid_color = Color32::from_rgba_unmultiplied(100, 100, 100, 50);
            for i in 0..=snapshot.grid_size {
                let x = origin.x + i as f32 * cell_size;
                let y = origin.y + i as f32 * cell_size;

                painter.line_segment(
                    [Pos2::new(x, origin.y), Pos2::new(x, origin.y + grid_pixels)],
                    Stroke::new(1.0, grid_color),
                );
                painter.line_segment(
                    [Pos2::new(origin.x, y), Pos2::new(origin.x + grid_pixels, y)],
                    Stroke::new(1.0, grid_color),
                );
            }
        }

        // Draw organisms
        for org in &snapshot.organisms {
            let center = Pos2::new(
                origin.x + (org.x as f32 + 0.5) * cell_size,
                origin.y + (org.y as f32 + 0.5) * cell_size,
            );

            let radius = (cell_size * 0.4 * org.size.sqrt()).max(2.0);
            let color = organism_color(org);

            // Draw organism circle
            painter.circle_filled(center, radius, color);

            // Draw selection indicator
            if selected_id == Some(org.id) {
                painter.circle_stroke(center, radius + 2.0, Stroke::new(2.0, Color32::WHITE));
            }
        }
    }
}

/// Get color for an organism based on its traits
fn organism_color(org: &OrganismView) -> Color32 {
    if org.is_predator {
        // Red tint for predators
        let intensity = (org.energy / 100.0).clamp(0.3, 1.0);
        Color32::from_rgb(
            (255.0 * intensity) as u8,
            (50.0 * intensity) as u8,
            (50.0 * intensity) as u8,
        )
    } else if org.is_aquatic {
        // Cyan for aquatic organisms
        let intensity = (org.energy / 100.0).clamp(0.3, 1.0);
        Color32::from_rgb(
            (50.0 * intensity) as u8,
            (200.0 * intensity) as u8,
            (255.0 * intensity) as u8,
        )
    } else {
        // Green for normal organisms
        let intensity = (org.energy / 100.0).clamp(0.3, 1.0);
        Color32::from_rgb(
            (50.0 * intensity) as u8,
            (200.0 * intensity) as u8,
            (50.0 * intensity) as u8,
        )
    }
}
