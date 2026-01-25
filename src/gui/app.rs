//! Main GUI application.

use eframe::egui;

use crate::config::Config;

use super::commands::{SimCommand, SimSettings};
use super::sim_thread::SimulationHandle;
use super::snapshot::WorldSnapshot;
use super::views::{ControlPanel, SettingsPanel, StatsPanel, WorldView};

/// Main application state
pub struct PrimordialApp {
    /// Simulation handle
    sim_handle: SimulationHandle,
    /// Latest world snapshot
    snapshot: Option<WorldSnapshot>,
    /// Currently selected organism ID
    selected_id: Option<u64>,
    /// World view component
    world_view: WorldView,
    /// Control panel component
    control_panel: ControlPanel,
    /// Stats panel component
    stats_panel: StatsPanel,
    /// Settings panel component
    settings_panel: SettingsPanel,
    /// Configuration
    config: Config,
}

impl PrimordialApp {
    /// Create a new application with the given configuration
    pub fn new(config: Config) -> Self {
        let sim_handle = SimulationHandle::spawn(config.clone());
        let settings = SimSettings::from_config(&config);

        Self {
            sim_handle,
            snapshot: None,
            selected_id: None,
            world_view: WorldView::new(),
            control_panel: ControlPanel::new(),
            stats_panel: StatsPanel::new(),
            settings_panel: SettingsPanel::new(settings),
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(Config::default())
    }
}

impl eframe::App for PrimordialApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll for new snapshot
        if let Some(new_snapshot) = self.sim_handle.try_recv_snapshot() {
            self.stats_panel.update(&new_snapshot);
            self.snapshot = Some(new_snapshot);
        }

        // Request repaint when simulation is running
        if self.sim_handle.is_running() {
            ctx.request_repaint();
        }

        // Top panel with controls
        egui::TopBottomPanel::top("control_panel").show(ctx, |ui| {
            let commands = self.control_panel.show(ui, self.sim_handle.state);
            for cmd in commands {
                if matches!(cmd, SimCommand::Reset) {
                    self.stats_panel.clear();
                    self.selected_id = None;
                }
                self.sim_handle.send(cmd);
            }
        });

        // Right panel with stats and settings
        egui::SidePanel::right("stats_panel")
            .min_width(250.0)
            .default_width(300.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Settings panel (collapsible)
                    if self.settings_panel.show(ui) {
                        // Apply & Reset was clicked
                        let settings = self.settings_panel.settings().clone();
                        self.stats_panel.clear();
                        self.selected_id = None;
                        self.sim_handle.send(SimCommand::ResetWithSettings(settings));
                    }

                    ui.separator();

                    // Stats panel
                    if let Some(ref snapshot) = self.snapshot {
                        self.stats_panel.show(ui, snapshot);
                    } else {
                        ui.label("Waiting for simulation...");
                    }
                });
            });

        // Central panel with world view
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref snapshot) = self.snapshot {
                // Show world view and handle organism selection
                if let Some(clicked_id) = self.world_view.show(ui, snapshot, self.selected_id) {
                    self.selected_id = Some(clicked_id);
                    self.sim_handle.send(SimCommand::SelectOrganism(Some(clicked_id)));
                }

                // Deselect on right-click or Escape
                if ui.input(|i| i.pointer.secondary_clicked() || i.key_pressed(egui::Key::Escape)) {
                    self.selected_id = None;
                    self.sim_handle.send(SimCommand::SelectOrganism(None));
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Loading simulation...");
                });
            }
        });

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(ref snapshot) = self.snapshot {
                    ui.label(format!(
                        "Time: {} | Pop: {} | Gen: {} | State: {:?}",
                        snapshot.time,
                        snapshot.stats.population,
                        snapshot.stats.generation_max,
                        self.sim_handle.state
                    ));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("PRIMORDIAL V2");
                });
            });
        });
    }
}

/// Run the GUI application
pub fn run_gui(config: Config) -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_maximized(true)
            .with_title("PRIMORDIAL V2 - Ecosystem Simulator"),
        ..Default::default()
    };

    eframe::run_native(
        "PRIMORDIAL V2",
        native_options,
        Box::new(|_cc| Box::new(PrimordialApp::new(config))),
    )
}
