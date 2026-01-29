//! GUI module for PRIMORDIAL V2.
//!
//! Provides a graphical frontend for the simulation using egui + eframe.
//!
//! ## Architecture
//!
//! The GUI runs in a separate thread from the simulation:
//! - **Simulation Thread**: Runs `World::step()` at high speed (~1000 steps/s)
//! - **Render Thread**: Runs egui at ~60fps for smooth visualization
//!
//! Communication is done via channels:
//! - Commands flow from GUI to simulation
//! - Snapshots flow from simulation to GUI
//!
//! ## Usage
//!
//! ```no_run
//! use primordial::Config;
//! use primordial::gui::run_gui;
//!
//! let config = Config::default();
//! run_gui(config).unwrap();
//! ```

mod app;
mod logging;
mod views;

// Re-export from shared module
pub use crate::shared::{
    LayerView, OrganismDetail, OrganismView, SimCommand, SimSettings, SimState,
    SimulationHandle, WorldSnapshot,
};

pub use app::{run_gui, PrimordialApp};
pub use logging::{LogPanel, SimulationLog};
