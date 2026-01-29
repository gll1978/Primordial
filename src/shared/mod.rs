//! Shared module for GUI and Web UI communication.
//!
//! This module contains types that are shared between the GUI (egui) and
//! the Web UI (Axum) backends for controlling the simulation.

pub mod commands;
pub mod snapshot;
pub mod sim_thread;

pub use commands::{SimCommand, SimSettings, SimState};
pub use snapshot::{LayerView, OrganismDetail, OrganismView, WorldSnapshot};
pub use sim_thread::SimulationHandle;
