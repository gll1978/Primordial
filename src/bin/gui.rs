//! PRIMORDIAL V2 GUI Entry Point
//!
//! Run with: `cargo run --features gui --bin primordial-gui`

use primordial::config::Config;
use primordial::gui::run_gui;

fn main() -> eframe::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Load config or use default
    let config = load_config();

    log::info!("Starting PRIMORDIAL V2 GUI");
    log::info!("Grid size: {}", config.world.grid_size);
    log::info!("Initial population: {}", config.organisms.initial_population);

    run_gui(config)
}

/// Load configuration from file or use default
fn load_config() -> Config {
    // Try to load from common locations
    let paths = ["config.yaml", "primordial.yaml", "../config.yaml"];

    for path in paths {
        if let Ok(config) = Config::from_file(path) {
            log::info!("Loaded config from: {}", path);
            return config;
        }
    }

    log::info!("Using default configuration");
    Config::default()
}
