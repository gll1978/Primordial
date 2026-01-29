//! Web UI entry point for PRIMORDIAL.
//!
//! Run with: cargo run --features web --bin primordial-web
//!
//! Then open http://127.0.0.1:8080 in your browser.

use clap::Parser;
use primordial::{web::run_server, Config};
use std::net::SocketAddr;

#[derive(Parser)]
#[command(name = "primordial-web")]
#[command(about = "PRIMORDIAL Web UI - Ecosystem simulator with browser interface")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    /// Address to bind the server to
    #[arg(short, long, default_value = "127.0.0.1:8080")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    // Load or create default config
    let config = load_config(&args.config);

    // Parse bind address
    let bind: SocketAddr = args.bind.parse().map_err(|e| {
        format!("Invalid bind address '{}': {}", args.bind, e)
    })?;

    // Run the server
    run_server(config, bind).await
}

/// Load configuration from file or use default
fn load_config(config_path: &str) -> Config {
    // Try specified path first
    if let Ok(config) = Config::from_file(config_path) {
        log::info!("Loaded config from: {}", config_path);
        return config;
    }

    // Try common locations
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
