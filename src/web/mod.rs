//! Web UI module for PRIMORDIAL V2.
//!
//! Provides a web-based frontend for the simulation using Axum + WebSocket.
//!
//! ## Architecture
//!
//! The web server runs in an async context with:
//! - **Simulation Thread**: Runs `World::step()` at high speed (~1000 steps/s)
//! - **WebSocket**: Broadcasts snapshots to connected clients
//! - **REST API**: Handles commands and settings
//!
//! ## Usage
//!
//! ```no_run
//! use primordial::Config;
//! use primordial::web::run_server;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = Config::default();
//!     run_server(config, "127.0.0.1:8080".parse().unwrap()).await.unwrap();
//! }
//! ```

mod routes;
mod server;
mod state;
mod websocket;

pub use server::run_server;
