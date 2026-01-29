//! Axum server setup.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{routing::get, Router};
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};

use crate::Config;

use super::routes::api_router;
use super::state::{spawn_snapshot_relay, AppState};
use super::websocket::ws_handler;

/// Run the web server
pub async fn run_server(config: Config, bind: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    // Create shared state
    let state = Arc::new(AppState::new(config));

    // Start snapshot relay task
    spawn_snapshot_relay(state.clone());

    // CORS layer for development
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build the router
    let app = Router::new()
        // WebSocket endpoint
        .route("/ws", get(ws_handler))
        // REST API
        .merge(api_router())
        // Static files (served from /static directory)
        .nest_service("/", ServeDir::new("static").append_index_html_on_directories(true))
        .layer(cors)
        .with_state(state);

    log::info!("Starting web server on http://{}", bind);
    println!("PRIMORDIAL Web UI available at http://{}", bind);

    // Create listener and serve
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
