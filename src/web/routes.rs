//! REST API routes for the web server.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};

use crate::shared::{SimCommand, SimSettings, SimState};

use super::state::AppState;

/// Create the API router
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        // Simulation control
        .route("/api/sim/pause", post(pause))
        .route("/api/sim/resume", post(resume))
        .route("/api/sim/step", post(step))
        .route("/api/sim/reset", post(reset))
        .route("/api/sim/speed", post(set_speed))
        .route("/api/sim/select", post(select_organism))
        // Settings
        .route("/api/settings", get(get_settings))
        .route("/api/settings", post(update_settings))
        // State
        .route("/api/state", get(get_state))
}

// --- Simulation Control ---

async fn pause(State(state): State<Arc<AppState>>) -> StatusCode {
    state.send_command(SimCommand::Pause).await;
    StatusCode::OK
}

async fn resume(State(state): State<Arc<AppState>>) -> StatusCode {
    state.send_command(SimCommand::Resume).await;
    StatusCode::OK
}

async fn step(State(state): State<Arc<AppState>>) -> StatusCode {
    state.send_command(SimCommand::Step).await;
    StatusCode::OK
}

async fn reset(State(state): State<Arc<AppState>>) -> StatusCode {
    state.send_command(SimCommand::Reset).await;
    StatusCode::OK
}

#[derive(Deserialize)]
struct SpeedRequest {
    speed: f32,
}

async fn set_speed(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SpeedRequest>,
) -> StatusCode {
    state
        .send_command(SimCommand::SetSpeed(payload.speed))
        .await;
    StatusCode::OK
}

#[derive(Deserialize)]
struct SelectRequest {
    id: Option<u64>,
}

async fn select_organism(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SelectRequest>,
) -> StatusCode {
    state
        .send_command(SimCommand::SelectOrganism(payload.id))
        .await;
    StatusCode::OK
}

// --- Settings ---

async fn get_settings(State(state): State<Arc<AppState>>) -> Json<SimSettings> {
    let settings = state.get_settings().await;
    Json(settings)
}

async fn update_settings(
    State(state): State<Arc<AppState>>,
    Json(settings): Json<SimSettings>,
) -> StatusCode {
    // Update stored settings
    state.update_settings(settings.clone()).await;
    // Reset simulation with new settings
    state
        .send_command(SimCommand::ResetWithSettings(settings))
        .await;
    StatusCode::OK
}

// --- State ---

#[derive(Serialize)]
struct StateResponse {
    state: SimState,
}

async fn get_state(State(state): State<Arc<AppState>>) -> Json<StateResponse> {
    let sim_state = state.get_state().await;
    Json(StateResponse { state: sim_state })
}
