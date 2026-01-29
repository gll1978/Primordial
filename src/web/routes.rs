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
        // Checkpoint control
        .route("/api/checkpoint/save", post(save_checkpoint))
        .route("/api/checkpoint/load", post(load_checkpoint))
        .route("/api/checkpoint/latest", get(get_latest_checkpoint))
        .route("/api/checkpoint/list", get(list_checkpoints))
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

// --- Checkpoint Control ---

use crate::shared::sim_thread::SimulationHandle;

async fn save_checkpoint(State(state): State<Arc<AppState>>) -> Json<CheckpointResponse> {
    state.send_command(SimCommand::SaveCheckpoint).await;
    // Wait a moment for the checkpoint to be saved
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let latest = SimulationHandle::find_latest_checkpoint();
    Json(CheckpointResponse {
        success: true,
        path: latest,
        message: "Checkpoint saved".to_string(),
    })
}

#[derive(Deserialize)]
struct LoadCheckpointRequest {
    path: String,
}

async fn load_checkpoint(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LoadCheckpointRequest>,
) -> Json<CheckpointResponse> {
    if payload.path.is_empty() {
        return Json(CheckpointResponse {
            success: false,
            path: None,
            message: "No path specified".to_string(),
        });
    }

    // Check if file exists
    if !std::path::Path::new(&payload.path).exists() {
        return Json(CheckpointResponse {
            success: false,
            path: None,
            message: format!("File not found: {}", payload.path),
        });
    }

    state
        .send_command(SimCommand::LoadCheckpoint(payload.path.clone()))
        .await;
    Json(CheckpointResponse {
        success: true,
        path: Some(payload.path),
        message: "Checkpoint loaded".to_string(),
    })
}

#[derive(Serialize)]
struct CheckpointResponse {
    success: bool,
    path: Option<String>,
    message: String,
}

async fn get_latest_checkpoint() -> Json<CheckpointResponse> {
    match SimulationHandle::find_latest_checkpoint() {
        Some(path) => Json(CheckpointResponse {
            success: true,
            path: Some(path),
            message: "Latest checkpoint found".to_string(),
        }),
        None => Json(CheckpointResponse {
            success: false,
            path: None,
            message: "No checkpoints found".to_string(),
        }),
    }
}

#[derive(Serialize)]
struct CheckpointListResponse {
    checkpoints: Vec<CheckpointInfo>,
}

#[derive(Serialize)]
struct CheckpointInfo {
    path: String,
    filename: String,
    size_bytes: u64,
}

async fn list_checkpoints() -> Json<CheckpointListResponse> {
    let dir = SimulationHandle::default_checkpoint_dir();
    let mut checkpoints = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "bin") {
                if let Ok(metadata) = entry.metadata() {
                    checkpoints.push(CheckpointInfo {
                        path: path.to_string_lossy().to_string(),
                        filename: entry.file_name().to_string_lossy().to_string(),
                        size_bytes: metadata.len(),
                    });
                }
            }
        }
    }

    // Sort by filename (which includes timestamp) descending
    checkpoints.sort_by(|a, b| b.filename.cmp(&a.filename));

    Json(CheckpointListResponse { checkpoints })
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
