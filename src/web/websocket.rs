//! WebSocket handler for real-time snapshot streaming.

use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};

use crate::shared::{SimCommand, SimState, WorldSnapshot};

use super::state::AppState;

/// WebSocket message from server to client
#[derive(Serialize)]
#[serde(tag = "type")]
enum ServerMessage {
    /// World snapshot update
    Snapshot(WorldSnapshot),
    /// Simulation state change
    StateChange { state: SimState },
}

/// WebSocket message from client to server
#[derive(Deserialize)]
#[serde(tag = "type")]
enum ClientMessage {
    /// Select an organism
    SelectOrganism { id: Option<u64> },
    /// Pause simulation
    Pause,
    /// Resume simulation
    Resume,
    /// Single step
    Step,
}

/// WebSocket upgrade handler
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle a WebSocket connection
async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to snapshot broadcasts
    let mut snapshot_rx = state.subscribe_snapshots();

    // Task to send snapshots to client
    let send_state = state.clone();
    let send_task = tokio::spawn(async move {
        // Send initial state
        let current_state = send_state.get_state().await;
        let state_msg = ServerMessage::StateChange {
            state: current_state,
        };
        if let Ok(json) = serde_json::to_string(&state_msg) {
            let _ = sender.send(Message::Text(json.into())).await;
        }

        // Stream snapshots
        loop {
            match snapshot_rx.recv().await {
                Ok(snapshot) => {
                    // Clone the snapshot out of Arc for serialization
                    let snapshot_data = (*snapshot).clone();
                    let msg = ServerMessage::Snapshot(snapshot_data);

                    match serde_json::to_string(&msg) {
                        Ok(json) => {
                            if sender.send(Message::Text(json.into())).await.is_err() {
                                // Client disconnected
                                break;
                            }
                        }
                        Err(e) => {
                            log::error!("Failed to serialize snapshot: {}", e);
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    log::warn!("WebSocket client lagged, skipped {} messages", n);
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    });

    // Task to receive commands from client
    let recv_state = state.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(result) = receiver.next().await {
            match result {
                Ok(Message::Text(text)) => {
                    if let Ok(msg) = serde_json::from_str::<ClientMessage>(&text) {
                        match msg {
                            ClientMessage::SelectOrganism { id } => {
                                recv_state
                                    .send_command(SimCommand::SelectOrganism(id))
                                    .await;
                            }
                            ClientMessage::Pause => {
                                recv_state.send_command(SimCommand::Pause).await;
                            }
                            ClientMessage::Resume => {
                                recv_state.send_command(SimCommand::Resume).await;
                            }
                            ClientMessage::Step => {
                                recv_state.send_command(SimCommand::Step).await;
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    break;
                }
                Err(e) => {
                    log::error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Wait for either task to complete (client disconnect)
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }

    log::debug!("WebSocket client disconnected");
}
