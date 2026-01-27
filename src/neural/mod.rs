//! Neural network module for organism brains.
//!
//! Implements NEAT-style neural networks with:
//! - Dense layer representation
//! - Weight mutations
//! - Structural mutations (add neurons)
//! - Crossover between networks

mod network;
mod mutations;
mod crossover;
pub mod hebbian;
pub mod memory_consolidation;

pub use network::{NeuralNet, Layer};
pub use mutations::MutationConfig;
pub use crossover::CrossoverStrategy;
pub use hebbian::{HebbianState, LearningConfig};
