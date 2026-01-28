//! Hebbian learning for lifetime adaptation of neural network weights.
//!
//! Implements "neurons that fire together wire together" with reward modulation.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Trace of activations through a single layer
#[derive(Clone, Debug)]
pub struct ActivationTrace {
    pub layer_idx: usize,
    pub pre_activations: Vec<f32>,
    pub post_activations: Vec<f32>,
    pub timestamp: u64,
}

/// Hebbian learning state attached to a neural network
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HebbianState {
    pub learning_rate: f32,
    pub decay_rate: f32,
    pub weight_limit: f32,
    #[serde(skip)]
    pub activation_traces: Vec<ActivationTrace>,
    pub update_count: u64,
    pub successful_updates: u64,
}

impl HebbianState {
    pub fn new(learning_rate: f32, decay_rate: f32, weight_limit: f32) -> Self {
        Self {
            learning_rate,
            decay_rate,
            weight_limit,
            activation_traces: Vec::new(),
            update_count: 0,
            successful_updates: 0,
        }
    }

    /// Record pre/post activations for a layer during forward pass
    pub fn record_activation(
        &mut self,
        layer_idx: usize,
        pre_activations: Vec<f32>,
        post_activations: Vec<f32>,
        timestamp: u64,
    ) {
        self.activation_traces.push(ActivationTrace {
            layer_idx,
            pre_activations,
            post_activations,
            timestamp,
        });
    }

    /// Apply Hebbian update to weights: Δw = η × pre × post × reward
    /// Then apply decay and clamp to weight_limit.
    /// Returns true if any weight changed meaningfully.
    pub fn apply_update(
        &mut self,
        weights: &mut Array2<f32>,
        reward: f32,
        layer_idx: usize,
    ) -> bool {
        self.update_count += 1;

        // Find the most recent trace for this layer
        let trace = self
            .activation_traces
            .iter()
            .rev()
            .find(|t| t.layer_idx == layer_idx);

        let trace = match trace {
            Some(t) => t,
            None => return false,
        };

        let (rows, cols) = weights.dim();
        let pre = &trace.pre_activations;
        let post = &trace.post_activations;

        if pre.len() != rows || post.len() != cols {
            return false;
        }

        let mut any_change = false;
        let eta = self.learning_rate;
        let limit = self.weight_limit;
        let decay = self.decay_rate;

        for i in 0..rows {
            for j in 0..cols {
                // Hebbian update: Δw = η × pre_i × post_j × reward
                let delta = eta * pre[i] * post[j] * reward;
                let w = &mut weights[[i, j]];

                // Apply decay toward zero
                *w *= 1.0 - decay;

                // Apply Hebbian delta
                *w += delta;

                // Clamp to limits
                *w = w.clamp(-limit, limit);

                if delta.abs() > 1e-6 {
                    any_change = true;
                }
            }
        }

        if any_change {
            self.successful_updates += 1;
        }

        // Clear traces after update
        self.activation_traces.retain(|t| t.layer_idx != layer_idx);

        any_change
    }

    /// Learning efficiency: fraction of updates that changed weights
    pub fn learning_efficiency(&self) -> f32 {
        if self.update_count == 0 {
            return 0.0;
        }
        self.successful_updates as f32 / self.update_count as f32
    }

    /// Clear all activation traces
    pub fn clear_traces(&mut self) {
        self.activation_traces.clear();
    }
}

/// Configuration for lifetime learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    pub enabled: bool,
    pub learning_rate: f32,
    pub decay_rate: f32,
    pub weight_limit: f32,
    pub consolidation_threshold: f32,
    pub working_memory_size: usize,
    /// Phase 2 Feature 3: Scale learning rate with brain complexity
    /// If true: actual_rate = learning_rate * (brain_layers / 10.0)
    #[serde(default)]
    pub scale_with_brain: bool,
    /// Minimum learning rate when scaling is enabled
    #[serde(default = "default_min_learning_rate")]
    pub min_learning_rate: f32,
    /// Maximum learning rate when scaling is enabled
    #[serde(default = "default_max_learning_rate")]
    pub max_learning_rate: f32,
}

fn default_min_learning_rate() -> f32 {
    0.0005
}

fn default_max_learning_rate() -> f32 {
    0.01
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            learning_rate: 0.001,
            decay_rate: 0.001,
            weight_limit: 5.0,
            consolidation_threshold: 0.1,
            working_memory_size: 100,
            scale_with_brain: false,
            min_learning_rate: 0.0005,
            max_learning_rate: 0.01,
        }
    }
}

impl LearningConfig {
    /// Calculate effective learning rate based on brain complexity
    /// Formula: base_rate * (brain_layers / 10.0), clamped to [min, max]
    pub fn effective_learning_rate(&self, brain_layers: usize) -> f32 {
        if !self.scale_with_brain {
            return self.learning_rate;
        }

        let scaled = self.learning_rate * (brain_layers as f32 / 10.0);
        scaled.clamp(self.min_learning_rate, self.max_learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_hebbian_strengthening() {
        // Co-activated neurons with positive reward should increase weight
        let mut state = HebbianState::new(0.1, 0.0, 5.0);
        let mut weights = Array2::zeros((3, 2));

        // Record activation: pre[0] and post[0] both active
        state.record_activation(
            0,
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0],
            0,
        );

        // Positive reward
        let changed = state.apply_update(&mut weights, 1.0, 0);
        assert!(changed);

        // Weight [0,0] should have increased (pre=1.0, post=1.0, reward=1.0)
        assert!(weights[[0, 0]] > 0.0, "Co-activated weight should increase");
        // Weight [1,0] should stay ~0 (pre=0.0)
        assert!(weights[[1, 0]].abs() < 1e-6, "Non-activated weight should stay near zero");
    }

    #[test]
    fn test_hebbian_weakening() {
        // Positive reward but mismatched activations should not strengthen
        let mut state = HebbianState::new(0.1, 0.0, 5.0);
        let mut weights = Array2::from_elem((2, 2), 0.5);

        state.record_activation(0, vec![1.0, 0.0], vec![0.0, 1.0], 0);
        state.apply_update(&mut weights, 1.0, 0);

        // Weight [0,0]: pre=1.0, post=0.0 → no change
        // Weight [0,1]: pre=1.0, post=1.0 → increase
        assert!(weights[[0, 1]] > 0.5);
    }

    #[test]
    fn test_negative_reward() {
        let mut state = HebbianState::new(0.1, 0.0, 5.0);
        let mut weights = Array2::from_elem((2, 2), 0.5);

        state.record_activation(0, vec![1.0, 1.0], vec![1.0, 1.0], 0);
        state.apply_update(&mut weights, -1.0, 0);

        // Negative reward should decrease co-activated weights
        assert!(weights[[0, 0]] < 0.5);
    }

    #[test]
    fn test_weight_clamping() {
        let mut state = HebbianState::new(10.0, 0.0, 2.0);
        let mut weights = Array2::zeros((1, 1));

        state.record_activation(0, vec![1.0], vec![1.0], 0);
        state.apply_update(&mut weights, 1.0, 0);

        assert!(weights[[0, 0]] <= 2.0, "Weight should be clamped to limit");
    }

    #[test]
    fn test_learning_efficiency() {
        let mut state = HebbianState::new(0.1, 0.0, 5.0);
        let mut weights = Array2::zeros((2, 2));

        // Active trace → successful
        state.record_activation(0, vec![1.0, 0.5], vec![0.8, 0.3], 0);
        state.apply_update(&mut weights, 1.0, 0);

        // No trace for layer 1 → unsuccessful
        state.apply_update(&mut weights, 1.0, 1);

        assert_eq!(state.update_count, 2);
        assert_eq!(state.successful_updates, 1);
        assert!((state.learning_efficiency() - 0.5).abs() < 1e-6);
    }
}
