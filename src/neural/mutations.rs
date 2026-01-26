//! Neural network mutations (NEAT-style).

use super::network::{Layer, NeuralNet};
use ndarray::{Array1, Array2};
use rand::Rng;

/// Configuration for mutation operations
#[derive(Clone, Debug)]
pub struct MutationConfig {
    /// Probability of mutating each weight
    pub weight_mutation_rate: f32,
    /// Magnitude of weight perturbations
    pub weight_mutation_strength: f32,
    /// Probability of adding a new neuron
    pub add_neuron_rate: f32,
    /// Probability of adding a new connection
    pub add_connection_rate: f32,
    /// Maximum hidden neurons allowed
    pub max_neurons: usize,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            weight_mutation_rate: 0.05,
            weight_mutation_strength: 0.3,
            add_neuron_rate: 0.03,
            add_connection_rate: 0.05,
            max_neurons: 50,
        }
    }
}

impl NeuralNet {
    /// Apply all mutations according to config
    pub fn mutate(&mut self, config: &MutationConfig) {
        let mut rng = rand::thread_rng();

        // Weight mutations
        self.mutate_weights(config.weight_mutation_rate, config.weight_mutation_strength);

        // Structural mutations
        if rng.gen::<f32>() < config.add_neuron_rate {
            if self.complexity() < config.max_neurons {
                self.add_neuron();
            }
        }

        if rng.gen::<f32>() < config.add_connection_rate {
            self.add_connection();
        }
    }

    /// Mutate weights with given rate and strength
    pub fn mutate_weights(&mut self, rate: f32, strength: f32) {
        let mut rng = rand::thread_rng();

        for layer in &mut self.layers {
            // Mutate weights
            layer.weights.mapv_inplace(|w| {
                if rng.gen::<f32>() < rate {
                    let delta = rng.gen_range(-strength..strength);
                    (w + delta).clamp(-5.0, 5.0) // Prevent extreme values
                } else {
                    w
                }
            });

            // Mutate biases
            layer.biases.mapv_inplace(|b| {
                if rng.gen::<f32>() < rate {
                    let delta = rng.gen_range(-strength..strength);
                    (b + delta).clamp(-5.0, 5.0)
                } else {
                    b
                }
            });
        }
    }

    /// Add a new hidden layer (Python-compatible: adds 2-6 neurons at once)
    pub fn add_neuron(&mut self) {
        let mut rng = rand::thread_rng();

        // Python adds a new layer with 2-6 neurons each time
        let layer_size = rng.gen_range(2..=6);

        if self.hidden_sizes.is_empty() {
            // Create first hidden layer
            self.insert_hidden_layer(0, layer_size);
        } else {
            // Add new layer at end (like Python's append)
            let pos = self.hidden_sizes.len();
            self.insert_hidden_layer(pos, layer_size);
        }

        self.next_node_id += layer_size;
    }

    /// Add a connection (strengthen random existing connection)
    pub fn add_connection(&mut self) {
        let mut rng = rand::thread_rng();

        if !self.layers.is_empty() {
            let layer_idx = rng.gen_range(0..self.layers.len());
            let layer = &mut self.layers[layer_idx];

            let (rows, cols) = layer.weights.dim();
            if rows > 0 && cols > 0 {
                let i = rng.gen_range(0..rows);
                let j = rng.gen_range(0..cols);

                // Strengthen connection
                layer.weights[[i, j]] += rng.gen_range(-0.5..0.5);
                layer.weights[[i, j]] = layer.weights[[i, j]].clamp(-5.0, 5.0);
            }
        }
    }

    /// Insert a new hidden layer at the given position
    fn insert_hidden_layer(&mut self, position: usize, size: usize) {
        let mut rng = rand::thread_rng();

        // Determine sizes of adjacent layers
        let prev_size = if position == 0 {
            self.n_inputs
        } else {
            self.hidden_sizes[position - 1]
        };

        let next_size = if position >= self.hidden_sizes.len() {
            self.n_outputs
        } else {
            self.hidden_sizes[position]
        };

        // Create new layer: prev_size -> size
        let weights_in = Array2::from_shape_fn((prev_size, size), |_| rng.gen_range(-0.3..0.3));
        let biases_in = Array1::zeros(size);
        let new_layer = Layer {
            weights: weights_in,
            biases: biases_in,
        };

        // Modify existing layer or create output layer: size -> next_size
        if self.layers.len() > position {
            // Resize existing layer
            let weights_out =
                Array2::from_shape_fn((size, next_size), |_| rng.gen_range(-0.3..0.3));
            let biases_out = self.layers[position].biases.clone();
            self.layers[position] = Layer {
                weights: weights_out,
                biases: biases_out,
            };
        }

        // Insert new layer
        self.layers.insert(position, new_layer);
        self.hidden_sizes.insert(position, size);
    }

    /// Grow an existing hidden layer by one neuron
    fn grow_hidden_layer(&mut self, layer_idx: usize) {
        let mut rng = rand::thread_rng();

        // Get current size
        let current_size = self.hidden_sizes[layer_idx];
        let new_size = current_size + 1;
        self.hidden_sizes[layer_idx] = new_size;

        // Resize input weights (add column)
        let layer = &mut self.layers[layer_idx];
        let (rows, _cols) = layer.weights.dim();

        let mut new_weights = Array2::zeros((rows, new_size));
        for i in 0..rows {
            for j in 0..current_size {
                new_weights[[i, j]] = layer.weights[[i, j]];
            }
            // New column with random weights
            new_weights[[i, current_size]] = rng.gen_range(-0.3..0.3);
        }
        layer.weights = new_weights;

        // Add bias for new neuron
        let mut new_biases = Array1::zeros(new_size);
        for i in 0..current_size {
            new_biases[i] = layer.biases[i];
        }
        new_biases[current_size] = 0.0;
        layer.biases = new_biases;

        // Resize output weights of this layer (add row to next layer)
        if layer_idx + 1 < self.layers.len() {
            let next_layer = &mut self.layers[layer_idx + 1];
            let (_old_rows, cols) = next_layer.weights.dim();

            let mut new_weights = Array2::zeros((new_size, cols));
            for i in 0..current_size {
                for j in 0..cols {
                    new_weights[[i, j]] = next_layer.weights[[i, j]];
                }
            }
            // New row with random weights
            for j in 0..cols {
                new_weights[[current_size, j]] = rng.gen_range(-0.3..0.3);
            }
            next_layer.weights = new_weights;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_mutation() {
        let mut net = NeuralNet::new_minimal(24, 10);
        let original = net.layers[0].weights.clone();

        net.mutate_weights(1.0, 0.1); // 100% mutation rate

        // At least some weights should have changed
        let changed = net
            .layers[0]
            .weights
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(changed, "Weights should change after mutation");
    }

    #[test]
    fn test_add_neuron() {
        let mut net = NeuralNet::new_minimal(24, 10);
        assert_eq!(net.complexity(), 0);

        net.add_neuron();

        assert!(net.complexity() > 0);
        assert!(!net.hidden_sizes.is_empty());
    }

    #[test]
    fn test_mutation_preserves_validity() {
        let mut net = NeuralNet::new_minimal(24, 10);

        let config = MutationConfig {
            weight_mutation_rate: 0.5,
            weight_mutation_strength: 1.0,
            add_neuron_rate: 0.5,
            add_connection_rate: 0.5,
            max_neurons: 10,
        };

        // Apply many mutations
        for _ in 0..100 {
            net.mutate(&config);
        }

        assert!(net.is_valid(), "Network should remain valid after mutations");

        // Forward pass should still work
        let inputs = vec![0.5; 24];
        let outputs = net.forward(&inputs);
        assert!(outputs.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_weight_clamping() {
        let mut net = NeuralNet::new_minimal(24, 10);

        // Apply extreme mutations
        for _ in 0..1000 {
            net.mutate_weights(1.0, 10.0);
        }

        // Weights should be clamped
        for layer in &net.layers {
            assert!(layer.weights.iter().all(|&w| w >= -5.0 && w <= 5.0));
            assert!(layer.biases.iter().all(|&b| b >= -5.0 && b <= 5.0));
        }
    }
}
