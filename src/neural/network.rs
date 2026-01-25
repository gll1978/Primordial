//! Neural network structure and forward propagation.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A single layer in the neural network
#[derive(Clone, Debug)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

impl Serialize for Layer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let shape = self.weights.shape();
        let weights_data: Vec<f32> = self.weights.iter().copied().collect();
        let biases_data: Vec<f32> = self.biases.iter().copied().collect();

        let mut state = serializer.serialize_struct("Layer", 3)?;
        state.serialize_field("shape", &[shape[0], shape[1]])?;
        state.serialize_field("weights", &weights_data)?;
        state.serialize_field("biases", &biases_data)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Layer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LayerData {
            shape: [usize; 2],
            weights: Vec<f32>,
            biases: Vec<f32>,
        }

        let data = LayerData::deserialize(deserializer)?;
        let weights = Array2::from_shape_vec((data.shape[0], data.shape[1]), data.weights)
            .map_err(serde::de::Error::custom)?;
        let biases = Array1::from_vec(data.biases);

        Ok(Layer { weights, biases })
    }
}

/// NEAT-style neural network
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralNet {
    /// Number of input neurons
    pub n_inputs: usize,
    /// Number of output neurons
    pub n_outputs: usize,
    /// Hidden layer sizes
    pub hidden_sizes: Vec<usize>,
    /// Network layers
    pub layers: Vec<Layer>,
    /// Next available node ID (for NEAT tracking)
    pub next_node_id: usize,
}

impl NeuralNet {
    /// Create a minimal network with no hidden layers
    pub fn new_minimal(n_inputs: usize, n_outputs: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights =
            Array2::from_shape_fn((n_inputs, n_outputs), |_| rng.gen_range(-0.5..0.5));
        let biases = Array1::zeros(n_outputs);

        Self {
            n_inputs,
            n_outputs,
            hidden_sizes: Vec::new(),
            layers: vec![Layer { weights, biases }],
            next_node_id: n_inputs + n_outputs,
        }
    }

    /// Create a network with bootstrap instincts for survival
    pub fn new_with_instincts(n_inputs: usize, n_outputs: usize) -> Self {
        let mut net = Self::new_minimal(n_inputs, n_outputs);

        // Bootstrap useful connections:
        // Input indices:
        //   0-3: food in 4 directions (N, E, S, W)
        //   4: threat count
        //   5: mate count
        //   6: energy (normalized)
        //   7: health (normalized)
        //   8: size (normalized)
        //   9: age (normalized)
        //   10-14: memory
        //   15: bias (always 1.0)
        //   16: time of day
        //   17-19: reserved

        // Output indices:
        //   0-3: move directions (N, E, S, W)
        //   4: eat
        //   5: reproduce
        //   6: attack
        //   7: signal
        //   8: wait
        //   9: reserved

        // Food direction -> Move in that direction
        net.layers[0].weights[[0, 0]] = 1.5; // Food north -> move north
        net.layers[0].weights[[1, 1]] = 1.5; // Food east -> move east
        net.layers[0].weights[[2, 2]] = 1.5; // Food south -> move south
        net.layers[0].weights[[3, 3]] = 1.5; // Food west -> move west

        // Low energy -> Eat
        net.layers[0].weights[[6, 4]] = -2.0; // Low energy encourages eating

        // High energy -> Reproduce
        net.layers[0].weights[[6, 5]] = 1.5; // High energy encourages reproduction

        // Threat -> Move away (slight randomness in direction)
        net.layers[0].weights[[4, 0]] = -0.5;
        net.layers[0].weights[[4, 2]] = 0.5;

        net
    }

    /// Perform forward pass through the network
    #[inline]
    pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        debug_assert_eq!(inputs.len(), self.n_inputs);

        let mut activation = Array1::from_vec(inputs.to_vec());

        for layer in &self.layers {
            activation = activation.dot(&layer.weights) + &layer.biases;
            // tanh activation
            activation.mapv_inplace(|x| x.tanh());
        }

        activation.to_vec()
    }

    /// Forward pass with fixed-size arrays (for performance)
    #[inline]
    pub fn forward_fixed<const N: usize, const M: usize>(&self, inputs: &[f32; N]) -> [f32; M] {
        let output_vec = self.forward(inputs);
        let mut outputs = [0.0f32; M];
        for (i, &val) in output_vec.iter().take(M).enumerate() {
            outputs[i] = val;
        }
        outputs
    }

    /// Get total number of neurons (complexity metric)
    #[inline]
    pub fn complexity(&self) -> usize {
        self.hidden_sizes.iter().sum::<usize>()
    }

    /// Get total number of parameters (weights + biases)
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }

    /// Check if network is valid (no NaN/Inf)
    pub fn is_valid(&self) -> bool {
        for layer in &self.layers {
            if layer.weights.iter().any(|&w| !w.is_finite()) {
                return false;
            }
            if layer.biases.iter().any(|&b| !b.is_finite()) {
                return false;
            }
        }
        true
    }

    /// Compute a hash of the network weights for genome identification
    pub fn hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash structure
        self.n_inputs.hash(&mut hasher);
        self.n_outputs.hash(&mut hasher);
        self.hidden_sizes.len().hash(&mut hasher);

        // Hash weights (sample to avoid being too slow)
        for layer in &self.layers {
            let weight_count = layer.weights.len();
            // Hash every 10th weight
            for (i, &w) in layer.weights.iter().enumerate() {
                if i % 10 == 0 {
                    w.to_bits().hash(&mut hasher);
                }
            }
            weight_count.hash(&mut hasher);
        }

        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_network() {
        let net = NeuralNet::new_minimal(20, 10);
        assert_eq!(net.n_inputs, 20);
        assert_eq!(net.n_outputs, 10);
        assert_eq!(net.layers.len(), 1);
        assert_eq!(net.complexity(), 0);
    }

    #[test]
    fn test_forward_pass() {
        let net = NeuralNet::new_minimal(20, 10);
        let inputs = vec![0.5; 20];
        let outputs = net.forward(&inputs);

        assert_eq!(outputs.len(), 10);
        // tanh outputs should be in [-1, 1]
        assert!(outputs.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_instinct_network() {
        let net = NeuralNet::new_with_instincts(20, 10);

        // Test that food north activates move north
        let mut inputs = vec![0.0; 20];
        inputs[0] = 1.0; // Food to the north
        inputs[15] = 1.0; // Bias

        let outputs = net.forward(&inputs);

        // Move north (output 0) should be highest among movement options
        let move_outputs = &outputs[0..4];
        let max_idx = move_outputs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(max_idx, 0, "Should prefer moving north when food is north");
    }

    #[test]
    fn test_network_validity() {
        let net = NeuralNet::new_minimal(20, 10);
        assert!(net.is_valid());
    }

    #[test]
    fn test_serialization() {
        let net = NeuralNet::new_with_instincts(20, 10);
        let serialized = bincode::serialize(&net).unwrap();
        let deserialized: NeuralNet = bincode::deserialize(&serialized).unwrap();

        assert_eq!(net.n_inputs, deserialized.n_inputs);
        assert_eq!(net.n_outputs, deserialized.n_outputs);
        assert_eq!(net.layers.len(), deserialized.layers.len());
    }
}
