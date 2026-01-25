//! Neural network crossover for sexual reproduction.

use crate::neural::NeuralNet;
use serde::{Deserialize, Serialize};

/// Crossover system for combining neural networks
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CrossoverSystem {
    /// Total crossovers performed
    pub total_crossovers: u64,
    /// Crossovers where parent1 dominated
    pub parent1_dominant: u64,
    /// Crossovers where parent2 dominated
    pub parent2_dominant: u64,
    /// Crossovers with equal contribution
    pub equal_contribution: u64,
}

impl CrossoverSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Perform crossover between two neural networks.
    /// fitness1 and fitness2 determine which parent's structure dominates.
    pub fn crossover(
        &mut self,
        net1: &NeuralNet,
        net2: &NeuralNet,
        fitness1: f32,
        fitness2: f32,
    ) -> NeuralNet {
        self.total_crossovers += 1;

        // Determine dominant parent based on fitness
        let (dominant, _recessive, dominance) = if fitness1 > fitness2 * 1.2 {
            self.parent1_dominant += 1;
            (net1, net2, Dominance::Parent1)
        } else if fitness2 > fitness1 * 1.2 {
            self.parent2_dominant += 1;
            (net2, net1, Dominance::Parent2)
        } else {
            self.equal_contribution += 1;
            // Equal fitness - random choice for structure
            if rand::random() {
                (net1, net2, Dominance::Equal)
            } else {
                (net2, net1, Dominance::Equal)
            }
        };

        // Clone dominant's structure
        let mut child = dominant.clone();

        // Crossover weights from both parents
        Self::crossover_weights(&mut child, net1, net2, dominance);

        child
    }

    /// Crossover weights between parents
    fn crossover_weights(
        child: &mut NeuralNet,
        net1: &NeuralNet,
        net2: &NeuralNet,
        dominance: Dominance,
    ) {
        // For each layer in child
        for (i, layer) in child.layers.iter_mut().enumerate() {
            // Check if both parents have this layer
            let layer1 = net1.layers.get(i);
            let layer2 = net2.layers.get(i);

            match (layer1, layer2) {
                (Some(l1), Some(l2)) => {
                    // Both parents have this layer - crossover
                    Self::crossover_layer_weights(layer, l1, l2, dominance);
                }
                (Some(l1), None) => {
                    // Only parent 1 has this layer
                    if dominance != Dominance::Parent2 {
                        Self::copy_layer_weights(layer, l1);
                    }
                }
                (None, Some(l2)) => {
                    // Only parent 2 has this layer
                    if dominance != Dominance::Parent1 {
                        Self::copy_layer_weights(layer, l2);
                    }
                }
                (None, None) => {
                    // Neither parent has this layer (shouldn't happen)
                }
            }
        }
    }

    /// Crossover weights for a single layer
    fn crossover_layer_weights(
        child_layer: &mut crate::neural::Layer,
        layer1: &crate::neural::Layer,
        layer2: &crate::neural::Layer,
        dominance: Dominance,
    ) {
        let shape = child_layer.weights.shape();
        let rows = shape[0];
        let cols = shape[1];

        // Only crossover if shapes match
        if layer1.weights.shape() == shape && layer2.weights.shape() == shape {
            for j in 0..rows {
                for k in 0..cols {
                    // Use different strategies based on dominance
                    child_layer.weights[[j, k]] = match dominance {
                        Dominance::Parent1 => {
                            // 70% from parent1, 30% from parent2
                            layer1.weights[[j, k]] * 0.7 + layer2.weights[[j, k]] * 0.3
                        }
                        Dominance::Parent2 => {
                            // 30% from parent1, 70% from parent2
                            layer1.weights[[j, k]] * 0.3 + layer2.weights[[j, k]] * 0.7
                        }
                        Dominance::Equal => {
                            // 50/50 average
                            (layer1.weights[[j, k]] + layer2.weights[[j, k]]) / 2.0
                        }
                    };
                }
            }

            // Crossover biases
            let bias_len = child_layer.biases.len();
            if layer1.biases.len() == bias_len && layer2.biases.len() == bias_len {
                for j in 0..bias_len {
                    child_layer.biases[j] = match dominance {
                        Dominance::Parent1 => {
                            layer1.biases[j] * 0.7 + layer2.biases[j] * 0.3
                        }
                        Dominance::Parent2 => {
                            layer1.biases[j] * 0.3 + layer2.biases[j] * 0.7
                        }
                        Dominance::Equal => {
                            (layer1.biases[j] + layer2.biases[j]) / 2.0
                        }
                    };
                }
            }
        }
    }

    /// Copy weights from source to destination
    fn copy_layer_weights(
        dest: &mut crate::neural::Layer,
        src: &crate::neural::Layer,
    ) {
        if dest.weights.shape() == src.weights.shape() {
            dest.weights.assign(&src.weights);
        }
        if dest.biases.len() == src.biases.len() {
            dest.biases.assign(&src.biases);
        }
    }

    /// Get statistics as a string
    pub fn stats_string(&self) -> String {
        format!(
            "Crossovers: {} (P1 dom: {}, P2 dom: {}, equal: {})",
            self.total_crossovers,
            self.parent1_dominant,
            self.parent2_dominant,
            self.equal_contribution
        )
    }
}

/// Dominance type for crossover
#[derive(Clone, Copy, Debug, PartialEq)]
enum Dominance {
    Parent1,
    Parent2,
    Equal,
}

/// Crossover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossoverConfig {
    /// Is crossover enabled
    pub enabled: bool,
    /// Fitness ratio threshold for dominance (1.2 = 20% fitter)
    pub dominance_threshold: f32,
    /// Weight contribution from dominant parent (0.5-1.0)
    pub dominant_weight: f32,
}

impl Default for CrossoverConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dominance_threshold: 1.2,
            dominant_weight: 0.7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_crossover_system_creation() {
        let system = CrossoverSystem::new();
        assert_eq!(system.total_crossovers, 0);
    }

    #[test]
    fn test_crossover_same_structure() {
        let config = Config::default();
        let mut system = CrossoverSystem::new();

        let net1 = NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs);
        let net2 = NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs);

        let child = system.crossover(&net1, &net2, 1.0, 1.0);

        // Child should have same structure
        assert_eq!(child.layers.len(), net1.layers.len());
        assert!(child.is_valid());
        assert_eq!(system.total_crossovers, 1);
    }

    #[test]
    fn test_crossover_fitness_dominance() {
        let config = Config::default();
        let mut system = CrossoverSystem::new();

        let net1 = NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs);
        let net2 = NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs);

        // Much higher fitness for net1
        let _child = system.crossover(&net1, &net2, 100.0, 10.0);

        assert_eq!(system.parent1_dominant, 1);
        assert_eq!(system.parent2_dominant, 0);

        // Much higher fitness for net2
        let _child = system.crossover(&net1, &net2, 10.0, 100.0);

        assert_eq!(system.parent1_dominant, 1);
        assert_eq!(system.parent2_dominant, 1);
    }

    #[test]
    fn test_crossover_equal_fitness() {
        let config = Config::default();
        let mut system = CrossoverSystem::new();

        let net1 = NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs);
        let net2 = NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs);

        // Equal fitness
        for _ in 0..10 {
            let _child = system.crossover(&net1, &net2, 50.0, 50.0);
        }

        // All should be equal contribution
        assert_eq!(system.equal_contribution, 10);
    }

    #[test]
    fn test_child_validity() {
        let config = Config::default();
        let mut system = CrossoverSystem::new();

        // Create parents with different complexity
        let mut net1 = NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs);
        let net2 = NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs);

        // Add some neurons to one parent
        net1.add_neuron();
        net1.add_neuron();

        let child = system.crossover(&net1, &net2, 1.0, 1.0);

        // Child should be valid
        assert!(child.is_valid());
    }
}
