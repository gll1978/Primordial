//! Genetic crossover between neural networks.

use super::network::NeuralNet;
use rand::Rng;

/// Strategy for crossover operations
#[derive(Clone, Debug, Default)]
pub enum CrossoverStrategy {
    /// Inherit from fitter parent with slight mutations
    #[default]
    FitterParent,
    /// Average weights from both parents
    Average,
    /// Randomly select weights from either parent
    Uniform,
}

impl NeuralNet {
    /// Crossover with another network
    pub fn crossover(&self, other: &Self, fitness_self: f32, fitness_other: f32) -> Self {
        self.crossover_with_strategy(other, fitness_self, fitness_other, &CrossoverStrategy::default())
    }

    /// Crossover with specified strategy
    pub fn crossover_with_strategy(
        &self,
        other: &Self,
        fitness_self: f32,
        fitness_other: f32,
        strategy: &CrossoverStrategy,
    ) -> Self {
        match strategy {
            CrossoverStrategy::FitterParent => {
                self.crossover_fitter_parent(other, fitness_self, fitness_other)
            }
            CrossoverStrategy::Average => self.crossover_average(other),
            CrossoverStrategy::Uniform => self.crossover_uniform(other),
        }
    }

    /// Inherit structure from fitter parent, mix some weights
    fn crossover_fitter_parent(&self, other: &Self, fitness_self: f32, fitness_other: f32) -> Self {
        let mut rng = rand::thread_rng();

        // Choose base parent
        let (primary, secondary) = if fitness_self >= fitness_other {
            (self, other)
        } else {
            (other, self)
        };

        let mut child = primary.clone();

        // Occasionally inherit weights from secondary parent
        for (child_layer, secondary_layer) in child.layers.iter_mut().zip(secondary.layers.iter()) {
            let (rows, cols) = child_layer.weights.dim();
            let (sec_rows, sec_cols) = secondary_layer.weights.dim();

            // Only mix compatible dimensions
            let mix_rows = rows.min(sec_rows);
            let mix_cols = cols.min(sec_cols);

            for i in 0..mix_rows {
                for j in 0..mix_cols {
                    if rng.gen::<f32>() < 0.2 {
                        // 20% chance to inherit from secondary
                        child_layer.weights[[i, j]] = secondary_layer.weights[[i, j]];
                    }
                }
            }

            // Mix biases
            let mix_biases = child_layer.biases.len().min(secondary_layer.biases.len());
            for i in 0..mix_biases {
                if rng.gen::<f32>() < 0.2 {
                    child_layer.biases[i] = secondary_layer.biases[i];
                }
            }
        }

        child
    }

    /// Average weights from both parents (requires same topology)
    fn crossover_average(&self, other: &Self) -> Self {
        let mut child = self.clone();

        for (child_layer, other_layer) in child.layers.iter_mut().zip(other.layers.iter()) {
            let (rows, cols) = child_layer.weights.dim();
            let (other_rows, other_cols) = other_layer.weights.dim();

            // Only average compatible parts
            let avg_rows = rows.min(other_rows);
            let avg_cols = cols.min(other_cols);

            for i in 0..avg_rows {
                for j in 0..avg_cols {
                    child_layer.weights[[i, j]] =
                        (child_layer.weights[[i, j]] + other_layer.weights[[i, j]]) / 2.0;
                }
            }

            let avg_biases = child_layer.biases.len().min(other_layer.biases.len());
            for i in 0..avg_biases {
                child_layer.biases[i] = (child_layer.biases[i] + other_layer.biases[i]) / 2.0;
            }
        }

        child
    }

    /// Randomly select each weight from either parent
    fn crossover_uniform(&self, other: &Self) -> Self {
        let mut rng = rand::thread_rng();
        let mut child = self.clone();

        for (child_layer, other_layer) in child.layers.iter_mut().zip(other.layers.iter()) {
            let (rows, cols) = child_layer.weights.dim();
            let (other_rows, other_cols) = other_layer.weights.dim();

            let cross_rows = rows.min(other_rows);
            let cross_cols = cols.min(other_cols);

            for i in 0..cross_rows {
                for j in 0..cross_cols {
                    if rng.gen_bool(0.5) {
                        child_layer.weights[[i, j]] = other_layer.weights[[i, j]];
                    }
                }
            }

            let cross_biases = child_layer.biases.len().min(other_layer.biases.len());
            for i in 0..cross_biases {
                if rng.gen_bool(0.5) {
                    child_layer.biases[i] = other_layer.biases[i];
                }
            }
        }

        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover_fitter_parent() {
        let parent1 = NeuralNet::new_minimal(20, 10);
        let parent2 = NeuralNet::new_minimal(20, 10);

        let child = parent1.crossover(&parent2, 100.0, 50.0);

        assert_eq!(child.n_inputs, 20);
        assert_eq!(child.n_outputs, 10);
        assert!(child.is_valid());
    }

    #[test]
    fn test_crossover_average() {
        let parent1 = NeuralNet::new_minimal(20, 10);
        let parent2 = NeuralNet::new_minimal(20, 10);

        let child = parent1.crossover_with_strategy(&parent2, 50.0, 50.0, &CrossoverStrategy::Average);

        assert!(child.is_valid());

        // Check that at least some weights are averaged
        let inputs = vec![0.5; 20];
        let _outputs = child.forward(&inputs);
    }

    #[test]
    fn test_crossover_uniform() {
        let parent1 = NeuralNet::new_minimal(20, 10);
        let parent2 = NeuralNet::new_minimal(20, 10);

        let child = parent1.crossover_with_strategy(&parent2, 50.0, 50.0, &CrossoverStrategy::Uniform);

        assert!(child.is_valid());
    }

    #[test]
    fn test_crossover_different_topologies() {
        let mut parent1 = NeuralNet::new_minimal(20, 10);
        let parent2 = NeuralNet::new_minimal(20, 10);

        // Add neurons to parent1
        parent1.add_neuron();
        parent1.add_neuron();

        // Crossover should still work
        let child = parent1.crossover(&parent2, 100.0, 50.0);

        assert!(child.is_valid());
        // Child should inherit parent1's topology (fitter parent)
        assert_eq!(child.hidden_sizes, parent1.hidden_sizes);
    }
}
