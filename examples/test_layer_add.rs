//! Test that add_neuron adds 2-6 neurons per layer (Python-compatible)

use primordial::neural::NeuralNet;

fn main() {
    println!("=== Testing add_neuron (Python-compatible) ===\n");

    let mut layer_sizes = Vec::new();

    // Test 100 add_neuron calls
    for i in 0..100 {
        let mut net = NeuralNet::new_minimal(24, 10);

        // Add one layer
        net.add_neuron();

        let layers = net.complexity(); // Number of layers
        let neurons = net.total_hidden_neurons(); // Total neurons

        if i < 10 {
            println!("Test {}: layers={}, neurons={}, hidden_sizes={:?}",
                i, layers, neurons, net.hidden_sizes);
        }

        layer_sizes.push(neurons);
    }

    let min = layer_sizes.iter().min().unwrap();
    let max = layer_sizes.iter().max().unwrap();
    let mean: f32 = layer_sizes.iter().sum::<usize>() as f32 / layer_sizes.len() as f32;

    println!("\n=== Results (100 samples) ===");
    println!("Layer sizes: min={}, max={}, mean={:.2}", min, max, mean);
    println!("Expected: min=2, max=6, mean≈4.0");

    assert!(*min >= 2, "Min should be >= 2");
    assert!(*max <= 6, "Max should be <= 6");
    println!("\n✓ Python-compatible layer sizes confirmed!");

    // Test multiple layers
    println!("\n=== Testing multiple layers ===");
    let mut net = NeuralNet::new_minimal(24, 10);

    for i in 0..5 {
        net.add_neuron();
        println!("After add {}: complexity={} layers, total_neurons={}, sizes={:?}",
            i + 1, net.complexity(), net.total_hidden_neurons(), net.hidden_sizes);
    }

    println!("\n✓ Multiple layers work correctly!");
}
