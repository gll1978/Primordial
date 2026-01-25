//! Debug brain evolution - track add_neuron mutations and inheritance

use primordial::{Config, World};
use std::collections::HashMap;

fn main() {
    let mut config = Config::default();

    // Boost mutation rates for testing
    config.evolution.add_neuron_rate = 0.10; // 10% instead of 3%
    config.evolution.add_connection_rate = 0.10;

    // Simpler environment
    config.predation.enabled = false;
    config.seasons.enabled = false;
    config.terrain.enabled = false;

    println!("=== BRAIN EVOLUTION DEBUG ===");
    println!("add_neuron_rate: {}", config.evolution.add_neuron_rate);
    println!("add_connection_rate: {}", config.evolution.add_connection_rate);
    println!("max_neurons: {}", config.safety.max_neurons);
    println!();

    let mut world = World::new(config.clone());

    // Track brain complexity distribution
    let mut complexity_history: Vec<(u64, HashMap<usize, usize>)> = Vec::new();

    // Initial state
    println!("=== T=0 INITIAL POPULATION ===");
    print_brain_stats(&world);

    for step in 1..=10000 {
        world.step();

        // Sample every 500 steps
        if step % 500 == 0 {
            let dist = get_complexity_distribution(&world);
            complexity_history.push((step, dist.clone()));

            let total: usize = dist.values().sum();
            let with_hidden: usize = dist.iter()
                .filter(|(&k, _)| k > 0)
                .map(|(_, &v)| v)
                .sum();
            let pct_with_hidden = if total > 0 {
                100.0 * with_hidden as f32 / total as f32
            } else {
                0.0
            };

            println!("Step {:5}: pop={:4}, gen_max={:3}, births={:5}, with_hidden={:4} ({:.1}%)",
                step,
                world.population(),
                world.generation_max,
                world.stats.births,
                with_hidden,
                pct_with_hidden
            );

            // Show distribution
            if !dist.is_empty() {
                let mut keys: Vec<_> = dist.keys().collect();
                keys.sort();
                print!("           complexity dist: ");
                for k in keys.iter().take(10) {
                    print!("{}:{} ", k, dist[k]);
                }
                println!();
            }
        }

        // Detailed brain structure at key points
        if step == 100 || step == 1000 || step == 5000 {
            println!("\n=== DETAILED BRAIN STATS at T={} ===", step);
            print_detailed_brain_stats(&world);
            println!();
        }

        if world.is_extinct() {
            println!("\n*** EXTINCTION at step {} ***", step);
            break;
        }
    }

    println!("\n=== FINAL ANALYSIS ===");
    if !world.is_extinct() {
        print_detailed_brain_stats(&world);

        // Find organisms with most neurons
        let mut organisms: Vec<_> = world.organisms.iter()
            .filter(|o| o.is_alive())
            .collect();
        organisms.sort_by_key(|o| std::cmp::Reverse(o.brain.complexity()));

        println!("\nTop 5 brains:");
        for (i, org) in organisms.iter().take(5).enumerate() {
            println!("  {}: id={}, gen={}, hidden={}, layers={}, energy={:.1}",
                i + 1,
                org.id,
                org.generation,
                org.brain.complexity(),
                org.brain.layers.len(),
                org.energy
            );
        }
    }
}

fn get_complexity_distribution(world: &World) -> HashMap<usize, usize> {
    let mut dist = HashMap::new();
    for org in world.organisms.iter().filter(|o| o.is_alive()) {
        *dist.entry(org.brain.complexity()).or_insert(0) += 1;
    }
    dist
}

fn print_brain_stats(world: &World) {
    let alive: Vec<_> = world.organisms.iter().filter(|o| o.is_alive()).collect();
    let n = alive.len();

    if n == 0 {
        println!("No organisms alive!");
        return;
    }

    let complexities: Vec<usize> = alive.iter().map(|o| o.brain.complexity()).collect();
    let total: usize = complexities.iter().sum();
    let max = complexities.iter().max().copied().unwrap_or(0);
    let with_hidden = complexities.iter().filter(|&&c| c > 0).count();

    println!("Population: {}", n);
    println!("Total hidden neurons (sum): {}", total);
    println!("Max complexity: {}", max);
    println!("Organisms with hidden neurons: {} ({:.1}%)",
        with_hidden, 100.0 * with_hidden as f32 / n as f32);
    println!("Mean complexity: {:.4}", total as f32 / n as f32);
}

fn print_detailed_brain_stats(world: &World) {
    let alive: Vec<_> = world.organisms.iter().filter(|o| o.is_alive()).collect();
    let n = alive.len();

    if n == 0 {
        println!("No organisms alive!");
        return;
    }

    print_brain_stats(world);

    // Distribution
    let mut dist = get_complexity_distribution(world);
    let mut keys: Vec<_> = dist.keys().copied().collect();
    keys.sort();

    println!("\nComplexity distribution:");
    for k in &keys {
        let count = dist[k];
        let pct = 100.0 * count as f32 / n as f32;
        let bar = "#".repeat((pct / 2.0) as usize);
        println!("  {:3} neurons: {:5} ({:5.1}%) {}", k, count, pct, bar);
    }

    // Layer structure of a sample organism with hidden neurons
    if let Some(org) = alive.iter().find(|o| o.brain.complexity() > 0) {
        println!("\nSample organism with hidden neurons (id={}):", org.id);
        println!("  hidden_sizes: {:?}", org.brain.hidden_sizes);
        println!("  layers: {}", org.brain.layers.len());
        for (i, layer) in org.brain.layers.iter().enumerate() {
            let shape = layer.weights.shape();
            println!("    layer {}: {}x{} weights", i, shape[0], shape[1]);
        }
    } else {
        println!("\nNo organisms with hidden neurons found!");
    }
}
