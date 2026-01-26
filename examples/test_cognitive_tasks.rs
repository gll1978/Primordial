//! Test cognitive tasks implementation - verify brain evolution beyond 1.0 plateau

use primordial::{Config, World};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("=== COGNITIVE TASKS TEST ===");
    println!("Target: brain_mean > 1.5 (breaking 1.0 plateau)\n");

    // Use default config with adjustments for cognitive task testing
    let mut config = Config::default();

    // Cognitive tasks settings
    config.neural.n_inputs = 38;  // 24 base + 8 memory + 3 temporal + 3 social
    config.neural.n_outputs = 12; // +2 signal actions

    // Survival-friendly parameters
    config.world.grid_size = 100;  // Smaller grid for faster testing
    config.world.food_regen_rate = 0.3;  // Good food availability
    config.organisms.initial_population = 300;
    config.organisms.metabolism_base = 0.3;  // Low metabolism
    config.organisms.food_energy = 30.0;     // Good energy from food
    config.organisms.move_cost = 0.2;        // Low move cost

    // Evolution parameters for brain growth
    config.evolution.add_neuron_rate = 0.30;  // High mutation rate
    config.evolution.add_connection_rate = 0.20;

    // Features
    config.predation.enabled = false;  // Disable to isolate cognitive effects
    config.seasons.enabled = true;
    config.terrain.enabled = false;    // Disable for simplicity

    config.safety.max_population = 1000;

    println!("Configuration:");
    println!("  n_inputs: {} (38 = cognitive tasks enabled)", config.neural.n_inputs);
    println!("  n_outputs: {}", config.neural.n_outputs);
    println!("  grid_size: {}", config.world.grid_size);
    println!("  initial_population: {}", config.organisms.initial_population);
    println!("  predation: {}", config.predation.enabled);
    println!("  seasons: {}", config.seasons.enabled);
    println!("  terrain: {}", config.terrain.enabled);
    println!();

    let mut world = World::new(config);

    let total_steps = 100000;
    let report_interval = 2500;

    println!("Running {} steps...\n", total_steps);
    println!("{:>8} {:>6} {:>5} {:>8} {:>8} {:>8} {:>6}",
        "Step", "Pop", "Gen", "Brain", "BrainMax", "Energy", "Kills");
    println!("{}", "-".repeat(60));

    // Track brain evolution
    let mut brain_samples: Vec<(u64, f32, f32)> = Vec::new();
    let mut total_kills = 0u64;

    for step in 1..=total_steps {
        world.step();
        total_kills += world.kills_this_step as u64;

        if step % report_interval == 0 {
            let pop = world.population();
            let brain_mean = world.stats.brain_mean;
            let brain_max = world.stats.brain_max;
            let energy_mean = world.stats.energy_mean;

            brain_samples.push((step as u64, brain_mean, brain_max as f32));

            println!("{:>8} {:>6} {:>5} {:>8.3} {:>8.1} {:>8.1} {:>6}",
                step, pop, world.generation_max, brain_mean, brain_max, energy_mean, total_kills);

            if world.is_extinct() {
                println!("\n*** EXTINCTION at step {} ***", step);
                break;
            }
        }
    }

    // Final analysis
    println!("\n{}", "=".repeat(60));
    println!("RESULTS ANALYSIS");
    println!("{}", "=".repeat(60));

    if brain_samples.is_empty() {
        println!("No data collected!");
        return;
    }

    let final_brain = brain_samples.last().unwrap().1;
    let max_brain_observed = brain_samples.iter().map(|s| s.2).fold(0.0f32, f32::max);
    let avg_brain = brain_samples.iter().map(|s| s.1).sum::<f32>() / brain_samples.len() as f32;

    println!("\nBrain Evolution:");
    println!("  Initial brain_mean: {:.3}", brain_samples.first().unwrap().1);
    println!("  Final brain_mean:   {:.3}", final_brain);
    println!("  Average brain_mean: {:.3}", avg_brain);
    println!("  Max brain observed: {:.1}", max_brain_observed);

    println!("\nSimulation Stats:");
    println!("  Total steps: {}", world.time);
    println!("  Final population: {}", world.population());
    println!("  Max generation: {}", world.generation_max);
    println!("  Total kills: {}", total_kills);

    // Verdict
    println!("\n{}", "=".repeat(60));
    if final_brain > 1.5 {
        println!("✓ SUCCESS: brain_mean {:.3} > 1.5 (plateau broken!)", final_brain);
    } else if final_brain > 1.2 {
        println!("~ PARTIAL: brain_mean {:.3} > 1.2 (improvement, but target is 1.5+)", final_brain);
    } else if final_brain > 1.0 {
        println!("~ MARGINAL: brain_mean {:.3} > 1.0 (slight improvement)", final_brain);
    } else {
        println!("✗ PLATEAU: brain_mean {:.3} ≤ 1.0 (no improvement)", final_brain);
    }
    println!("{}", "=".repeat(60));

    // Save results to file
    let filename = format!("logs/cognitive_test_{}.csv", world.time);
    if let Ok(mut file) = File::create(&filename) {
        writeln!(file, "step,brain_mean,brain_max").ok();
        for (step, mean, max) in &brain_samples {
            writeln!(file, "{},{:.4},{:.1}", step, mean, max).ok();
        }
        println!("\nResults saved to: {}", filename);
    }
}
