//! Test 500 steps with config.yaml

use primordial::{Config, World};

fn main() {
    // Load config from file
    let config = Config::from_file("config.yaml")
        .expect("Failed to load config.yaml");

    println!("=== PRIMORDIAL V2 - Test 500 Steps ===");
    println!("Config: config.yaml");
    println!();
    println!("=== CONFIG CHECK ===");
    println!("use_instincts: {}", config.neural.use_instincts);
    println!("add_neuron_rate: {}", config.evolution.add_neuron_rate);
    println!("mutation_rate: {}", config.evolution.mutation_rate);
    println!("grid_size: {}", config.world.grid_size);
    println!("initial_population: {}", config.organisms.initial_population);
    println!("reproduction.enabled: {}", config.reproduction.enabled);
    println!("===================");
    println!();

    let mut world = World::new(config);

    println!("Step      Pop   Gen   Births  Deaths  Brain_Mean  Brain_Max  Energy_Mean");
    println!("--------- ----- ----- ------- ------- ----------- ---------- -----------");

    for step in 0..=10000 {
        if step > 0 {
            world.step();
        }

        if step % 1000 == 0 {
            println!("{:9} {:5} {:5} {:7} {:7} {:11.3} {:10} {:11.1}",
                step,
                world.population(),
                world.generation_max,
                world.stats.births,
                world.stats.deaths,
                world.stats.brain_mean,
                world.stats.brain_max,
                world.stats.energy_mean
            );
        }

        if world.is_extinct() {
            println!("\n*** EXTINCTION at step {} ***", step);
            break;
        }
    }

    println!();
    println!("=== FINAL RESULTS ===");
    println!("Population: {}", world.population());
    println!("Max Generation: {}", world.generation_max);
    println!("Brain Mean: {:.3}", world.stats.brain_mean);
    println!("Brain Max: {}", world.stats.brain_max);
    println!("Energy Mean: {:.1}", world.stats.energy_mean);
    println!("Lineages: {}", world.stats.lineage_count);
    println!("Status: {}", if world.is_extinct() { "EXTINCT" } else { "ALIVE" });
}
