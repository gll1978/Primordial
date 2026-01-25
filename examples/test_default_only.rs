//! Test using ONLY Config::default() with same settings as debug test

use primordial::{Config, World};

fn main() {
    let mut config = Config::default();

    // EXACTLY like debug_brain_no_instincts
    config.neural.use_instincts = false;
    config.evolution.add_neuron_rate = 0.15;
    config.evolution.add_connection_rate = 0.15;
    config.evolution.mutation_rate = 0.10;
    config.predation.enabled = false;
    config.seasons.enabled = false;
    config.terrain.enabled = false;

    println!("=== TEST DEFAULT CONFIG ===");
    println!("use_instincts: {}", config.neural.use_instincts);
    println!("add_neuron_rate: {}", config.evolution.add_neuron_rate);
    println!();

    let mut world = World::new(config);

    println!("Step      Pop   Gen   Brain_Mean  Brain_Max");
    println!("--------- ----- ----- ----------- ----------");

    for step in 0..=10000 {
        if step > 0 {
            world.step();
        }

        if step % 1000 == 0 {
            println!("{:9} {:5} {:5} {:11.3} {:10}",
                step,
                world.population(),
                world.generation_max,
                world.stats.brain_mean,
                world.stats.brain_max
            );
        }

        if world.is_extinct() {
            println!("\n*** EXTINCTION ***");
            break;
        }
    }

    println!("\n=== FINAL ===");
    println!("Brain Mean: {:.3}", world.stats.brain_mean);
    println!("Brain Max: {}", world.stats.brain_max);
}
