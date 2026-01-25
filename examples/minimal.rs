//! Minimal example of PRIMORDIAL simulation

use primordial::{Config, World};

fn main() {
    println!("PRIMORDIAL V2 - Minimal Example");
    println!("================================\n");

    // Create default configuration
    let config = Config::default();

    // Create world with seeded RNG for reproducibility
    let mut world = World::new_with_seed(config, 42);

    println!("Initial state:");
    println!("  Population: {}", world.population());
    println!("  Grid: {}x{}", world.config.world.grid_size, world.config.world.grid_size);
    println!();

    // Run simulation
    let steps = 1000;
    println!("Running {} steps...\n", steps);

    for i in 0..steps {
        world.step();

        // Print progress every 100 steps
        if (i + 1) % 100 == 0 {
            println!(
                "Step {:4} | Pop: {:4} | Gen: {:3} | Brain: {:.1}",
                world.time,
                world.population(),
                world.generation_max,
                world.stats.brain_mean
            );
        }

        // Stop if extinct
        if world.is_extinct() {
            println!("\nPopulation went extinct at step {}", world.time);
            break;
        }
    }

    println!("\nFinal state:");
    println!("  Population: {}", world.population());
    println!("  Max generation: {}", world.generation_max);
    println!("  Lineages: {}", world.lineage_tracker.surviving_count());
}
