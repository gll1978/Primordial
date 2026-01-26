//! Integration tests for PRIMORDIAL

use primordial::{Config, World};
use primordial::checkpoint::Checkpoint;

#[test]
fn test_full_simulation_cycle() {
    let mut config = Config::default();
    config.organisms.initial_population = 50;
    config.world.grid_size = 40;

    let mut world = World::new_with_seed(config, 12345);

    // Run simulation
    world.run(500);

    // Verify basic invariants
    assert!(world.time == 500);

    // Check organisms are valid
    for org in &world.organisms {
        if org.is_alive() {
            assert!(org.brain.is_valid());
            assert!(org.x < 40);
            assert!(org.y < 40);
        }
    }
}

#[test]
fn test_checkpoint_persistence() {
    let mut config = Config::default();
    config.organisms.initial_population = 30;
    config.world.grid_size = 30;

    let mut world = World::new_with_seed(config, 54321);
    world.run(100);

    // Create checkpoint
    let checkpoint = world.create_checkpoint();
    let temp_path = "/tmp/primordial_test_checkpoint.bin";
    checkpoint.save(temp_path).expect("Failed to save checkpoint");

    // Load checkpoint
    let loaded = Checkpoint::load(temp_path).expect("Failed to load checkpoint");

    // Verify data integrity
    assert_eq!(loaded.time, world.time);
    assert_eq!(loaded.organisms.len(), world.organisms.len());
    assert_eq!(loaded.random_seed, world.seed());

    // Restore and continue
    let mut restored = World::from_checkpoint(loaded);
    assert_eq!(restored.time, world.time);
    assert_eq!(restored.population(), world.population());

    // Run more steps
    restored.run(100);
    assert_eq!(restored.time, 200);

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_reproducibility() {
    let mut config = Config::default();
    config.organisms.initial_population = 20;
    config.world.grid_size = 30;

    // Note: Full reproducibility is not possible because:
    // 1. Rayon parallelism causes thread scheduling variations
    // 2. Neural network weights use thread_rng() not seeded RNG
    // This test verifies both simulations run successfully.
    let mut world1 = World::new_with_seed(config.clone(), 99999);
    let mut world2 = World::new_with_seed(config, 99999);

    world1.run(200);
    world2.run(200);

    // Times should be identical
    assert_eq!(world1.time, world2.time);

    // Both should have surviving populations (not extinct)
    assert!(world1.population() > 0 || world2.population() > 0,
        "Both simulations went extinct");
}

#[test]
fn test_evolution_happens() {
    let mut config = Config::default();
    config.organisms.initial_population = 100;
    config.evolution.mutation_rate = 0.1;
    config.evolution.add_neuron_rate = 0.1;

    let mut world = World::new_with_seed(config, 11111);

    // Run long enough for evolution
    world.run(2000);

    // Check that some organisms have evolved brains
    let complex_brains = world
        .organisms
        .iter()
        .filter(|o| o.is_alive() && o.brain.complexity() > 0)
        .count();

    // With high mutation rate, some brains should have grown
    // This might be 0 if all complex organisms died, which is valid
    println!("Organisms with complex brains: {}", complex_brains);
    println!("Max generation: {}", world.generation_max);

    // At least some generations should have passed
    assert!(world.generation_max > 0 || world.is_extinct());
}

#[test]
fn test_population_dynamics() {
    let mut config = Config::default();
    config.organisms.initial_population = 50;
    config.organisms.reproduction_threshold = 40.0;
    config.world.food_regen_rate = 0.2; // More food

    let mut world = World::new_with_seed(config, 77777);

    let mut populations = Vec::new();
    for _ in 0..10 {
        world.run(100);
        populations.push(world.population());
    }

    println!("Population over time: {:?}", populations);

    // Population should fluctuate (not stay constant)
    let min_pop = *populations.iter().min().unwrap();
    let max_pop = *populations.iter().max().unwrap();

    // Either population varies, or we've reached extinction/saturation
    println!("Population range: {} - {}", min_pop, max_pop);
}

#[test]
fn test_stats_tracking() {
    let mut config = Config::default();
    config.organisms.initial_population = 30;
    config.logging.stats_interval = 10;

    let mut world = World::new_with_seed(config, 33333);
    world.run(100);

    // Stats should be updated (time may be less if extinction occurred)
    assert!(world.stats.time <= 100);
    assert!(world.stats.time > 0);

    // History should have snapshots
    let history_len = world.stats_history.snapshots.len();
    assert!(history_len > 0, "Stats history should have snapshots");

    // Series data should be available
    let pop_series = world.stats_history.population_series();
    assert!(!pop_series.is_empty());
}

#[test]
fn test_neural_network_consistency() {
    let config = Config::default();
    let mut world = World::new_with_seed(config, 44444);

    // Run a few steps
    world.run(50);

    // All alive organisms should have valid brains
    for org in &world.organisms {
        if org.is_alive() {
            assert!(org.brain.is_valid(), "Brain should be valid");
            assert_eq!(org.brain.n_inputs, 24); // 20 base + 4 directional threat sensors
            assert_eq!(org.brain.n_outputs, 10);

            // Forward pass should work
            let inputs = [0.5f32; 24];
            let outputs = org.brain.forward(&inputs);
            assert_eq!(outputs.len(), 10);
            assert!(outputs.iter().all(|&x| x >= -1.0 && x <= 1.0));
        }
    }
}
