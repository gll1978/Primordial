//! Debug brain evolution WITHOUT instincts - do brains evolve from scratch?

use primordial::{Config, World};
use std::collections::HashMap;

fn main() {
    let mut config = Config::default();

    // DISABLE INSTINCTS - start from random weights
    config.neural.use_instincts = false;

    // High mutation rates
    config.evolution.add_neuron_rate = 0.15; // 15%
    config.evolution.add_connection_rate = 0.15;
    config.evolution.mutation_rate = 0.10; // More weight mutations

    // Simpler environment
    config.predation.enabled = false;
    config.seasons.enabled = false;
    config.terrain.enabled = false;

    println!("=== BRAIN EVOLUTION WITHOUT INSTINCTS ===");
    println!("use_instincts: {}", config.neural.use_instincts);
    println!("add_neuron_rate: {}", config.evolution.add_neuron_rate);
    println!("mutation_rate: {}", config.evolution.mutation_rate);
    println!();

    let mut world = World::new(config.clone());

    println!("=== T=0 INITIAL POPULATION ===");
    print_brain_stats(&world);

    for step in 1..=20000 {
        world.step();

        if step % 1000 == 0 {
            let dist = get_complexity_distribution(&world);
            let total: usize = dist.values().sum();
            let with_hidden: usize = dist.iter()
                .filter(|(&k, _)| k > 0)
                .map(|(_, &v)| v)
                .sum();
            let max_complexity = dist.keys().max().copied().unwrap_or(0);
            let mean: f32 = if total > 0 {
                dist.iter().map(|(&k, &v)| k * v).sum::<usize>() as f32 / total as f32
            } else { 0.0 };

            println!("Step {:5}: pop={:4}, gen={:3}, births={:4}, hidden={:4} ({:.1}%), max={}, mean={:.3}",
                step,
                world.population(),
                world.generation_max,
                world.stats.births,
                with_hidden,
                if total > 0 { 100.0 * with_hidden as f32 / total as f32 } else { 0.0 },
                max_complexity,
                mean
            );
        }

        if world.is_extinct() {
            println!("\n*** EXTINCTION at step {} ***", step);
            break;
        }
    }

    println!("\n=== FINAL ANALYSIS ===");
    if !world.is_extinct() {
        print_detailed_stats(&world);
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
    println!("With hidden neurons: {} ({:.1}%)", with_hidden, 100.0 * with_hidden as f32 / n as f32);
    println!("Max complexity: {}", max);
    println!("Mean complexity: {:.4}", total as f32 / n as f32);
}

fn print_detailed_stats(world: &World) {
    let alive: Vec<_> = world.organisms.iter().filter(|o| o.is_alive()).collect();
    print_brain_stats(world);

    // Find organisms with most neurons and positive energy
    let mut with_brains: Vec<_> = alive.iter()
        .filter(|o| o.brain.complexity() > 0)
        .collect();
    with_brains.sort_by(|a, b| {
        b.brain.complexity().cmp(&a.brain.complexity())
            .then(b.energy.partial_cmp(&a.energy).unwrap_or(std::cmp::Ordering::Equal))
    });

    println!("\nTop organisms with hidden neurons:");
    for (i, org) in with_brains.iter().take(10).enumerate() {
        println!("  {}: id={}, gen={}, hidden={}, energy={:.1}, age={}",
            i + 1, org.id, org.generation, org.brain.complexity(), org.energy, org.age);
    }

    // Check if any successful (positive energy) organisms have brains
    let successful_with_brains = alive.iter()
        .filter(|o| o.brain.complexity() > 0 && o.energy > 0.0)
        .count();
    println!("\nSuccessful (energy>0) organisms with hidden neurons: {}", successful_with_brains);
}
