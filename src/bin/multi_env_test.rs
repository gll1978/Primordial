//! Multi-environment test: measures generalization across varied patch layouts.

use clap::Parser;
use primordial::{Config, World};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "multi-env-test")]
struct Args {
    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,
    #[arg(short, long, default_value = "10000")]
    steps: u64,
    #[arg(short, long, default_value = "5")]
    envs: usize,
}

struct EnvResult {
    #[allow(dead_code)]
    env_idx: usize,
    #[allow(dead_code)]
    start_pop: usize,
    end_pop: usize,
    #[allow(dead_code)]
    start_brain: f32,
    end_brain: f32,
    survived: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    println!("=== Multi-Environment Test ===");
    println!("Environments: {}", args.envs);
    println!("Steps per env: {}", args.steps);
    println!();

    let config = Config::from_file(&args.config)?;
    let mut world = World::new(config.clone());

    // Warm up for 2000 steps first
    println!("Warming up (2000 steps)...");
    for _ in 0..2000 {
        world.step();
        if world.is_extinct() {
            println!("Extinction during warmup!");
            return Ok(());
        }
    }
    println!(
        "After warmup: Pop={}, Brain={:.2}",
        world.population(),
        world.stats.brain_mean
    );
    println!();

    let mut results = Vec::new();

    for env_idx in 0..args.envs {
        println!("--- Environment {} ---", env_idx + 1);

        // Reshuffle patches with unique seed
        if let Some(ref mut patches) = world.patch_world {
            let seed = 1000 + env_idx as u64 * 777;
            patches.reshuffle_patches(seed);
        }

        let start_pop = world.population();
        let start_brain = world.stats.brain_mean;

        let mut survived = true;
        for step in 0..args.steps {
            world.step();
            if world.is_extinct() {
                println!("  EXTINCTION at step {}", step);
                survived = false;
                break;
            }
        }

        let end_pop = world.population();
        let end_brain = world.stats.brain_mean;

        println!("  Population: {} -> {}", start_pop, end_pop);
        println!("  Brain: {:.2} -> {:.2}", start_brain, end_brain);
        println!();

        results.push(EnvResult {
            env_idx,
            start_pop,
            end_pop,
            start_brain,
            end_brain,
            survived,
        });

        if !survived {
            println!("Stopping: extinction occurred.");
            break;
        }
    }

    // Summary
    println!("=== SUMMARY ===");
    println!();

    let survived_results: Vec<_> = results.iter().filter(|r| r.survived).collect();
    if survived_results.is_empty() {
        println!("All environments resulted in extinction.");
        return Ok(());
    }

    let n = survived_results.len() as f32;
    let avg_pop = survived_results.iter().map(|r| r.end_pop).sum::<usize>() as f32 / n;
    let avg_brain = survived_results.iter().map(|r| r.end_brain).sum::<f32>() / n;

    println!("Survived: {}/{}", survived_results.len(), results.len());
    println!("Average final population: {:.0}", avg_pop);
    println!("Average final brain: {:.2}", avg_brain);

    let brain_variance = survived_results
        .iter()
        .map(|r| (r.end_brain - avg_brain).powi(2))
        .sum::<f32>()
        / n;
    let brain_std = brain_variance.sqrt();

    println!("Brain std dev: {:.2}", brain_std);
    if brain_std < 0.5 {
        println!("LOW variance - Good generalization!");
    } else if brain_std < 1.0 {
        println!("MODERATE variance - Some overfitting");
    } else {
        println!("HIGH variance - Poor generalization");
    }

    Ok(())
}
