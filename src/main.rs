//! PRIMORDIAL V2 - CLI Entry Point
//!
//! High-performance ecosystem simulator.

use clap::{Parser, Subcommand};
use primordial::checkpoint::{Checkpoint, CheckpointManager};
use primordial::{benchmark, Config, World};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "primordial")]
#[command(author = "Gabriele (dbowie)")]
#[command(version)]
#[command(about = "High-performance ecosystem simulator with NEAT-style neural networks")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a new simulation
    Run {
        /// Configuration file (YAML)
        #[arg(short, long, default_value = "config.yaml")]
        config: PathBuf,

        /// Number of steps to simulate
        #[arg(short, long, default_value = "10000")]
        steps: u64,

        /// Output directory for checkpoints
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,

        /// Quiet mode (minimal output)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Resume simulation from checkpoint
    Resume {
        /// Checkpoint file to resume from
        #[arg(short, long)]
        checkpoint: PathBuf,

        /// Number of additional steps
        #[arg(short, long, default_value = "10000")]
        steps: u64,

        /// Output directory
        #[arg(short, long, default_value = "output")]
        output: PathBuf,
    },

    /// Run performance benchmark
    Benchmark {
        /// Number of steps
        #[arg(short, long, default_value = "1000")]
        steps: u64,

        /// Population size
        #[arg(short, long, default_value = "1000")]
        population: usize,
    },

    /// Generate default configuration file
    Init {
        /// Output path
        #[arg(short, long, default_value = "config.yaml")]
        output: PathBuf,
    },

    /// Analyze a checkpoint file
    Analyze {
        /// Checkpoint file
        checkpoint: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            config,
            steps,
            output,
            seed,
            quiet,
        } => run_simulation(config, steps, output, seed, quiet),

        Commands::Resume {
            checkpoint,
            steps,
            output,
        } => resume_simulation(checkpoint, steps, output),

        Commands::Benchmark { steps, population } => run_benchmark(steps, population),

        Commands::Init { output } => generate_config(output),

        Commands::Analyze { checkpoint } => analyze_checkpoint(checkpoint),
    }
}

fn run_simulation(
    config_path: PathBuf,
    steps: u64,
    output: PathBuf,
    seed: Option<u64>,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load or create config
    let config = if config_path.exists() {
        println!("Loading config from: {:?}", config_path);
        Config::from_file(&config_path)?
    } else {
        println!("Using default configuration");
        Config::default()
    };

    // Create output directory
    std::fs::create_dir_all(&output)?;

    // Create world
    let mut world = if let Some(s) = seed {
        println!("Using seed: {}", s);
        World::new_with_seed(config.clone(), s)
    } else {
        World::new(config.clone())
    };

    println!("Starting simulation");
    println!("  Initial population: {}", world.population());
    println!("  Grid size: {}x{}", config.world.grid_size, config.world.grid_size);
    println!("  Steps: {}", steps);
    println!();

    // Checkpoint manager
    let mut checkpoint_mgr = CheckpointManager::new(
        output.to_string_lossy().to_string(),
        config.logging.checkpoint_interval,
        10, // Keep last 10 checkpoints
    );

    let start = Instant::now();
    let stats_interval = config.logging.stats_interval;

    for i in 0..steps {
        world.step();

        // Stats output
        if !quiet && i % stats_interval == 0 {
            println!("{}", world.stats.summary());
        }

        // Checkpoint
        if checkpoint_mgr.should_save(world.time) {
            let checkpoint = world.create_checkpoint();
            match checkpoint_mgr.save(&checkpoint) {
                Ok(path) => {
                    if !quiet {
                        println!("  Checkpoint saved: {}", path);
                    }
                }
                Err(e) => eprintln!("  Checkpoint error: {}", e),
            }
        }

        // Check for extinction
        if world.is_extinct() {
            println!("\nPopulation extinct at step {}", world.time);
            break;
        }
    }

    let elapsed = start.elapsed();
    let steps_per_sec = world.time as f64 / elapsed.as_secs_f64();

    println!();
    println!("=== Simulation Complete ===");
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Steps: {}", world.time);
    println!("Speed: {:.1} steps/s", steps_per_sec);
    println!("Final population: {}", world.population());
    println!("Max generation: {}", world.generation_max);
    println!("Lineages: {}", world.lineage_tracker.surviving_count());

    // Final checkpoint
    let final_checkpoint = world.create_checkpoint();
    let final_path = output.join("checkpoint_final.bin");
    final_checkpoint.save(&final_path)?;
    println!("Final checkpoint: {:?}", final_path);

    // Save stats history
    let stats_path = output.join("stats_history.json");
    world.stats_history.save(stats_path.to_str().unwrap())?;
    println!("Stats history: {:?}", stats_path);

    Ok(())
}

fn resume_simulation(
    checkpoint_path: PathBuf,
    steps: u64,
    output: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading checkpoint: {:?}", checkpoint_path);

    let checkpoint = Checkpoint::load(&checkpoint_path)?;
    let mut world = World::from_checkpoint(checkpoint);

    println!("Resumed at step {}", world.time);
    println!("Population: {}", world.population());
    println!("Running {} additional steps", steps);
    println!();

    std::fs::create_dir_all(&output)?;

    let mut checkpoint_mgr = CheckpointManager::new(
        output.to_string_lossy().to_string(),
        world.config.logging.checkpoint_interval,
        10,
    );

    let start = Instant::now();
    let target_time = world.time + steps;
    let stats_interval = world.config.logging.stats_interval;

    while world.time < target_time {
        world.step();

        if world.time % stats_interval == 0 {
            println!("{}", world.stats.summary());
        }

        if checkpoint_mgr.should_save(world.time) {
            let checkpoint = world.create_checkpoint();
            if let Ok(path) = checkpoint_mgr.save(&checkpoint) {
                println!("  Checkpoint: {}", path);
            }
        }

        if world.is_extinct() {
            println!("\nPopulation extinct at step {}", world.time);
            break;
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("=== Resume Complete ===");
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Final step: {}", world.time);
    println!("Speed: {:.1} steps/s", steps as f64 / elapsed.as_secs_f64());
    println!("Population: {}", world.population());

    Ok(())
}

fn run_benchmark(steps: u64, population: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PRIMORDIAL Benchmark ===");
    println!("Steps: {}", steps);
    println!("Population: {}", population);
    println!();

    let result = benchmark(steps, population);
    println!("{}", result);

    Ok(())
}

fn generate_config(output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::default();
    config.save(&output)?;
    println!("Configuration saved to: {:?}", output);
    Ok(())
}

fn analyze_checkpoint(checkpoint_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Checkpoint Analysis ===");
    println!("File: {:?}", checkpoint_path);
    println!();

    let checkpoint = Checkpoint::load(&checkpoint_path)?;

    println!("Time: {}", checkpoint.time);
    println!("Population: {}", checkpoint.organisms.len());
    println!();

    // Population stats
    let alive: Vec<_> = checkpoint.organisms.iter().filter(|o| o.is_alive()).collect();
    println!("Alive organisms: {}", alive.len());

    if !alive.is_empty() {
        let max_gen = alive.iter().map(|o| o.generation).max().unwrap_or(0);
        let avg_energy: f32 = alive.iter().map(|o| o.energy).sum::<f32>() / alive.len() as f32;
        let avg_brain: f32 =
            alive.iter().map(|o| o.brain.complexity() as f32).sum::<f32>() / alive.len() as f32;
        let max_brain = alive.iter().map(|o| o.brain.complexity()).max().unwrap_or(0);

        println!("Max generation: {}", max_gen);
        println!("Average energy: {:.1}", avg_energy);
        println!("Average brain complexity: {:.2}", avg_brain);
        println!("Max brain complexity: {}", max_brain);

        // Lineage distribution
        use std::collections::HashMap;
        let mut lineages: HashMap<u32, usize> = HashMap::new();
        for org in &alive {
            *lineages.entry(org.lineage_id).or_insert(0) += 1;
        }

        println!();
        println!("Lineages: {}", lineages.len());
        if let Some((&dominant_id, &count)) = lineages.iter().max_by_key(|(_, &c)| c) {
            println!(
                "Dominant lineage: {} ({} organisms, {:.1}%)",
                dominant_id,
                count,
                100.0 * count as f32 / alive.len() as f32
            );
        }
    }

    println!();
    println!(
        "Checkpoint size: {:.2} MB",
        checkpoint.size_bytes() as f64 / 1_000_000.0
    );

    Ok(())
}
