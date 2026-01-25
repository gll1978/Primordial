//! # PRIMORDIAL V2
//!
//! High-performance ecosystem simulator with NEAT-style neural networks.
//!
//! ## Features
//!
//! - **Fast**: 500-2000 steps/second with 5000+ organisms
//! - **Parallel**: Leverages all CPU cores via Rayon
//! - **Evolvable**: NEAT-style neural network mutations
//! - **Configurable**: YAML configuration files
//! - **Reproducible**: Seeded random number generation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use primordial::{World, Config};
//!
//! // Create world with default config
//! let config = Config::default();
//! let mut world = World::new(config);
//!
//! // Run simulation
//! world.run(1000);
//!
//! // Check results
//! println!("Population: {}", world.population());
//! println!("Max generation: {}", world.generation_max);
//! ```
//!
//! ## Configuration
//!
//! ```rust
//! use primordial::Config;
//!
//! let mut config = Config::default();
//! config.organisms.initial_population = 200;
//! config.evolution.mutation_rate = 0.1;
//! ```
//!
//! ## Checkpoints
//!
//! ```rust,no_run
//! use primordial::{World, Config};
//! use primordial::checkpoint::Checkpoint;
//!
//! let mut world = World::new(Config::default());
//! world.run(1000);
//!
//! // Save checkpoint
//! let checkpoint = world.create_checkpoint();
//! checkpoint.save("checkpoint.bin").unwrap();
//!
//! // Load checkpoint
//! let loaded = Checkpoint::load("checkpoint.bin").unwrap();
//! let restored_world = World::from_checkpoint(loaded);
//! ```

pub mod checkpoint;
pub mod config;
pub mod ecology;
pub mod evolution;
pub mod grid;
pub mod neural;
pub mod organism;
pub mod stats;
pub mod world;

// Re-export main types
pub use config::Config;
pub use organism::Organism;
pub use world::World;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Run a quick benchmark
pub fn benchmark(steps: u64, population: usize) -> BenchmarkResult {
    use std::time::Instant;

    let mut config = Config::default();
    config.organisms.initial_population = population;

    let mut world = World::new(config);

    let start = Instant::now();
    world.run(steps);
    let elapsed = start.elapsed();

    BenchmarkResult {
        steps,
        initial_population: population,
        final_population: world.population(),
        elapsed_secs: elapsed.as_secs_f64(),
        steps_per_second: steps as f64 / elapsed.as_secs_f64(),
        max_generation: world.generation_max,
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub steps: u64,
    pub initial_population: usize,
    pub final_population: usize,
    pub elapsed_secs: f64,
    pub steps_per_second: f64,
    pub max_generation: u16,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Benchmark Results ===")?;
        writeln!(f, "Steps: {}", self.steps)?;
        writeln!(f, "Population: {} -> {}", self.initial_population, self.final_population)?;
        writeln!(f, "Time: {:.3}s", self.elapsed_secs)?;
        writeln!(f, "Speed: {:.1} steps/s", self.steps_per_second)?;
        writeln!(f, "Max generation: {}", self.max_generation)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_quick_simulation() {
        let config = Config::default();
        let mut world = World::new(config);

        world.run(100);

        assert!(world.time == 100);
    }

    #[test]
    fn test_benchmark() {
        let result = benchmark(100, 50);

        assert_eq!(result.steps, 100);
        assert!(result.steps_per_second > 0.0);
    }
}
