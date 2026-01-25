//! Configuration system for PRIMORDIAL simulation.
//!
//! Supports YAML configuration files with sensible defaults.

use crate::ecology::{DepletionConfig, FoodConfig, PredationConfig, SeasonsConfig, TerrainConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub world: WorldConfig,
    pub organisms: OrganismConfig,
    pub neural: NeuralConfig,
    pub evolution: EvolutionConfig,
    pub safety: SafetyConfig,
    pub logging: LoggingConfig,
    // Fase 2 additions
    #[serde(default)]
    pub predation: PredationConfig,
    #[serde(default)]
    pub seasons: SeasonsConfig,
    #[serde(default)]
    pub food: FoodConfig,
    #[serde(default)]
    pub terrain: TerrainConfig,
    #[serde(default)]
    pub depletion: DepletionConfig,
}

/// World/environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    /// Size of the square grid
    pub grid_size: usize,
    /// Food regeneration rate per step
    pub food_regen_rate: f32,
    /// Maximum food per cell
    pub food_max: f32,
    /// Initial food density (0.0 - 1.0)
    pub initial_food_density: f32,
}

/// Organism configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismConfig {
    /// Number of organisms at start
    pub initial_population: usize,
    /// Starting energy for new organisms
    pub initial_energy: f32,
    /// Energy cost to reproduce
    pub reproduction_cost: f32,
    /// Minimum energy to reproduce
    pub reproduction_threshold: f32,
    /// Base metabolic cost per step
    pub metabolism_base: f32,
    /// Energy gained from eating food
    pub food_energy: f32,
    /// Movement energy cost
    pub move_cost: f32,
}

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Number of input neurons
    pub n_inputs: usize,
    /// Number of output neurons
    pub n_outputs: usize,
    /// Use bootstrap instincts for new organisms
    pub use_instincts: bool,
}

/// Evolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Probability of weight mutation per weight
    pub mutation_rate: f32,
    /// Magnitude of weight mutations
    pub mutation_strength: f32,
    /// Probability of adding a neuron
    pub add_neuron_rate: f32,
    /// Probability of adding a connection
    pub add_connection_rate: f32,
}

/// Safety limits to prevent runaway simulations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Maximum allowed population
    pub max_population: usize,
    /// Maximum energy an organism can have
    pub max_energy: f32,
    /// Maximum age before death
    pub max_age: u32,
    /// Maximum neurons per brain
    pub max_neurons: usize,
}

/// Logging and checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Steps between checkpoints
    pub checkpoint_interval: u64,
    /// Steps between stats logging
    pub stats_interval: u64,
    /// Log level (error, warn, info, debug, trace)
    pub log_level: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            world: WorldConfig::default(),
            organisms: OrganismConfig::default(),
            neural: NeuralConfig::default(),
            evolution: EvolutionConfig::default(),
            safety: SafetyConfig::default(),
            logging: LoggingConfig::default(),
            predation: PredationConfig::default(),
            seasons: SeasonsConfig::default(),
            food: FoodConfig::default(),
            terrain: TerrainConfig::default(),
            depletion: DepletionConfig::default(),
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            grid_size: 80,
            food_regen_rate: 0.3,
            food_max: 100.0,
            initial_food_density: 0.5,
        }
    }
}

impl Default for OrganismConfig {
    fn default() -> Self {
        Self {
            initial_population: 200,
            initial_energy: 80.0,
            reproduction_cost: 25.0,
            reproduction_threshold: 50.0,
            metabolism_base: 0.5,
            food_energy: 25.0,
            move_cost: 0.3,
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            n_inputs: 20,
            n_outputs: 10,
            use_instincts: true,
        }
    }
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            mutation_rate: 0.05,
            mutation_strength: 0.3,
            add_neuron_rate: 0.03,
            add_connection_rate: 0.05,
        }
    }
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            max_population: 5000,
            max_energy: 500.0,
            max_age: 5000,
            max_neurons: 50,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: 500,
            stats_interval: 50,
            log_level: "info".to_string(),
        }
    }
}

impl Config {
    /// Load configuration from a YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a YAML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), String> {
        if self.world.grid_size == 0 || self.world.grid_size > 255 {
            return Err("grid_size must be between 1 and 255".to_string());
        }
        if self.organisms.initial_population == 0 {
            return Err("initial_population must be > 0".to_string());
        }
        if self.organisms.initial_population > self.safety.max_population {
            return Err("initial_population cannot exceed max_population".to_string());
        }
        if self.neural.n_inputs == 0 || self.neural.n_outputs == 0 {
            return Err("neural inputs/outputs must be > 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_roundtrip() {
        let config = Config::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let loaded: Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.world.grid_size, loaded.world.grid_size);
    }
}
