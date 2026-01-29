//! Configuration system for PRIMORDIAL simulation.
//!
//! Supports YAML configuration files with sensible defaults.

use crate::ecology::{DepletionConfig, DynamicObstacleConfig, EnvironmentConfig, FoodConfig, LargePreyConfig, PredationConfig, SeasonsConfig, TerrainConfig};
use crate::genetics::sex::SexualReproductionConfig;
use crate::neural::LearningConfig;
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
    // Fase 2 Week 3-4: Genetics
    #[serde(default)]
    pub reproduction: SexualReproductionConfig,
    // B3: Large Prey and Cooperation
    #[serde(default)]
    pub large_prey: LargePreyConfig,
    // Phase 1: Foraging Memory
    #[serde(default)]
    pub food_patches: FoodPatchesConfig,
    #[serde(default)]
    pub behavior_tracking: BehaviorTrackingConfig,
    // Phase 2: Procedural Environments
    #[serde(default)]
    pub procedural_environment: EnvironmentConfig,
    // Phase 3: Lifetime Learning (Hebbian)
    #[serde(default)]
    pub learning: LearningConfig,
    // Database tracking
    #[serde(default)]
    pub database: DatabaseConfig,
    // Cognitive Gate system
    #[serde(default)]
    pub cognitive_gate: CognitiveGateConfig,
    // Brain tax system - penalizes overly complex brains
    #[serde(default)]
    pub brain_tax: BrainTaxConfig,
    // Phase 2 Feature 1: Enhanced Sensory System
    #[serde(default)]
    pub sensory: SensoryConfig,
    #[serde(default)]
    pub day_night: DayNightConfig,
    // Phase 2 Feature 2: Short-Term Memory System
    #[serde(default)]
    pub memory: MemoryConfig,
    // Phase 2 Feature 4: Dynamic Obstacles
    #[serde(default)]
    pub dynamic_obstacles: DynamicObstacleConfig,
    // Anti-bottleneck diversity mechanisms
    #[serde(default)]
    pub diversity: DiversityConfig,
}

/// Database configuration for individual organism tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Enable database logging
    pub enabled: bool,
    /// PostgreSQL connection URL
    pub url: String,
    /// Steps between organism snapshots
    pub snapshot_interval: u64,
    /// Whether to log learning events
    pub log_learning_events: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            url: "postgresql://primordial:primordial@localhost/primordial_v2".to_string(),
            snapshot_interval: 100,
            log_learning_events: false,
        }
    }
}

/// Cognitive Gate configuration - makes brain complexity necessary for eating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveGateConfig {
    /// Enable cognitive gate system
    pub enabled: bool,
    /// Tolerance for pattern matching (0.20 = brain can eat food 0.20 above its capability)
    pub tolerance: f32,
    /// Energy cost for failed eating attempt (frustration)
    pub failure_cost: f32,
    /// Food distribution: percentage of simple food (0.0-1.0)
    pub simple_ratio: f32,
    /// Food distribution: percentage of medium food (0.0-1.0)
    pub medium_ratio: f32,
    // Complex ratio is implicit: 1.0 - simple_ratio - medium_ratio

    // Complexity ranges (with overlap for smoother transitions)
    #[serde(default = "default_simple_min")]
    pub simple_complexity_min: f32,
    #[serde(default = "default_simple_max")]
    pub simple_complexity_max: f32,
    #[serde(default = "default_medium_min")]
    pub medium_complexity_min: f32,
    #[serde(default = "default_medium_max")]
    pub medium_complexity_max: f32,
    #[serde(default = "default_complex_min")]
    pub complex_complexity_min: f32,
    #[serde(default = "default_complex_max")]
    pub complex_complexity_max: f32,

    // Energy per food tier
    #[serde(default = "default_simple_energy")]
    pub simple_energy: f32,
    #[serde(default = "default_medium_energy")]
    pub medium_energy: f32,
    #[serde(default = "default_complex_energy")]
    pub complex_energy: f32,
}

// Default functions for serde
fn default_simple_min() -> f32 { 0.00 }
fn default_simple_max() -> f32 { 0.25 }
fn default_medium_min() -> f32 { 0.20 }
fn default_medium_max() -> f32 { 0.60 }
fn default_complex_min() -> f32 { 0.55 }
fn default_complex_max() -> f32 { 1.00 }
fn default_simple_energy() -> f32 { 23.0 }
fn default_medium_energy() -> f32 { 27.0 }
fn default_complex_energy() -> f32 { 32.0 }

impl Default for CognitiveGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tolerance: 0.20,
            failure_cost: 0.3,
            simple_ratio: 0.75,
            medium_ratio: 0.20,
            // complex_ratio: 0.05 (implicit)
            simple_complexity_min: 0.00,
            simple_complexity_max: 0.25,
            medium_complexity_min: 0.20,
            medium_complexity_max: 0.60,
            complex_complexity_min: 0.55,
            complex_complexity_max: 1.00,
            simple_energy: 23.0,
            medium_energy: 27.0,
            complex_energy: 32.0,
        }
    }
}

/// Brain tax configuration - penalizes overly complex brains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainTaxConfig {
    /// Enable brain tax
    pub enabled: bool,
    /// Layer threshold before tax applies
    pub threshold: usize,
    /// Energy cost per layer above threshold (per step)
    pub cost_per_layer: f32,
}

impl Default for BrainTaxConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 10,
            cost_per_layer: 0.01,
        }
    }
}

/// Enhanced sensory system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryConfig {
    /// Enable enhanced senses (8-direction vision, olfaction, audition)
    pub enhanced_senses: bool,
    /// Vision fovea range (high resolution)
    pub vision_fovea_range: u8,
    /// Vision peripheral range (low resolution, presence detection only)
    pub vision_peripheral_range: u8,
    /// Olfaction detection range
    pub olfaction_range: u8,
    /// Audition detection range
    pub audition_range: u8,
}

impl Default for SensoryConfig {
    fn default() -> Self {
        Self {
            enhanced_senses: false,
            vision_fovea_range: 5,
            vision_peripheral_range: 10,
            olfaction_range: 8,
            audition_range: 6,
        }
    }
}

/// Day/Night cycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayNightConfig {
    /// Enable day/night cycle
    pub enabled: bool,
    /// Steps per complete day/night cycle
    pub cycle_length: u64,
    /// Vision multiplier during night (0.0-1.0)
    pub night_vision_penalty: f32,
    /// Olfaction multiplier during night (>1.0 for bonus)
    pub night_olfaction_bonus: f32,
}

impl Default for DayNightConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cycle_length: 2000,
            night_vision_penalty: 0.3,
            night_olfaction_bonus: 1.5,
        }
    }
}

/// Diversity configuration for anti-bottleneck mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Enable diversity mechanisms
    pub enabled: bool,

    // Speciation
    /// Enable speciation based on genetic distance
    pub speciation_enabled: bool,
    /// Minimum genetic distance for same species (always compatible)
    pub min_genetic_distance: usize,
    /// Maximum genetic distance for different species (never compatible)
    pub max_genetic_distance: usize,
    /// Mating probability in boundary zone
    pub boundary_mating_probability: f32,

    // Niches
    /// Enable niche specialization
    pub niche_enabled: bool,
    /// Bonus for organisms specialized in a terrain
    pub niche_specialist_bonus: f32,
    /// Penalty for generalists
    pub niche_generalist_penalty: f32,
    /// Steps required to become adapted to a terrain
    pub niche_adaptation_time: u64,
    /// Bonus for matching diet and terrain
    pub terrain_diet_synergy_bonus: f32,

    // Frequency-dependent selection
    /// Enable frequency-dependent selection
    pub frequency_dependent_enabled: bool,
    /// Maximum bonus for rare strategies
    pub rare_strategy_max_bonus: f32,
    /// Threshold below which strategy is considered rare
    pub rare_strategy_threshold: f32,
    /// Penalty for common strategies
    pub common_strategy_penalty: f32,
    /// Threshold above which strategy is considered common
    pub common_strategy_threshold: f32,
    /// Strategy classification method: "diet", "terrain", "brain", or "combined"
    pub strategy_classification: String,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            speciation_enabled: true,
            min_genetic_distance: 5,
            max_genetic_distance: 20,
            boundary_mating_probability: 0.3,
            niche_enabled: true,
            niche_specialist_bonus: 1.4,
            niche_generalist_penalty: 0.85,
            niche_adaptation_time: 150,
            terrain_diet_synergy_bonus: 1.25,
            frequency_dependent_enabled: true,
            rare_strategy_max_bonus: 1.6,
            rare_strategy_threshold: 0.08,
            common_strategy_penalty: 0.7,
            common_strategy_threshold: 0.25,
            strategy_classification: "combined".to_string(),
        }
    }
}

/// Short-term memory system configuration (Phase 2 Feature 2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable short-term memory system
    pub enabled: bool,
    /// Base buffer size multiplier (actual size = brain_layers * multiplier)
    pub buffer_multiplier: usize,
    /// Minimum buffer size (even for small brains)
    pub min_buffer_size: usize,
    /// Maximum buffer size (cap for large brains)
    pub max_buffer_size: usize,
    /// Memory decay rate per step (0.0-1.0, higher = faster decay)
    pub decay_rate: f32,
    /// Number of sensory features to store per frame (compressed snapshot)
    pub features_per_frame: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            buffer_multiplier: 2,      // buffer_size = brain_layers * 2
            min_buffer_size: 4,        // At least 4 frames
            max_buffer_size: 40,       // Cap at 40 frames
            decay_rate: 0.05,          // 5% decay per step
            features_per_frame: 8,     // Store 8 key features per frame
        }
    }
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
    /// Extra metabolic cost per brain layer (0.0 = use default bonus system)
    #[serde(default)]
    pub brain_cost_per_layer: f32,
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

/// Food patches configuration for foraging memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoodPatchesConfig {
    pub enabled: bool,
    pub patch_count: usize,
    pub initial_capacity: f32,
    pub depletion_rate: f32,
    pub regeneration_rate: f32,
    pub regeneration_time: u64,
    pub min_distance: u8,
}

impl Default for FoodPatchesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            patch_count: 8,
            initial_capacity: 80.0,
            depletion_rate: 5.0,
            regeneration_rate: 0.5,
            regeneration_time: 200,
            min_distance: 10,
        }
    }
}

/// Behavior tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorTrackingConfig {
    pub enabled: bool,
    pub sample_rate: u64,
    pub max_tracked: usize,
}

impl Default for BehaviorTrackingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_rate: 10,
            max_tracked: 200,
        }
    }
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
            reproduction: SexualReproductionConfig::default(),
            large_prey: LargePreyConfig::default(),
            food_patches: FoodPatchesConfig::default(),
            behavior_tracking: BehaviorTrackingConfig::default(),
            procedural_environment: EnvironmentConfig::default(),
            learning: LearningConfig::default(),
            database: DatabaseConfig::default(),
            cognitive_gate: CognitiveGateConfig::default(),
            brain_tax: BrainTaxConfig::default(),
            sensory: SensoryConfig::default(),
            day_night: DayNightConfig::default(),
            memory: MemoryConfig::default(),
            dynamic_obstacles: DynamicObstacleConfig::default(),
            diversity: DiversityConfig::default(),
        }
    }
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            grid_size: 80,
            food_regen_rate: 0.5, // Was 0.3 - faster food regeneration
            food_max: 100.0,
            initial_food_density: 0.6, // Was 0.5 - more initial food
        }
    }
}

impl Default for OrganismConfig {
    fn default() -> Self {
        Self {
            initial_population: 200,
            initial_energy: 100.0,  // Was 80.0 - more starting energy
            reproduction_cost: 30.0, // Was 25.0
            reproduction_threshold: 60.0, // Was 50.0
            metabolism_base: 0.3,   // Was 0.5 - reduced base cost
            food_energy: 30.0,      // Was 25.0 - more energy from food
            move_cost: 0.2,         // Was 0.3 - cheaper movement
            brain_cost_per_layer: 0.0, // Default: use bonus system
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            // 75 inputs (legacy): 24 base + 8 spatial + 3 temporal + 3 social + 10 sequential + 12 predator + 15 cooperation
            // 95 inputs (enhanced): +8 vision + 8 olfaction + 4 audition
            n_inputs: 75,
            // 15 outputs: 4 movement + eat + reproduce + attack + signal + wait + 2 social + 3 cooperation
            n_outputs: 15,
            use_instincts: false, // Instincts prevent brain evolution
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
