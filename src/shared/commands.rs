//! Commands for controlling the simulation from the GUI/Web UI.

use serde::{Deserialize, Serialize};

/// Commands sent from GUI/Web UI to simulation thread
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimCommand {
    /// Pause the simulation
    Pause,
    /// Resume the simulation
    Resume,
    /// Execute a single step
    Step,
    /// Set simulation speed multiplier (0.1 - 10.0)
    SetSpeed(f32),
    /// Select an organism by ID for detailed view
    SelectOrganism(Option<u64>),
    /// Reset simulation with current config
    Reset,
    /// Reset simulation with new settings
    ResetWithSettings(SimSettings),
    /// Save checkpoint manually
    SaveCheckpoint,
    /// Load checkpoint from file
    LoadCheckpoint(String),
    /// Set checkpoint directory
    SetCheckpointDir(String),
    /// Shutdown the simulation thread
    Shutdown,
}

/// Simulation settings that can be modified from GUI/Web UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimSettings {
    /// Maximum population limit
    pub max_population: usize,
    /// Maximum simulation steps (0 = unlimited)
    pub max_steps: u64,
    /// Initial population
    pub initial_population: usize,
    /// Grid size
    pub grid_size: usize,
    /// Mutation rate
    pub mutation_rate: f32,
    /// Mutation strength
    pub mutation_strength: f32,
    /// Food regeneration rate
    pub food_regen_rate: f32,
    /// Reproduction energy threshold
    pub reproduction_threshold: f32,
    /// Enable predation
    pub predation_enabled: bool,
    /// Enable seasons
    pub seasons_enabled: bool,
    /// Enable terrain
    pub terrain_enabled: bool,
    /// Enable lifetime learning (Hebbian)
    pub learning_enabled: bool,
    /// Learning rate for Hebbian updates
    pub learning_rate: f32,
    /// Enable diversity mechanisms (anti-bottleneck)
    pub diversity_enabled: bool,
    /// Enable database logging
    pub database_enabled: bool,
    /// Enable cognitive gate
    pub cognitive_gate_enabled: bool,
    /// Enable food patches
    pub food_patches_enabled: bool,
    /// Enable enhanced senses (requires n_inputs=95)
    pub enhanced_senses: bool,
    /// Number of neural inputs
    pub n_inputs: usize,
}

impl Default for SimSettings {
    fn default() -> Self {
        Self {
            max_population: 5000,
            max_steps: 0,
            initial_population: 200,
            grid_size: 80,
            mutation_rate: 0.05,
            mutation_strength: 0.3,
            food_regen_rate: 0.3,
            reproduction_threshold: 50.0,
            predation_enabled: true,
            seasons_enabled: true,
            terrain_enabled: true,
            learning_enabled: false,
            learning_rate: 0.001,
            diversity_enabled: false,
            database_enabled: false,
            cognitive_gate_enabled: false,
            food_patches_enabled: false,
            enhanced_senses: false,
            n_inputs: 75,
        }
    }
}

impl SimSettings {
    /// Create settings from a Config
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self {
            max_population: config.safety.max_population,
            max_steps: 0,
            initial_population: config.organisms.initial_population,
            grid_size: config.world.grid_size,
            mutation_rate: config.evolution.mutation_rate,
            mutation_strength: config.evolution.mutation_strength,
            food_regen_rate: config.world.food_regen_rate,
            reproduction_threshold: config.organisms.reproduction_threshold,
            predation_enabled: config.predation.enabled,
            seasons_enabled: config.seasons.enabled,
            terrain_enabled: config.terrain.enabled,
            learning_enabled: config.learning.enabled,
            learning_rate: config.learning.learning_rate,
            diversity_enabled: config.diversity.enabled,
            database_enabled: config.database.enabled,
            cognitive_gate_enabled: config.cognitive_gate.enabled,
            food_patches_enabled: config.food_patches.enabled,
            enhanced_senses: config.sensory.enhanced_senses,
            n_inputs: config.neural.n_inputs,
        }
    }

    /// Apply settings to a Config
    pub fn apply_to_config(&self, config: &mut crate::config::Config) {
        config.safety.max_population = self.max_population;
        config.organisms.initial_population = self.initial_population;
        config.world.grid_size = self.grid_size;
        config.evolution.mutation_rate = self.mutation_rate;
        config.evolution.mutation_strength = self.mutation_strength;
        config.world.food_regen_rate = self.food_regen_rate;
        config.organisms.reproduction_threshold = self.reproduction_threshold;
        config.predation.enabled = self.predation_enabled;
        config.seasons.enabled = self.seasons_enabled;
        config.terrain.enabled = self.terrain_enabled;
        config.learning.enabled = self.learning_enabled;
        config.learning.learning_rate = self.learning_rate;
        config.diversity.enabled = self.diversity_enabled;
        config.database.enabled = self.database_enabled;
        config.cognitive_gate.enabled = self.cognitive_gate_enabled;
        config.food_patches.enabled = self.food_patches_enabled;
        config.sensory.enhanced_senses = self.enhanced_senses;
        config.neural.n_inputs = self.n_inputs;
    }
}

/// Current simulation state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimState {
    /// Simulation is running
    Running,
    /// Simulation is paused
    Paused,
    /// Simulation has stopped (extinct or shutdown)
    Stopped,
}

impl Default for SimState {
    fn default() -> Self {
        Self::Paused
    }
}
