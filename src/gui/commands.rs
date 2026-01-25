//! Commands for controlling the simulation from the GUI.

/// Commands sent from GUI to simulation thread
#[derive(Debug, Clone)]
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
    /// Shutdown the simulation thread
    Shutdown,
}

/// Simulation settings that can be modified from GUI
#[derive(Debug, Clone)]
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
    }
}

/// Current simulation state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
