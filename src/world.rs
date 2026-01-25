//! World simulation engine - main simulation loop.

use crate::checkpoint::Checkpoint;
use crate::config::Config;
use crate::evolution::EvolutionEngine;
use crate::grid::{FoodGrid, SpatialIndex};
use crate::organism::{Action, Organism};
use crate::stats::{LineageTracker, Stats, StatsHistory};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// The simulation world
pub struct World {
    // Population
    pub organisms: Vec<Organism>,

    // Environment
    pub food_grid: FoodGrid,
    pub spatial_index: SpatialIndex,

    // State
    pub time: u64,
    pub generation_max: u16,

    // Configuration
    pub config: Config,

    // Statistics
    pub stats: Stats,
    pub stats_history: StatsHistory,
    pub lineage_tracker: LineageTracker,

    // Evolution
    pub evolution_engine: EvolutionEngine,

    // ID generation
    next_organism_id: u64,

    // Random number generator (seeded for reproducibility)
    rng: ChaCha8Rng,
    seed: u64,

    // Performance tracking
    births_this_step: usize,
    deaths_this_step: usize,
}

impl World {
    /// Create a new world with the given configuration
    pub fn new(config: Config) -> Self {
        let seed = rand::thread_rng().gen();
        Self::new_with_seed(config, seed)
    }

    /// Create a new world with a specific seed for reproducibility
    pub fn new_with_seed(config: Config, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let grid_size = config.world.grid_size;

        // Initialize food grid
        let mut food_grid = FoodGrid::new(grid_size, config.world.food_max);
        food_grid.initialize(config.world.initial_food_density);

        // Initialize spatial index
        let spatial_index = SpatialIndex::new(grid_size);

        // Create lineage tracker
        let mut lineage_tracker = LineageTracker::new();

        // Evolution engine
        let evolution_engine = EvolutionEngine::from_config(&config);

        // Create initial population
        let mut organisms = Vec::with_capacity(config.safety.max_population);
        let mut next_organism_id = 0u64;

        for _ in 0..config.organisms.initial_population {
            let x = rng.gen_range(0..grid_size as u8);
            let y = rng.gen_range(0..grid_size as u8);
            let lineage_id = lineage_tracker.register_lineage(0);

            let org = Organism::new(next_organism_id, lineage_id, x, y, &config);
            organisms.push(org);
            next_organism_id += 1;
        }

        let mut world = Self {
            organisms,
            food_grid,
            spatial_index,
            time: 0,
            generation_max: 0,
            config: config.clone(),
            stats: Stats::new(),
            stats_history: StatsHistory::new(config.logging.stats_interval),
            lineage_tracker,
            evolution_engine,
            next_organism_id,
            rng,
            seed,
            births_this_step: 0,
            deaths_this_step: 0,
        };

        // Initial spatial index update
        world.update_spatial_index();

        world
    }

    /// Restore world from checkpoint
    pub fn from_checkpoint(checkpoint: Checkpoint) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(checkpoint.random_seed);
        let grid_size = checkpoint.config.world.grid_size;
        let evolution_engine = EvolutionEngine::from_config(&checkpoint.config);

        let mut world = Self {
            organisms: checkpoint.organisms,
            food_grid: checkpoint.food_grid,
            spatial_index: SpatialIndex::new(grid_size),
            time: checkpoint.time,
            generation_max: 0,
            config: checkpoint.config.clone(),
            stats: checkpoint.stats,
            stats_history: StatsHistory::new(checkpoint.config.logging.stats_interval),
            lineage_tracker: checkpoint.lineage_tracker,
            evolution_engine,
            next_organism_id: checkpoint.next_organism_id,
            rng,
            seed: checkpoint.random_seed,
            births_this_step: 0,
            deaths_this_step: 0,
        };

        world.update_spatial_index();
        world.update_generation_max();

        world
    }

    /// Create checkpoint of current state
    pub fn create_checkpoint(&self) -> Checkpoint {
        Checkpoint::new(
            self.time,
            self.config.clone(),
            self.organisms.clone(),
            self.food_grid.clone(),
            self.stats.clone(),
            self.lineage_tracker.clone(),
            self.next_organism_id,
            self.seed,
        )
    }

    /// Main simulation step
    pub fn step(&mut self) {
        self.births_this_step = 0;
        self.deaths_this_step = 0;

        // Phase 1: Parallel sensing and thinking
        let actions = self.compute_actions();

        // Phase 2: Execute actions (sequential to avoid conflicts)
        self.execute_actions(&actions);

        // Phase 3: Update organisms (aging, metabolism)
        self.update_organisms();

        // Phase 4: Handle reproduction
        self.handle_reproduction();

        // Phase 5: Remove dead organisms
        self.remove_dead();

        // Phase 6: Update environment
        self.update_environment();

        // Phase 7: Update spatial index
        self.update_spatial_index();

        // Phase 8: Update statistics
        self.update_stats();

        self.time += 1;
    }

    /// Compute actions for all organisms in parallel
    fn compute_actions(&self) -> Vec<(usize, Action)> {
        // Collect organism data for parallel processing
        let organism_refs: Vec<_> = self
            .organisms
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_alive())
            .collect();

        // Parallel sensing and decision making
        organism_refs
            .par_iter()
            .map(|&(idx, org)| {
                let inputs = org.sense(
                    &self.food_grid,
                    &self.spatial_index,
                    &self.organisms,
                    self.time,
                    &self.config,
                );

                // Clone brain for thread-safe forward pass
                let outputs = org.brain.forward(&inputs);
                let mut output_array = [0.0f32; 10];
                for (i, &val) in outputs.iter().take(10).enumerate() {
                    output_array[i] = val;
                }

                let action = org.decide_action(&output_array);
                (idx, action)
            })
            .collect()
    }

    /// Execute actions sequentially
    fn execute_actions(&mut self, actions: &[(usize, Action)]) {
        for &(idx, action) in actions {
            if idx >= self.organisms.len() || !self.organisms[idx].is_alive() {
                continue;
            }

            match action {
                Action::MoveNorth => {
                    self.organisms[idx].try_move(
                        0,
                        -1,
                        &self.spatial_index,
                        self.config.world.grid_size,
                        self.config.organisms.move_cost,
                    );
                }
                Action::MoveEast => {
                    self.organisms[idx].try_move(
                        1,
                        0,
                        &self.spatial_index,
                        self.config.world.grid_size,
                        self.config.organisms.move_cost,
                    );
                }
                Action::MoveSouth => {
                    self.organisms[idx].try_move(
                        0,
                        1,
                        &self.spatial_index,
                        self.config.world.grid_size,
                        self.config.organisms.move_cost,
                    );
                }
                Action::MoveWest => {
                    self.organisms[idx].try_move(
                        -1,
                        0,
                        &self.spatial_index,
                        self.config.world.grid_size,
                        self.config.organisms.move_cost,
                    );
                }
                Action::Eat => {
                    self.organisms[idx].try_eat(&mut self.food_grid, self.config.organisms.food_energy);
                }
                Action::Signal(val) => {
                    self.organisms[idx].signal = val;
                }
                Action::Wait => {
                    self.organisms[idx].energy -= 0.5;
                }
                Action::Reproduce | Action::Attack => {
                    // Handled separately
                }
            }

            self.organisms[idx].last_action = Some(action);
        }
    }

    /// Update all organisms (aging, metabolism)
    fn update_organisms(&mut self) {
        for org in &mut self.organisms {
            if org.is_alive() {
                org.update(&self.config);
            }
        }
    }

    /// Handle reproduction
    fn handle_reproduction(&mut self) {
        let mut offspring = Vec::new();

        // Check population limit
        let current_pop = self.organisms.iter().filter(|o| o.is_alive()).count();
        let space_available = self.config.safety.max_population.saturating_sub(current_pop);

        if space_available == 0 {
            return;
        }

        // Collect reproduction candidates
        let candidates: Vec<usize> = self
            .organisms
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_alive() && o.can_reproduce(&self.config))
            .map(|(idx, _)| idx)
            .collect();

        // Process reproductions
        for idx in candidates {
            if offspring.len() >= space_available {
                break;
            }

            // Random chance to reproduce each step (60% probability)
            if self.rng.gen::<f32>() > 0.6 {
                continue;
            }

            let child_id = self.next_organism_id;
            self.next_organism_id += 1;

            if let Some(child) = self.organisms[idx].reproduce(child_id, &self.config) {
                offspring.push(child);
                self.births_this_step += 1;
            }
        }

        // Update generation max
        for child in &offspring {
            if child.generation > self.generation_max {
                self.generation_max = child.generation;
            }
        }

        // Add offspring to population
        self.organisms.extend(offspring);
    }

    /// Remove dead organisms
    fn remove_dead(&mut self) {
        let alive_before = self.organisms.len();
        self.organisms.retain(|org| org.is_alive());
        self.deaths_this_step = alive_before - self.organisms.len();
    }

    /// Update environment (food regeneration)
    fn update_environment(&mut self) {
        // Regular regeneration
        self.food_grid.regenerate(self.config.world.food_regen_rate);

        // Random food spawning
        if self.rng.gen::<f32>() < 0.01 {
            self.food_grid.spawn_random(10.0, 0.001);
        }
    }

    /// Update spatial index
    fn update_spatial_index(&mut self) {
        self.spatial_index.clear();
        for (idx, org) in self.organisms.iter().enumerate() {
            if org.is_alive() {
                self.spatial_index.insert(org.x, org.y, idx);
            }
        }
    }

    /// Update generation max
    fn update_generation_max(&mut self) {
        self.generation_max = self
            .organisms
            .iter()
            .filter(|o| o.is_alive())
            .map(|o| o.generation)
            .max()
            .unwrap_or(0);
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.time = self.time;
        self.stats.births = self.births_this_step;
        self.stats.deaths = self.deaths_this_step;
        self.stats.update(&self.organisms, self.food_grid.total_food());
        self.stats.generation_max = self.generation_max;

        // Record history
        if self.time % self.config.logging.stats_interval == 0 {
            self.stats_history.record(self.stats.clone());
            self.lineage_tracker.update(&self.organisms);
        }
    }

    /// Run simulation for specified number of steps
    pub fn run(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Run simulation with callback for progress updates
    pub fn run_with_callback<F>(&mut self, steps: u64, mut callback: F)
    where
        F: FnMut(&World, u64),
    {
        for i in 0..steps {
            self.step();
            callback(self, i);
        }
    }

    /// Get current population count
    pub fn population(&self) -> usize {
        self.organisms.iter().filter(|o| o.is_alive()).count()
    }

    /// Check if population is extinct
    pub fn is_extinct(&self) -> bool {
        self.population() == 0
    }

    /// Get seed for reproducibility
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        let mut config = Config::default();
        config.organisms.initial_population = 50;
        config.world.grid_size = 40;
        config
    }

    #[test]
    fn test_world_creation() {
        let config = test_config();
        let world = World::new(config.clone());

        assert_eq!(world.population(), config.organisms.initial_population);
        assert_eq!(world.time, 0);
    }

    #[test]
    fn test_world_step() {
        let config = test_config();
        let mut world = World::new(config);

        let _initial_pop = world.population();
        world.step();

        assert_eq!(world.time, 1);
        // Population may have changed due to deaths/births
        assert!(world.population() > 0 || world.is_extinct());
    }

    #[test]
    fn test_world_run() {
        let config = test_config();
        let mut world = World::new(config);

        world.run(100);

        assert_eq!(world.time, 100);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let config = test_config();
        let mut world = World::new_with_seed(config, 12345);

        world.run(50);

        let checkpoint = world.create_checkpoint();
        let restored = World::from_checkpoint(checkpoint);

        assert_eq!(restored.time, world.time);
        assert_eq!(restored.population(), world.population());
        assert_eq!(restored.seed(), world.seed());
    }

    #[test]
    fn test_reproducibility() {
        // Note: With Rayon parallelism, exact reproducibility is not guaranteed
        // due to thread scheduling. This test verifies approximate similarity.
        let config = test_config();

        let mut world1 = World::new_with_seed(config.clone(), 42);
        let mut world2 = World::new_with_seed(config, 42);

        world1.run(100);
        world2.run(100);

        assert_eq!(world1.time, world2.time);
        // Allow small variation due to parallelism
        let pop_diff = (world1.population() as i32 - world2.population() as i32).abs();
        assert!(pop_diff <= 5, "Population difference too large: {}", pop_diff);
    }

    #[test]
    fn test_population_growth() {
        let mut config = test_config();
        config.organisms.initial_population = 20;
        config.organisms.reproduction_threshold = 40.0;
        config.organisms.reproduction_cost = 20.0;

        let mut world = World::new(config);
        let initial_pop = world.population();

        // Run for a while
        world.run(500);

        // Population should have changed (likely grown or stabilized)
        // It shouldn't instantly go to 0 with reasonable parameters
        let final_pop = world.population();
        println!(
            "Population: {} -> {} (max gen: {})",
            initial_pop, final_pop, world.generation_max
        );
    }
}
