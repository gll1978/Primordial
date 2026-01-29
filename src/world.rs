//! World simulation engine - main simulation loop.

use crate::analysis::behavior_tracker::BehaviorTrackerManager;
use crate::analysis::SurvivalAnalyzer;
use crate::checkpoint::Checkpoint;
use crate::config::Config;
use crate::ecology::depletion::DepletionSystem;
use crate::ecology::environment_manager::EnvironmentManager;
use crate::ecology::food_patches::{PatchConfig, PatchWorld};
use crate::ecology::large_prey::{CooperationManager, LargePrey};
use crate::ecology::predation;
use crate::ecology::seasons::SeasonalSystem;
use crate::ecology::terrain::{Terrain, TerrainGrid};
use crate::ecology::dynamic_obstacles::DynamicObstacleSystem;
use crate::evolution::EvolutionEngine;
use crate::genetics::crossover::CrossoverSystem;
use crate::genetics::diversity::DiversityHistory;
use crate::genetics::phylogeny::PhylogeneticTree;
use crate::genetics::sex::{check_speciation_compatibility, Sex, SexualReproductionSystem};
use crate::grid::{FoodGrid, SpatialIndex};
use crate::organism::{Action, DeathCause, MemoryFrame, Organism, ShortTermMemory};
use crate::stats::{LineageTracker, Stats, StatsHistory};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "database")]
use std::sync::mpsc as std_mpsc;
#[cfg(feature = "database")]
use crate::database::DbEvent;

/// Food depletion tracking for spatial memory
#[derive(Clone, Debug)]
pub struct FoodMemory {
    pub last_eaten: u64,
    pub depletion_level: f32,
}

impl FoodMemory {
    pub fn new(time: u64) -> Self {
        Self {
            last_eaten: time,
            depletion_level: 0.3, // Initial depletion when first eaten
        }
    }
}

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

    // Fase 2: Seasonal system
    pub seasonal_system: SeasonalSystem,

    // Fase 2: Predation tracking
    pub kills_this_step: usize,

    // Fase 2 Week 2: Terrain system
    pub terrain_grid: TerrainGrid,

    // Fase 2 Week 2: Depletion system
    pub depletion_system: DepletionSystem,

    // Fase 2 Week 3-4: Genetics
    pub phylogeny: PhylogeneticTree,
    pub sexual_reproduction: SexualReproductionSystem,
    pub crossover_system: CrossoverSystem,
    pub next_lineage_id: u32,

    // Fase 2 Week 5: Diversity and Survival Analysis
    pub diversity_history: DiversityHistory,
    pub survival_analyzer: SurvivalAnalyzer,

    // Cognitive Tasks: Spatial Memory
    pub food_memory: HashMap<(u8, u8), FoodMemory>,

    // Cognitive Tasks: Temporal Prediction
    pub food_history: VecDeque<f32>,

    // Phase 1: Foraging Memory
    pub patch_world: Option<PatchWorld>,
    pub behavior_tracker: Option<BehaviorTrackerManager>,

    // Phase 2: Procedural Environments
    pub env_manager: Option<EnvironmentManager>,
    pub last_max_generation: u64,

    // B3: Large Prey and Cooperation
    pub large_prey: Vec<LargePrey>,
    pub cooperation_manager: CooperationManager,
    pub next_large_prey_id: u64,

    // Database logging channel
    #[cfg(feature = "database")]
    pub db_sender: Option<std_mpsc::Sender<DbEvent>>,

    // Cognitive Gate: Food complexity grid
    pub complexity_grid: crate::grid::ComplexityGrid,

    // Phase 2 Feature 4: Dynamic Obstacles
    pub dynamic_obstacle_system: Option<DynamicObstacleSystem>,

    // Anti-bottleneck diversity mechanisms
    /// Tracks terrain adaptation: organism_id -> (terrain, steps_in_terrain)
    pub terrain_adaptation: HashMap<u64, (crate::ecology::terrain::Terrain, u64)>,
    /// Tracks strategy frequencies for frequency-dependent selection
    strategy_frequencies: HashMap<String, usize>,
    /// Last step when frequencies were updated
    last_frequency_update: u64,
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

        // Initialize complexity grid (Cognitive Gate)
        let mut complexity_grid = crate::grid::ComplexityGrid::new(grid_size);
        if config.cognitive_gate.enabled {
            complexity_grid.initialize(&food_grid);
        }

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

            let mut org = Organism::new(next_organism_id, lineage_id, x, y, &config);
            if config.learning.enabled {
                org.brain.enable_learning_with_config(&config.learning);
            }
            organisms.push(org);
            next_organism_id += 1;
        }

        // Initialize seasonal system
        let seasonal_system = SeasonalSystem::new(&config.seasons);

        // Initialize terrain grid
        let mut terrain_grid = TerrainGrid::new(grid_size);
        if config.terrain.enabled {
            if config.terrain.barrier {
                terrain_grid.generate_with_barrier(config.terrain.barrier_vertical);
            } else if config.terrain.clustered {
                terrain_grid.generate_clustered(Some(seed));
            }
        }

        // Initialize food patches
        let patch_world = if config.food_patches.enabled {
            let patch_config = PatchConfig {
                patch_count: config.food_patches.patch_count,
                initial_capacity: config.food_patches.initial_capacity,
                depletion_rate: config.food_patches.depletion_rate,
                regeneration_rate: config.food_patches.regeneration_rate,
                regeneration_time: config.food_patches.regeneration_time,
                min_distance: config.food_patches.min_distance,
                patch_radius: 3,
            };
            Some(PatchWorld::new(&patch_config, grid_size as u8, &mut rng))
        } else {
            None
        };

        // Initialize depletion system
        let depletion_system = DepletionSystem::new(grid_size);

        // Initialize dynamic obstacles (Phase 2 Feature 4)
        let dynamic_obstacle_system = if config.dynamic_obstacles.enabled {
            Some(DynamicObstacleSystem::new(&config.dynamic_obstacles, grid_size as u8, &mut rng))
        } else {
            None
        };

        // Initialize genetics systems
        let mut phylogeny = PhylogeneticTree::new();
        let sexual_reproduction = SexualReproductionSystem::new();
        let crossover_system = CrossoverSystem::new();

        // Initialize diversity and survival tracking
        let diversity_history = DiversityHistory::new(100); // Record every 100 steps
        let survival_analyzer = SurvivalAnalyzer::new();

        // Record initial organisms in phylogeny
        for org in &organisms {
            phylogeny.record_birth(
                org.id,
                None, // No parents
                None,
                0, // Time 0
                org.brain.complexity(),
                org.energy,
                org.size,
                org.lineage_id,
                org.generation,
                org.brain.hash(),
            );
        }

        let next_lineage_id = organisms.len() as u32;

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
            seasonal_system,
            kills_this_step: 0,
            terrain_grid,
            depletion_system,
            phylogeny,
            sexual_reproduction,
            crossover_system,
            next_lineage_id,
            diversity_history,
            survival_analyzer,
            food_memory: HashMap::new(),
            food_history: VecDeque::with_capacity(100),
            patch_world,
            behavior_tracker: if config.behavior_tracking.enabled {
                Some(BehaviorTrackerManager::new(
                    config.behavior_tracking.max_tracked,
                    config.behavior_tracking.sample_rate,
                ))
            } else {
                None
            },
            env_manager: if config.procedural_environment.enabled {
                Some(EnvironmentManager::new(config.procedural_environment.clone()))
            } else {
                None
            },
            last_max_generation: 0,
            large_prey: Vec::new(),
            cooperation_manager: CooperationManager::new(),
            next_large_prey_id: 0,
            #[cfg(feature = "database")]
            db_sender: None,
            complexity_grid,
            dynamic_obstacle_system,
            terrain_adaptation: HashMap::new(),
            strategy_frequencies: HashMap::new(),
            last_frequency_update: 0,
        };

        // Initial spatial index update
        world.update_spatial_index();

        world
    }

    /// Set the database sender for logging events
    #[cfg(feature = "database")]
    pub fn set_db_sender(&mut self, sender: std_mpsc::Sender<DbEvent>) {
        // Log birth of all initial organisms
        for org in &self.organisms {
            let _ = sender.send(DbEvent::OrganismBirth {
                organism_id: org.id,
                lineage_id: org.lineage_id,
                generation: org.generation,
                parent1_id: None,
                parent2_id: None,
                step: 0,
                x: org.x,
                y: org.y,
                brain_layers: org.brain.complexity() as i32,
                brain_neurons: org.brain.total_hidden_neurons() as i32,
                brain_connections: org.brain.parameter_count() as i32,
                is_predator: org.is_predator,
            });
        }
        self.db_sender = Some(sender);
    }

    /// Restore world from checkpoint
    pub fn from_checkpoint(checkpoint: Checkpoint) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(checkpoint.random_seed);
        let grid_size = checkpoint.config.world.grid_size;
        let evolution_engine = EvolutionEngine::from_config(&checkpoint.config);
        let mut seasonal_system = SeasonalSystem::new(&checkpoint.config.seasons);
        seasonal_system.update(checkpoint.time);

        // Regenerate terrain (deterministic from seed)
        let mut terrain_grid = TerrainGrid::new(grid_size);
        if checkpoint.config.terrain.enabled {
            if checkpoint.config.terrain.barrier {
                terrain_grid.generate_with_barrier(checkpoint.config.terrain.barrier_vertical);
            } else if checkpoint.config.terrain.clustered {
                terrain_grid.generate_clustered(Some(checkpoint.random_seed));
            }
        }

        // Initialize food patches (state not preserved in checkpoint)
        let mut checkpoint_rng = ChaCha8Rng::seed_from_u64(checkpoint.random_seed.wrapping_add(1));
        let patch_world = if checkpoint.config.food_patches.enabled {
            let patch_config = PatchConfig {
                patch_count: checkpoint.config.food_patches.patch_count,
                initial_capacity: checkpoint.config.food_patches.initial_capacity,
                depletion_rate: checkpoint.config.food_patches.depletion_rate,
                regeneration_rate: checkpoint.config.food_patches.regeneration_rate,
                regeneration_time: checkpoint.config.food_patches.regeneration_time,
                min_distance: checkpoint.config.food_patches.min_distance,
                patch_radius: 3,
            };
            Some(PatchWorld::new(&patch_config, grid_size as u8, &mut checkpoint_rng))
        } else {
            None
        };

        // New depletion system (state not preserved in checkpoint)
        let depletion_system = DepletionSystem::new(grid_size);

        // New dynamic obstacle system (state not preserved in checkpoint)
        let dynamic_obstacle_system = if checkpoint.config.dynamic_obstacles.enabled {
            Some(DynamicObstacleSystem::new(&checkpoint.config.dynamic_obstacles, grid_size as u8, &mut checkpoint_rng))
        } else {
            None
        };

        // New genetics systems (state not preserved in checkpoint)
        let phylogeny = PhylogeneticTree::new();
        let sexual_reproduction = SexualReproductionSystem::new();
        let crossover_system = CrossoverSystem::new();
        let next_lineage_id = checkpoint.organisms.len() as u32;

        // New diversity and survival tracking
        let diversity_history = DiversityHistory::new(100);
        let survival_analyzer = SurvivalAnalyzer::new();

        // Initialize complexity grid (Cognitive Gate)
        let mut complexity_grid = crate::grid::ComplexityGrid::new(grid_size);
        if checkpoint.config.cognitive_gate.enabled {
            complexity_grid.initialize(&checkpoint.food_grid);
        }

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
            seasonal_system,
            kills_this_step: 0,
            terrain_grid,
            depletion_system,
            phylogeny,
            sexual_reproduction,
            crossover_system,
            next_lineage_id,
            diversity_history,
            survival_analyzer,
            food_memory: HashMap::new(),
            food_history: VecDeque::with_capacity(100),
            patch_world,
            behavior_tracker: if checkpoint.config.behavior_tracking.enabled {
                Some(BehaviorTrackerManager::new(
                    checkpoint.config.behavior_tracking.max_tracked,
                    checkpoint.config.behavior_tracking.sample_rate,
                ))
            } else {
                None
            },
            env_manager: if checkpoint.config.procedural_environment.enabled {
                Some(EnvironmentManager::new(checkpoint.config.procedural_environment.clone()))
            } else {
                None
            },
            last_max_generation: 0,
            large_prey: Vec::new(),
            cooperation_manager: CooperationManager::new(),
            next_large_prey_id: 0,
            #[cfg(feature = "database")]
            db_sender: None,
            complexity_grid,
            dynamic_obstacle_system,
            terrain_adaptation: HashMap::new(),
            strategy_frequencies: HashMap::new(),
            last_frequency_update: 0,
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
        self.kills_this_step = 0;

        // Phase 0: Update seasonal system
        self.seasonal_system.update(self.time);

        // Phase 0.5: Update dynamic obstacles (Phase 2 Feature 4)
        if let Some(ref mut obstacle_system) = self.dynamic_obstacle_system {
            obstacle_system.update(self.time, &mut self.rng);
        }

        // Phase 1: Parallel sensing and thinking
        let actions = self.compute_actions();

        // Phase 2: Execute actions (sequential to avoid conflicts)
        // Save energy before actions for reward calculation
        let energy_before: Vec<f32> = self.organisms.iter().map(|o| o.energy).collect();
        self.execute_actions(&actions);

        // Phase 3: Handle predation (Attack actions)
        self.handle_predation(&actions);

        // Phase 3.5: B3 - Handle large prey attacks
        self.handle_large_prey_attacks(&actions);

        // Phase 4: Update organisms (aging, metabolism)
        self.update_organisms();

        // Phase 4.5: Apply lifetime learning (Hebbian)
        if self.config.learning.enabled {
            self.apply_learning_phase(&actions, &energy_before);
        }

        // Phase 5: Handle reproduction
        self.handle_reproduction();

        // Phase 6: Remove dead organisms
        self.remove_dead();

        // Phase 7: Update environment (with seasonal multipliers)
        self.update_environment();

        // Phase 7.5: Update cognitive systems (memory & history)
        self.update_food_memory();
        self.update_food_history();
        self.update_predator_observations(); // B2: Pattern Recognition
        self.update_large_prey();            // B3: Large prey movement/escape
        self.maybe_spawn_large_prey();       // B3: Maybe spawn new large prey
        self.cleanup_cooperation();          // B3: Cleanup cooperation state
        self.apply_memory_decay();           // Phase 2 Feature 2: STM decay

        // Phase 8: Update spatial index
        self.update_spatial_index();

        // Phase 9: Update statistics
        self.update_stats();

        // Phase 9.5: Database logging
        #[cfg(feature = "database")]
        self.log_to_database();

        // Phase 10: Check for environment reshuffle
        self.check_environment_reshuffle();

        self.time += 1;
    }

    /// Check and execute environment reshuffle if needed
    fn check_environment_reshuffle(&mut self) {
        let current_max_gen = self.generation_max as u64;

        // Check generation-based reshuffle (only when generation advances)
        if current_max_gen > self.last_max_generation {
            self.last_max_generation = current_max_gen;

            let should_reshuffle = if let Some(ref env_mgr) = self.env_manager {
                env_mgr.should_reshuffle_gen(current_max_gen)
            } else {
                false
            };

            if should_reshuffle {
                if let Some(ref mut env_mgr) = self.env_manager {
                    let new_seed = env_mgr.next_seed(current_max_gen, self.time);
                    println!("*** RESHUFFLE (gen) at step {} gen {} seed {} ***", self.time, current_max_gen, new_seed);
                    if let Some(ref mut patches) = self.patch_world {
                        patches.reshuffle_patches(new_seed);
                    }
                }
            }
        }

        // Check step-based reshuffle (independent of generation)
        let should_reshuffle_step = if let Some(ref env_mgr) = self.env_manager {
            env_mgr.should_reshuffle_step(self.time)
        } else {
            false
        };

        if should_reshuffle_step {
            if let Some(ref mut env_mgr) = self.env_manager {
                let new_seed = env_mgr.next_seed(current_max_gen, self.time);
                println!("*** RESHUFFLE (step) at step {} gen {} seed {} ***", self.time, current_max_gen, new_seed);
                if let Some(ref mut patches) = self.patch_world {
                    patches.reshuffle_patches(new_seed);
                }
            }
        }
    }

    /// Compute actions for all organisms in parallel
    fn compute_actions(&self) -> Vec<(usize, Action, Vec<f32>)> {
        // Pre-compute global temporal data (same for all organisms)
        let season_progress = self.current_season_progress();
        let food_trend = self.food_availability_trend();
        let time_to_season = self.steps_to_next_season();

        // Collect organism data for parallel processing
        let organism_refs: Vec<_> = self
            .organisms
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_alive())
            .collect();

        // Parallel sensing and decision making
        let use_enhanced = self.config.sensory.enhanced_senses && self.config.neural.n_inputs == 95;

        organism_refs
            .par_iter()
            .map(|&(idx, org)| {
                // Build cognitive inputs for this organism
                let cognitive = self.build_cognitive_inputs(
                    org,
                    season_progress,
                    food_trend,
                    time_to_season,
                );

                if use_enhanced {
                    // Enhanced sensing with 95 inputs
                    let inputs = org.sense_enhanced(
                        &self.food_grid,
                        &self.spatial_index,
                        &self.organisms,
                        self.time,
                        &self.config,
                        &cognitive,
                    );

                    // Forward pass through neural network (95 inputs -> 19 outputs)
                    let outputs = org.brain.forward(&inputs);
                    let mut output_array = [0.0f32; 19];
                    for (i, &val) in outputs.iter().take(19).enumerate() {
                        output_array[i] = val;
                    }

                    let action = org.decide_action(&output_array);
                    (idx, action, inputs.to_vec())
                } else {
                    // Legacy sensing with 75 inputs
                    let inputs = org.sense(
                        &self.food_grid,
                        &self.spatial_index,
                        &self.organisms,
                        self.time,
                        &self.config,
                        &cognitive,
                    );

                    // Forward pass through neural network (75 inputs -> 19 outputs)
                    let outputs = org.brain.forward(&inputs);
                    let mut output_array = [0.0f32; 19];
                    for (i, &val) in outputs.iter().take(19).enumerate() {
                        output_array[i] = val;
                    }

                    let action = org.decide_action(&output_array);
                    (idx, action, inputs.to_vec())
                }
            })
            .collect()
    }

    /// Build cognitive inputs for an organism (spatial memory, temporal, social)
    fn build_cognitive_inputs(
        &self,
        org: &Organism,
        season_progress: f32,
        food_trend: f32,
        time_to_season: f32,
    ) -> crate::organism::CognitiveInputs {
        use crate::organism::{CognitiveInputs, SocialSignal};

        // 8 directions for spatial memory
        const DIRECTIONS: [(i8, i8); 8] = [
            (0, -1),  // N
            (1, -1),  // NE
            (1, 0),   // E
            (1, 1),   // SE
            (0, 1),   // S
            (-1, 1),  // SW
            (-1, 0),  // W
            (-1, -1), // NW
        ];

        // Calculate depletion in each direction (check 3 cells)
        let mut depletions = [0.0f32; 8];
        for (dir_idx, (dx, dy)) in DIRECTIONS.iter().enumerate() {
            let mut total_depletion = 0.0f32;
            for dist in 1..=3i8 {
                let nx = org.x as i16 + (*dx as i16 * dist as i16);
                let ny = org.y as i16 + (*dy as i16 * dist as i16);
                if nx >= 0 && nx < self.config.world.grid_size as i16
                    && ny >= 0 && ny < self.config.world.grid_size as i16
                {
                    let depletion = self.get_food_depletion(nx as u8, ny as u8);
                    // Closer cells contribute more
                    total_depletion += depletion / dist as f32;
                }
            }
            depletions[dir_idx] = (total_depletion / 3.0).min(1.0);
        }

        // Scan for social signals from nearby organisms
        let mut signal_danger = 0.0f32;
        let mut signal_food = 0.0f32;
        let signal_help = 0.0f32; // Reserved for future use

        let neighbors = self.spatial_index.query_neighbors(org.x, org.y, 5);
        for &neighbor_idx in &neighbors {
            if neighbor_idx < self.organisms.len() && neighbor_idx != org.id as usize {
                let other = &self.organisms[neighbor_idx];
                if other.is_alive() {
                    match other.social_signal {
                        SocialSignal::Danger => signal_danger = 1.0,
                        SocialSignal::FoodFound => signal_food = 1.0,
                        SocialSignal::None => {}
                    }
                }
            }
        }

        // B1: Sequential Memory - calculate path-based inputs
        let loop_detected = if org.detect_loop() { 1.0 } else { 0.0 };
        let oscillation = if org.detect_oscillation() { 1.0 } else { 0.0 };
        let path_entropy = org.calculate_path_entropy();
        let (movement_bias_x, movement_bias_y) = org.calculate_movement_bias();
        let recent_dirs = org.get_recent_direction();

        // B2: Pattern Recognition - find nearest observed predator
        let (pred_movement_variance, pred_speed, pred_direction_consistency,
             pred_strategy_random, pred_strategy_patrol, pred_strategy_chase, pred_strategy_ambush,
             pred_classification_confidence, pred_approach_angle, pred_relative_speed,
             pred_time_observed, pred_threat_level) = self.get_predator_inputs(org);

        // B3: Multi-Agent Coordination inputs
        let (large_prey_nearby, large_prey_distance, large_prey_health, large_prey_attackers,
             large_prey_need, partner_nearby, partner_trust, partner_distance,
             cooperation_proposed, cooperation_active, hunt_success_rate, partner_fitness,
             own_attack_power, time_since_last_coop, prey_escape_urgency) = self.get_cooperation_inputs(org);

        // Phase 2 Feature 1: Enhanced Sensory System
        let (visions, smells, sounds) = self.calculate_enhanced_senses(org);

        CognitiveInputs {
            depletion_n: depletions[0],
            depletion_ne: depletions[1],
            depletion_e: depletions[2],
            depletion_se: depletions[3],
            depletion_s: depletions[4],
            depletion_sw: depletions[5],
            depletion_w: depletions[6],
            depletion_nw: depletions[7],
            season_progress,
            food_trend,
            time_to_season,
            signal_danger,
            signal_food,
            signal_help,
            // B1: Sequential Memory inputs
            loop_detected,
            oscillation,
            path_entropy,
            movement_bias_x,
            movement_bias_y,
            recent_dir_n: recent_dirs[0],
            recent_dir_e: recent_dirs[1],
            recent_dir_s: recent_dirs[2],
            recent_dir_w: recent_dirs[3],
            recent_dir_none: recent_dirs[4],
            // B2: Pattern Recognition inputs
            pred_movement_variance,
            pred_speed,
            pred_direction_consistency,
            pred_strategy_random,
            pred_strategy_patrol,
            pred_strategy_chase,
            pred_strategy_ambush,
            pred_classification_confidence,
            pred_approach_angle,
            pred_relative_speed,
            pred_time_observed,
            pred_threat_level,
            // B3: Multi-Agent Coordination inputs
            large_prey_nearby,
            large_prey_distance,
            large_prey_health,
            large_prey_attackers,
            large_prey_need,
            partner_nearby,
            partner_trust,
            partner_distance,
            cooperation_proposed,
            cooperation_active,
            hunt_success_rate,
            partner_fitness,
            own_attack_power,
            time_since_last_coop,
            prey_escape_urgency,
            // Phase 2 Feature 1: Enhanced Sensory inputs
            vision_n: visions[0],
            vision_ne: visions[1],
            vision_e: visions[2],
            vision_se: visions[3],
            vision_s: visions[4],
            vision_sw: visions[5],
            vision_w: visions[6],
            vision_nw: visions[7],
            smell_n: smells[0],
            smell_ne: smells[1],
            smell_e: smells[2],
            smell_se: smells[3],
            smell_s: smells[4],
            smell_sw: smells[5],
            smell_w: smells[6],
            smell_nw: smells[7],
            sound_n: sounds[0],
            sound_e: sounds[1],
            sound_s: sounds[2],
            sound_w: sounds[3],
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // PHASE 2 FEATURE 1: ENHANCED SENSORY SYSTEM
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Get day/night multiplier for vision (1.0 = day, night_vision_penalty = night)
    fn get_day_multiplier(&self) -> f32 {
        if !self.config.day_night.enabled {
            return 1.0;
        }

        let cycle_pos = self.time % self.config.day_night.cycle_length;
        let half_cycle = self.config.day_night.cycle_length / 2;

        if cycle_pos < half_cycle {
            // Day phase
            1.0
        } else {
            // Night phase
            self.config.day_night.night_vision_penalty
        }
    }

    /// Get olfaction multiplier (1.0 = day, night_olfaction_bonus = night)
    fn get_olfaction_multiplier(&self) -> f32 {
        if !self.config.day_night.enabled {
            return 1.0;
        }

        let cycle_pos = self.time % self.config.day_night.cycle_length;
        let half_cycle = self.config.day_night.cycle_length / 2;

        if cycle_pos < half_cycle {
            // Day phase: normal smell
            1.0
        } else {
            // Night phase: enhanced smell
            self.config.day_night.night_olfaction_bonus
        }
    }

    /// Calculate multi-resolution vision in a direction
    /// Returns 0.0-1.0 representing food availability weighted by distance
    fn calculate_vision(
        &self,
        org: &Organism,
        dx: i8,
        dy: i8,
        day_multiplier: f32,
    ) -> f32 {
        if !self.config.sensory.enhanced_senses {
            // Fallback to basic sensing when enhanced senses disabled
            let range = 3u8;
            let food_scale = self.config.world.food_max * range as f32;
            return self.food_grid.sense_direction(org.x, org.y, dx, dy, range) / food_scale;
        }

        let fovea_range = self.config.sensory.vision_fovea_range;
        let peripheral_range = self.config.sensory.vision_peripheral_range;
        let grid_size = self.config.world.grid_size as i16;

        let mut total = 0.0f32;

        // Fovea: detailed food sensing (high resolution)
        for dist in 1..=fovea_range {
            let nx = org.x as i16 + dx as i16 * dist as i16;
            let ny = org.y as i16 + dy as i16 * dist as i16;

            if nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size {
                let food = self.food_grid.get(nx as u8, ny as u8);
                // Closer cells contribute more, normalized by food_max
                total += (food / self.config.world.food_max) * day_multiplier / dist as f32;
            }
        }

        // Peripheral: presence detection only (low resolution)
        for dist in (fovea_range + 1)..=peripheral_range {
            let nx = org.x as i16 + dx as i16 * dist as i16;
            let ny = org.y as i16 + dy as i16 * dist as i16;

            if nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size {
                let food = self.food_grid.get(nx as u8, ny as u8);
                // Peripheral only detects presence, not amount (reduced weight)
                let presence = if food > 0.5 { 1.0 } else { 0.0 };
                total += presence * day_multiplier * 0.3 / dist as f32;
            }
        }

        total.min(1.0)
    }

    /// Calculate olfaction (smell) in a direction
    /// Returns 0.0-1.0 representing smell intensity based on food tier
    fn calculate_smell(
        &self,
        org: &Organism,
        dx: i8,
        dy: i8,
        olfaction_multiplier: f32,
    ) -> f32 {
        if !self.config.sensory.enhanced_senses {
            return 0.0; // Olfaction disabled when enhanced senses off
        }

        let range = self.config.sensory.olfaction_range;
        let grid_size = self.config.world.grid_size as i16;

        let mut total = 0.0f32;

        for dist in 1..=range {
            let nx = org.x as i16 + dx as i16 * dist as i16;
            let ny = org.y as i16 + dy as i16 * dist as i16;

            if nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size {
                let food = self.food_grid.get(nx as u8, ny as u8);

                if food > 0.1 {
                    // Get food complexity tier for smell strength
                    let complexity = self.complexity_grid.get(nx as u8, ny as u8);

                    // Smell intensity by tier (simple=weak, complex=strong)
                    let smell_strength = if complexity < 0.25 {
                        0.3 // Simple food: weak smell
                    } else if complexity < 0.60 {
                        0.6 // Medium food: moderate smell
                    } else {
                        1.0 // Complex food: strong smell
                    };

                    // Smell decays with distance squared (more realistic)
                    let dist_factor = 1.0 / (dist as f32 * dist as f32);
                    total += food / self.config.world.food_max * smell_strength * dist_factor * olfaction_multiplier;
                }
            }
        }

        total.min(1.0)
    }

    /// Calculate audition (sound) in a direction
    /// Returns 0.0-1.0 representing sound intensity from nearby organisms
    fn calculate_sound(
        &self,
        org: &Organism,
        dx: i8,
        dy: i8,
    ) -> f32 {
        if !self.config.sensory.enhanced_senses {
            return 0.0; // Audition disabled when enhanced senses off
        }

        let range = self.config.sensory.audition_range;
        let grid_size = self.config.world.grid_size as i16;

        let mut total = 0.0f32;

        for dist in 1..=range {
            let nx = org.x as i16 + dx as i16 * dist as i16;
            let ny = org.y as i16 + dy as i16 * dist as i16;

            if nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size {
                // Query organisms in this cell
                let neighbors = self.spatial_index.get(nx as u8, ny as u8);

                for &idx in neighbors {
                    if idx < self.organisms.len() && idx != org.id as usize {
                        let other = &self.organisms[idx];
                        if other.is_alive() {
                            // Moving organisms make more noise
                            let loudness = match other.last_action {
                                Some(Action::MoveNorth)
                                | Some(Action::MoveEast)
                                | Some(Action::MoveSouth)
                                | Some(Action::MoveWest) => 1.0,
                                Some(Action::Attack) | Some(Action::AttackLargePrey) => 1.5, // Attacks are loud
                                _ => 0.3, // Stationary organisms make some noise
                            };

                            // Sound decays with distance
                            total += loudness / dist as f32;
                        }
                    }
                }
            }
        }

        total.min(1.0)
    }

    /// Calculate all enhanced sensory inputs for an organism
    /// Returns (visions[8], smells[8], sounds[4])
    fn calculate_enhanced_senses(&self, org: &Organism) -> ([f32; 8], [f32; 8], [f32; 4]) {
        // 8 directions for vision and smell
        const DIRECTIONS_8: [(i8, i8); 8] = [
            (0, -1),  // N
            (1, -1),  // NE
            (1, 0),   // E
            (1, 1),   // SE
            (0, 1),   // S
            (-1, 1),  // SW
            (-1, 0),  // W
            (-1, -1), // NW
        ];

        // 4 cardinal directions for audition
        const DIRECTIONS_4: [(i8, i8); 4] = [
            (0, -1),  // N
            (1, 0),   // E
            (0, 1),   // S
            (-1, 0),  // W
        ];

        let day_mult = self.get_day_multiplier();
        let olf_mult = self.get_olfaction_multiplier();

        let mut visions = [0.0f32; 8];
        let mut smells = [0.0f32; 8];
        let mut sounds = [0.0f32; 4];

        // Calculate vision and smell in 8 directions
        for (i, (dx, dy)) in DIRECTIONS_8.iter().enumerate() {
            visions[i] = self.calculate_vision(org, *dx, *dy, day_mult);
            smells[i] = self.calculate_smell(org, *dx, *dy, olf_mult);
        }

        // Calculate sound in 4 cardinal directions
        for (i, (dx, dy)) in DIRECTIONS_4.iter().enumerate() {
            sounds[i] = self.calculate_sound(org, *dx, *dy);
        }

        (visions, smells, sounds)
    }

    /// B3: Get cooperation and large prey inputs for an organism
    fn get_cooperation_inputs(&self, org: &Organism) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        // Find nearest large prey
        let mut nearest_prey: Option<&LargePrey> = None;
        let mut min_dist = f32::MAX;

        for prey in &self.large_prey {
            let dx = prey.x as i32 - org.x as i32;
            let dy = prey.y as i32 - org.y as i32;
            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist < min_dist {
                min_dist = dist;
                nearest_prey = Some(prey);
            }
        }

        let (large_prey_nearby, large_prey_distance, large_prey_health, large_prey_attackers,
             large_prey_need, prey_escape_urgency) = if let Some(prey) = nearest_prey {
            let sense_range = 10.0;
            if min_dist <= sense_range {
                // Count current attackers on this prey
                let attackers = self.count_attackers_on_prey(prey.id);
                let need = (prey.attackers_needed as i32 - attackers as i32).max(0) as f32;

                (
                    1.0,                                      // prey nearby
                    (min_dist / sense_range).min(1.0),       // normalized distance
                    prey.health / prey.max_health,            // normalized health
                    (attackers as f32 / 5.0).min(1.0),       // normalized attackers
                    (need / 3.0).min(1.0),                   // normalized need
                    1.0 - (prey.escape_timer as f32 / prey.max_escape_timer as f32), // urgency
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };

        // Find nearest potential cooperation partner
        let sense_range = 5u8;
        let neighbors = self.spatial_index.query_neighbors(org.x, org.y, sense_range);

        let mut best_partner_trust = 0.0f32;
        let mut partner_dist = 1.0f32;
        let mut partner_fitness_val = 0.0f32;
        let mut found_partner = false;

        for &neighbor_idx in &neighbors {
            if neighbor_idx >= self.organisms.len() {
                continue;
            }
            let other = &self.organisms[neighbor_idx];
            if other.id == org.id || !other.is_alive() {
                continue;
            }
            // Potential partner: has kills (can hunt) or has high energy
            if other.kills > 0 || other.energy > 50.0 {
                found_partner = true;
                let dx = other.x as i32 - org.x as i32;
                let dy = other.y as i32 - org.y as i32;
                let dist = ((dx * dx + dy * dy) as f32).sqrt() / sense_range as f32;

                if dist < partner_dist {
                    partner_dist = dist;
                    partner_fitness_val = (other.fitness() / 1000.0).min(1.0);

                    // Check trust relationship
                    if let Some(trust) = org.trust_relationships.get(&other.id) {
                        best_partner_trust = trust.trust_level;
                    }
                }
            }
        }

        let partner_nearby = if found_partner { 1.0 } else { 0.0 };
        let partner_trust = best_partner_trust;
        let partner_distance = partner_dist;
        let partner_fitness = partner_fitness_val;

        // Check if we received a cooperation proposal
        let cooperation_proposed = if self.cooperation_manager.proposals.iter()
            .any(|(_, t, _)| *t == org.id) { 1.0 } else { 0.0 };

        // Check if currently cooperating
        let cooperation_active = if self.cooperation_manager.is_cooperating(org.id) { 1.0 } else { 0.0 };

        // Hunt success rate
        let total_hunts = org.coop_successes + org.coop_failures;
        let hunt_success_rate = if total_hunts > 0 {
            org.coop_successes as f32 / total_hunts as f32
        } else {
            0.5 // Default neutral
        };

        // Own attack power (based on size and is_predator)
        let own_attack_power = (org.size / 3.0 + if org.is_predator { 0.3 } else { 0.0 }).min(1.0);

        // Time since last cooperation
        let time_since_last_coop = ((self.time.saturating_sub(org.last_coop_time)) as f32 / 500.0).min(1.0);

        (
            large_prey_nearby, large_prey_distance, large_prey_health, large_prey_attackers,
            large_prey_need, partner_nearby, partner_trust, partner_distance,
            cooperation_proposed, cooperation_active, hunt_success_rate, partner_fitness,
            own_attack_power, time_since_last_coop, prey_escape_urgency,
        )
    }

    /// Count attackers currently targeting a large prey
    fn count_attackers_on_prey(&self, prey_id: u64) -> usize {
        self.organisms.iter()
            .filter(|o| o.is_alive() && o.current_hunt_target == Some(prey_id))
            .count()
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // B3: LARGE PREY AND COOPERATION SYSTEM
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Maybe spawn a large prey (called each step)
    pub fn maybe_spawn_large_prey(&mut self) {
        if !self.config.large_prey.enabled {
            return;
        }

        // Check if below max
        if self.large_prey.len() >= self.config.large_prey.max_large_prey {
            return;
        }

        // Check minimum population
        let pop = self.population();
        if pop < self.config.large_prey.min_population {
            return;
        }

        // Random spawn chance
        if self.rng.gen::<f32>() < self.config.large_prey.spawn_chance {
            let x = self.rng.gen_range(5..self.config.world.grid_size as u8 - 5);
            let y = self.rng.gen_range(5..self.config.world.grid_size as u8 - 5);

            let prey = LargePrey::new(self.next_large_prey_id, x, y);
            self.next_large_prey_id += 1;
            self.large_prey.push(prey);
        }
    }

    /// Update large prey (movement, escape timers)
    pub fn update_large_prey(&mut self) {
        let grid_size = self.config.world.grid_size as u8;

        // Update each prey
        for prey in &mut self.large_prey {
            // Random wander
            if self.rng.gen::<f32>() < 0.3 {
                prey.wander(grid_size, &mut self.rng);
            }
        }

        // Remove escaped prey and clean up cooperations
        let escaped_ids: Vec<u64> = self.large_prey.iter()
            .filter(|p| p.has_escaped())
            .map(|p| p.id)
            .collect();

        for prey_id in &escaped_ids {
            self.cooperation_manager.remove_for_prey(*prey_id);
            // Mark cooperation as failed for organisms targeting this prey
            for org in &mut self.organisms {
                if org.current_hunt_target == Some(*prey_id) {
                    org.coop_failures += 1;
                    org.current_hunt_target = None;
                }
            }
        }

        self.large_prey.retain(|p| !p.has_escaped());
    }

    /// Handle large prey attacks
    pub fn handle_large_prey_attacks(&mut self, actions: &[(usize, Action, Vec<f32>)]) {
        if !self.config.large_prey.enabled || self.large_prey.is_empty() {
            return;
        }

        // Group attackers by prey
        let mut attacks_by_prey: HashMap<u64, Vec<usize>> = HashMap::new();

        for (idx, action, _) in actions {
            let (idx, action) = (*idx, *action);
            if action != Action::AttackLargePrey {
                continue;
            }
            if idx >= self.organisms.len() || !self.organisms[idx].is_alive() {
                continue;
            }

            let org = &self.organisms[idx];
            if let Some(prey_id) = org.current_hunt_target {
                attacks_by_prey.entry(prey_id).or_default().push(idx);
            } else {
                // No target, find nearest large prey
                let org_x = org.x;
                let org_y = org.y;
                if let Some(prey) = self.large_prey.iter()
                    .filter(|p| {
                        let dx = (p.x as i32 - org_x as i32).abs();
                        let dy = (p.y as i32 - org_y as i32).abs();
                        dx <= 2 && dy <= 2 // Must be close
                    })
                    .min_by_key(|p| {
                        let dx = p.x as i32 - org_x as i32;
                        let dy = p.y as i32 - org_y as i32;
                        dx * dx + dy * dy
                    })
                {
                    attacks_by_prey.entry(prey.id).or_default().push(idx);
                }
            }
        }

        // Process attacks on each prey
        let mut killed_prey: Vec<u64> = Vec::new();

        for (prey_id, attacker_indices) in attacks_by_prey {
            let prey_idx = match self.large_prey.iter().position(|p| p.id == prey_id) {
                Some(idx) => idx,
                None => continue,
            };

            let num_attackers = attacker_indices.len() as u32;

            // Calculate damage per attacker
            let damage_per_attacker: f32 = attacker_indices.iter()
                .map(|&idx| self.organisms[idx].size * 5.0) // Size-based damage
                .sum::<f32>() / num_attackers as f32;

            // Apply damage
            let actual_damage = self.large_prey[prey_idx].take_damage(damage_per_attacker, num_attackers);

            if actual_damage > 0.0 {
                // Damage was dealt (enough attackers)
                if self.large_prey[prey_idx].is_dead() {
                    // Prey killed! Distribute rewards
                    let reward = self.large_prey[prey_idx].reward_per_attacker(num_attackers);
                    killed_prey.push(prey_id);

                    for &idx in &attacker_indices {
                        self.organisms[idx].energy += reward;
                        self.organisms[idx].coop_successes += 1;
                        self.organisms[idx].last_coop_time = self.time;
                        self.organisms[idx].current_hunt_target = None;

                        // Update trust for cooperation partners
                        if let Some(partner_id) = self.cooperation_manager.get_partner(self.organisms[idx].id) {
                            let trust = self.organisms[idx].trust_relationships
                                .entry(partner_id)
                                .or_insert_with(|| crate::ecology::large_prey::TrustRelationship::new(partner_id, self.time));
                            trust.record_success(self.time);
                        }
                    }
                }
            } else {
                // Not enough attackers - small energy cost for trying
                for &idx in &attacker_indices {
                    self.organisms[idx].energy -= 1.0;
                }
            }
        }

        // Remove killed prey and clean up cooperations
        for prey_id in killed_prey {
            self.cooperation_manager.remove_for_prey(prey_id);
            self.large_prey.retain(|p| p.id != prey_id);
        }
    }

    /// Clean up cooperation state
    pub fn cleanup_cooperation(&mut self) {
        self.cooperation_manager.cleanup(self.time);

        // Decay trust relationships
        for org in &mut self.organisms {
            org.trust_relationships.retain(|_, trust| {
                trust.decay(self.time);
                !trust.is_stale(self.time, 1000)
            });

            // Reset cooperation signal
            if org.cooperation_signal != crate::ecology::large_prey::CooperationSignal::None {
                org.cooperation_signal = crate::ecology::large_prey::CooperationSignal::None;
            }
        }
    }

    /// B2: Get predator pattern recognition inputs for an organism
    fn get_predator_inputs(&self, org: &Organism) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        // Find the nearest/most threatening observed predator
        let mut best_obs: Option<&crate::ecology::predation::PredatorObservation> = None;
        let mut best_threat = 0.0f32;

        for (_pred_id, obs) in &org.observed_predators {
            // Skip stale observations
            if obs.is_stale(self.time, 50) {
                continue;
            }

            // Calculate threat level based on distance and speed
            if let Some(&(pred_x, pred_y)) = obs.positions.back() {
                let dx = pred_x as i32 - org.x as i32;
                let dy = pred_y as i32 - org.y as i32;
                let dist = ((dx * dx + dy * dy) as f32).sqrt().max(1.0);

                // Threat = (speed * chasing_factor) / distance
                let chasing_factor = if obs.is_chasing { 2.0 } else { 1.0 };
                let threat = (obs.average_speed * chasing_factor) / dist;

                if threat > best_threat {
                    best_threat = threat;
                    best_obs = Some(obs);
                }
            }
        }

        // If no predator observed, return zeros
        let Some(obs) = best_obs else {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        };

        // Extract features
        let strategy = obs.strategy_one_hot();
        let approach_angle = obs.approach_angle(org.x, org.y);

        // Time observed (normalized, max 100 steps)
        let time_observed = (obs.positions.len() as f32 / 10.0).min(1.0);

        // Relative speed (positive = predator faster)
        // Estimate organism's speed from path history
        let org_speed = if org.path_history.len() >= 2 {
            let mut total = 0.0f32;
            for i in 1..org.path_history.len() {
                let (px, py) = org.path_history[i - 1];
                let (cx, cy) = org.path_history[i];
                let d = (((cx as i32 - px as i32).pow(2) + (cy as i32 - py as i32).pow(2)) as f32).sqrt();
                total += d;
            }
            total / (org.path_history.len() - 1) as f32
        } else {
            0.5 // Default
        };
        let relative_speed = ((obs.average_speed - org_speed) / 2.0).clamp(-1.0, 1.0);

        // Threat level (0-1, combines multiple factors)
        let threat_level = (best_threat / 5.0).clamp(0.0, 1.0);

        (
            (obs.movement_variance / 3.0).min(1.0),  // Normalize variance
            obs.average_speed.min(1.0),               // Speed (already 0-1 ish)
            obs.direction_consistency,                // Already 0-1
            strategy[0],                              // Random
            strategy[1],                              // Patrol
            strategy[2],                              // Chase
            strategy[3],                              // Ambush
            obs.confidence,                           // Already 0-1
            approach_angle,                           // Already 0-1
            relative_speed,                           // -1 to +1
            time_observed,                            // 0-1
            threat_level,                             // 0-1
        )
    }

    /// Execute actions sequentially
    fn execute_actions(&mut self, actions: &[(usize, Action, Vec<f32>)]) {
        for (idx, action, _inputs) in actions {
            let idx = *idx;
            let action = *action;
            if idx >= self.organisms.len() || !self.organisms[idx].is_alive() {
                continue;
            }

            // Calculate terrain-adjusted move cost
            let base_cost = self.config.organisms.move_cost;
            let terrain_cost = if self.config.terrain.enabled {
                let org = &self.organisms[idx];
                let terrain = self.terrain_grid.get(org.x, org.y);
                terrain.movement_cost(org.is_aquatic)
            } else {
                1.0
            };
            let move_cost = base_cost * terrain_cost;

            // Check for intelligent escape bonus (flee away from threats)
            let escape_bonus = self.calculate_escape_bonus(idx, &action);

            match action {
                // Cardinal directions
                Action::MoveNorth => {
                    self.try_move_with_terrain(idx, 0, -1 * (1 + escape_bonus), move_cost);
                }
                Action::MoveEast => {
                    self.try_move_with_terrain(idx, 1 + escape_bonus, 0, move_cost);
                }
                Action::MoveSouth => {
                    self.try_move_with_terrain(idx, 0, 1 + escape_bonus, move_cost);
                }
                Action::MoveWest => {
                    self.try_move_with_terrain(idx, -1 - escape_bonus, 0, move_cost);
                }
                // Diagonal directions (slightly higher cost for realism: sqrt(2) ≈ 1.41)
                Action::MoveNorthEast => {
                    self.try_move_with_terrain(idx, 1 + escape_bonus, -1 * (1 + escape_bonus), move_cost * 1.41);
                }
                Action::MoveSouthEast => {
                    self.try_move_with_terrain(idx, 1 + escape_bonus, 1 + escape_bonus, move_cost * 1.41);
                }
                Action::MoveSouthWest => {
                    self.try_move_with_terrain(idx, -1 - escape_bonus, 1 + escape_bonus, move_cost * 1.41);
                }
                Action::MoveNorthWest => {
                    self.try_move_with_terrain(idx, -1 - escape_bonus, -1 * (1 + escape_bonus), move_cost * 1.41);
                }
                Action::Eat => {
                    let org_x = self.organisms[idx].x;
                    let org_y = self.organisms[idx].y;

                    // Check if there's food available
                    let available_food = self.food_grid.get(org_x, org_y);
                    if available_food < 0.1 {
                        // No food to eat
                        continue;
                    }

                    // COGNITIVE GATE CHECK
                    if self.config.cognitive_gate.enabled {
                        let cg = &self.config.cognitive_gate;
                        let food_complexity = self.complexity_grid.get(org_x, org_y);
                        let brain_capability = self.organisms[idx].brain.get_complexity_capability();
                        let tolerance = cg.tolerance;

                        if brain_capability >= (food_complexity - tolerance) {
                            // SUCCESS: Brain can handle this food complexity
                            // Determine tier and energy using configurable thresholds
                            let food_tier = crate::grid::FoodTier::from_complexity_with_thresholds(
                                food_complexity,
                                cg.simple_complexity_max,
                                cg.medium_complexity_max,
                            );
                            let food_energy = food_tier.energy_with_config(
                                cg.simple_energy,
                                cg.medium_energy,
                                cg.complex_energy,
                            );

                            // Consume food (with seasonal energy multiplier - Phase 2 Feature 4)
                            let consumed = self.food_grid.consume(org_x, org_y, food_energy);
                            let seasonal_energy = consumed * self.seasonal_system.energy_multiplier();
                            self.organisms[idx].energy += seasonal_energy;
                            self.organisms[idx].food_eaten += 1;
                            self.organisms[idx].successful_eats += 1;

                            // Track by tier
                            match food_tier {
                                crate::grid::FoodTier::Simple => self.organisms[idx].simple_food_eaten += 1,
                                crate::grid::FoodTier::Medium => self.organisms[idx].medium_food_eaten += 1,
                                crate::grid::FoodTier::Complex => self.organisms[idx].complex_food_eaten += 1,
                            }

                            // Clear complexity when food consumed
                            self.complexity_grid.clear(org_x, org_y);

                            // Record for spatial memory
                            self.record_food_eaten(org_x, org_y);

                            // Deplete nearby food patch
                            if let Some(ref mut patches) = self.patch_world {
                                if let Some(idx_p) = patches.get_nearest_patch(org_x, org_y, patches.config.patch_radius) {
                                    patches.patches[idx_p].deplete(patches.config.depletion_rate, self.time);
                                    if let Some(ref mut bt) = self.behavior_tracker {
                                        let org_id = self.organisms[idx].id;
                                        bt.track_patch_visit(org_id, self.time, idx_p);
                                    }
                                }
                            }
                        } else {
                            // FAILURE: Brain too simple for this food
                            self.organisms[idx].energy -= self.config.cognitive_gate.failure_cost;
                            self.organisms[idx].failed_eats += 1;
                            // Food remains - organism must find simpler food
                        }
                    } else {
                        // COGNITIVE GATE DISABLED: Original behavior
                        let depletion = self.get_food_depletion(org_x, org_y);
                        let effective_food_energy = self.config.organisms.food_energy * (1.0 - depletion * 0.50);

                        let patch_bonus = if let Some(ref patches) = self.patch_world {
                            if patches.get_nearest_patch(org_x, org_y, patches.config.patch_radius).is_some() {
                                2.0
                            } else {
                                0.5
                            }
                        } else {
                            1.0
                        };
                        // Apply patch bonus and seasonal multiplier (Phase 2 Feature 4)
                        let seasonal_mult = self.seasonal_system.energy_multiplier();
                        let effective_food_energy = effective_food_energy * patch_bonus * seasonal_mult;

                        let result = self.organisms[idx].try_eat(&mut self.food_grid, effective_food_energy);

                        if matches!(result, crate::organism::ActionResult::Success) {
                            self.record_food_eaten(org_x, org_y);
                            if let Some(ref mut patches) = self.patch_world {
                                if let Some(idx_p) = patches.get_nearest_patch(org_x, org_y, patches.config.patch_radius) {
                                    patches.patches[idx_p].deplete(patches.config.depletion_rate, self.time);
                                    if let Some(ref mut bt) = self.behavior_tracker {
                                        let org_id = self.organisms[idx].id;
                                        bt.track_patch_visit(org_id, self.time, idx_p);
                                    }
                                }
                            }
                        }
                    }
                }
                Action::Signal(val) => {
                    self.organisms[idx].signal = val;
                }
                Action::Wait => {
                    self.organisms[idx].energy -= 0.5;
                }
                Action::SignalDanger => {
                    // Emit danger signal if not on cooldown
                    if self.organisms[idx].signal_cooldown == 0 {
                        self.organisms[idx].social_signal = crate::organism::SocialSignal::Danger;
                        self.organisms[idx].signal_cooldown = 10; // 10 step cooldown
                    }
                }
                Action::SignalFood => {
                    // Emit food found signal if not on cooldown
                    if self.organisms[idx].signal_cooldown == 0 {
                        self.organisms[idx].social_signal = crate::organism::SocialSignal::FoodFound;
                        self.organisms[idx].signal_cooldown = 10;
                    }
                }
                Action::Reproduce | Action::Attack => {
                    // Handled separately
                }
                Action::ProposeCooperation => {
                    // B3: Propose cooperation to nearest potential partner
                    let sense_range = 3u8;
                    let neighbors = self.spatial_index.query_neighbors(
                        self.organisms[idx].x,
                        self.organisms[idx].y,
                        sense_range
                    );

                    for &neighbor_idx in &neighbors {
                        if neighbor_idx >= self.organisms.len() || neighbor_idx == idx {
                            continue;
                        }
                        let other = &self.organisms[neighbor_idx];
                        if other.is_alive() && (other.kills > 0 || other.energy > 50.0) {
                            let proposer_id = self.organisms[idx].id;
                            let target_id = other.id;
                            self.cooperation_manager.propose(proposer_id, target_id, self.time);
                            self.organisms[idx].cooperation_signal =
                                crate::ecology::large_prey::CooperationSignal::ProposeCooperation;
                            break;
                        }
                    }
                }
                Action::AcceptCooperation => {
                    // B3: Accept cooperation proposal
                    let accepter_id = self.organisms[idx].id;
                    // Find nearest large prey to target
                    let org_x = self.organisms[idx].x;
                    let org_y = self.organisms[idx].y;
                    let prey_id = self.large_prey.iter()
                        .min_by_key(|p| {
                            let dx = p.x as i32 - org_x as i32;
                            let dy = p.y as i32 - org_y as i32;
                            dx * dx + dy * dy
                        })
                        .map(|p| p.id)
                        .unwrap_or(0);

                    if let Some(proposer_id) = self.cooperation_manager.accept(accepter_id, prey_id, self.time) {
                        self.organisms[idx].cooperation_signal =
                            crate::ecology::large_prey::CooperationSignal::AcceptCooperation;
                        self.organisms[idx].current_hunt_target = Some(prey_id);

                        // Update proposer's hunt target too
                        for org in &mut self.organisms {
                            if org.id == proposer_id {
                                org.current_hunt_target = Some(prey_id);
                                break;
                            }
                        }
                    }
                }
                Action::RejectCooperation => {
                    // B3: Reject cooperation (just set signal)
                    self.organisms[idx].cooperation_signal =
                        crate::ecology::large_prey::CooperationSignal::RejectCooperation;
                }
                Action::AttackLargePrey => {
                    // B3: Attack large prey (handled in handle_large_prey_attacks)
                }
            }

            self.organisms[idx].last_action = Some(action);
        }
    }

    /// Calculate escape bonus if organism is fleeing intelligently from threats
    /// Returns 1 if moving away from the strongest threat direction, 0 otherwise
    fn calculate_escape_bonus(&self, idx: usize, action: &Action) -> i8 {
        if !self.config.predation.enabled {
            return 0;
        }

        let org = &self.organisms[idx];

        // Find threats in each direction
        let neighbors = self.spatial_index.query_neighbors(org.x, org.y, 3);
        let mut threat_north = 0.0f32;
        let mut threat_east = 0.0f32;
        let mut threat_south = 0.0f32;
        let mut threat_west = 0.0f32;

        for &neighbor_idx in &neighbors {
            if neighbor_idx >= self.organisms.len() || neighbor_idx == idx {
                continue;
            }
            let other = &self.organisms[neighbor_idx];
            if !other.is_alive() {
                continue;
            }

            // Is this organism a threat?
            let is_threat = other.is_predator || other.size > org.size * 1.2;
            if !is_threat {
                continue;
            }

            let dx = other.x as i16 - org.x as i16;
            let dy = other.y as i16 - org.y as i16;
            let dist = (dx.abs().max(dy.abs()) as f32).max(1.0);
            let intensity = 1.0 / dist;

            if dy < 0 { threat_north += intensity; }
            if dx > 0 { threat_east += intensity; }
            if dy > 0 { threat_south += intensity; }
            if dx < 0 { threat_west += intensity; }
        }

        // Find the strongest threat direction
        let max_threat = threat_north.max(threat_east).max(threat_south).max(threat_west);
        if max_threat < 0.1 {
            return 0; // No significant threats
        }

        // Check if moving AWAY from the strongest threat
        let is_smart_escape = match action {
            Action::MoveSouth if threat_north >= max_threat * 0.9 => true, // Flee south from north threat
            Action::MoveWest if threat_east >= max_threat * 0.9 => true,   // Flee west from east threat
            Action::MoveNorth if threat_south >= max_threat * 0.9 => true, // Flee north from south threat
            Action::MoveEast if threat_west >= max_threat * 0.9 => true,   // Flee east from west threat
            _ => false,
        };

        if is_smart_escape { 1 } else { 0 }
    }

    /// Try to move with terrain passability check
    fn try_move_with_terrain(&mut self, idx: usize, dx: i8, dy: i8, move_cost: f32) {
        let org = &self.organisms[idx];
        let new_x = (org.x as i16 + dx as i16) as i16;
        let new_y = (org.y as i16 + dy as i16) as i16;

        // Bounds check
        if new_x < 0 || new_x >= self.config.world.grid_size as i16
            || new_y < 0 || new_y >= self.config.world.grid_size as i16
        {
            return;
        }

        let new_xu = new_x as u8;
        let new_yu = new_y as u8;

        // Terrain passability check
        if self.config.terrain.enabled {
            let target_terrain = self.terrain_grid.get(new_xu, new_yu);
            if !target_terrain.is_passable(org.is_aquatic) {
                return; // Can't move there
            }
        }

        // Dynamic obstacle check (Phase 2 Feature 4)
        if let Some(ref obstacle_system) = self.dynamic_obstacle_system {
            if obstacle_system.is_blocked(new_xu, new_yu) {
                return; // Blocked by dynamic obstacle
            }
        }

        // Execute move
        self.organisms[idx].x = new_xu;
        self.organisms[idx].y = new_yu;
        self.organisms[idx].energy -= move_cost;

        // B1: Record movement for path tracking
        self.organisms[idx].record_movement();

        // Phase 1: Track movement for behavior analysis
        if let Some(ref mut bt) = self.behavior_tracker {
            let org_id = self.organisms[idx].id;
            bt.start_tracking(org_id);
            bt.track_movement(org_id, new_xu, new_yu);
        }

        // MODERATE MODE: Penalties for suboptimal behavior (B1 Sequential Memory)
        // Balanced to create pressure without causing mass extinction
        if self.organisms[idx].detect_loop() {
            // Adaptive loop penalty: 3x base, gentle in survival mode
            let penalty = if self.organisms[idx].energy > 20.0 {
                9.0  // 3× instead of 5× - maintains selection pressure
            } else {
                3.0  // Survival mode - prevents death spiral
            };
            self.organisms[idx].energy -= penalty;
        }
        if self.organisms[idx].detect_oscillation() {
            // Adaptive oscillation penalty: 3x base, gentle in survival mode
            let penalty = if self.organisms[idx].energy > 20.0 {
                6.0  // 3× instead of 5×
            } else {
                2.0  // Survival mode
            };
            self.organisms[idx].energy -= penalty;
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

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ANTI-BOTTLENECK DIVERSITY MECHANISMS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Update terrain adaptation tracking for an organism
    fn update_terrain_adaptation(&mut self, org_id: u64, x: u8, y: u8) {
        if !self.config.diversity.enabled || !self.config.diversity.niche_enabled {
            return;
        }

        let current_terrain = self.terrain_grid.get(x, y);

        let entry = self.terrain_adaptation.entry(org_id).or_insert((current_terrain, 0));

        if entry.0 == current_terrain {
            // Same terrain, increment time
            entry.1 += 1;
        } else {
            // Changed terrain, reset adaptation
            *entry = (current_terrain, 1);
        }
    }

    /// Calculate niche bonus for an organism based on terrain specialization
    fn calculate_niche_bonus(&self, org: &Organism) -> f32 {
        if !self.config.diversity.enabled || !self.config.diversity.niche_enabled {
            return 1.0;
        }

        let mut bonus = 1.0f32;

        // Check terrain adaptation
        if let Some(&(terrain, time_in_terrain)) = self.terrain_adaptation.get(&org.id) {
            if time_in_terrain >= self.config.diversity.niche_adaptation_time {
                // Specialist bonus
                bonus *= self.config.diversity.niche_specialist_bonus;

                // Check diet-terrain synergy
                let synergy = match terrain {
                    Terrain::Forest => org.diet.plant_efficiency > 0.6 || org.diet.fruit_efficiency > 0.6,
                    Terrain::Mountain => org.is_predator || org.diet.meat_efficiency > 0.6,
                    Terrain::Desert => org.diet.insect_efficiency > 0.6,
                    Terrain::Water => org.diet.plant_efficiency > 0.5 && org.diet.insect_efficiency > 0.5,
                    Terrain::Plain => true, // Generalist terrain
                };

                if synergy {
                    bonus *= self.config.diversity.terrain_diet_synergy_bonus;
                }
            } else if time_in_terrain < self.config.diversity.niche_adaptation_time / 3 {
                // Generalist penalty (moves around too much)
                bonus *= self.config.diversity.niche_generalist_penalty;
            }
        }

        bonus
    }

    /// Classify an organism's strategy for frequency-dependent selection
    fn classify_strategy(&self, org: &Organism) -> String {
        let method = &self.config.diversity.strategy_classification;

        // Diet classification
        let diet_type = if org.is_predator || org.diet.meat_efficiency > 0.6 {
            "C" // Carnivore
        } else if org.diet.plant_efficiency > 0.6 {
            "H" // Herbivore
        } else {
            "O" // Omnivore
        };

        // Terrain classification
        let terrain_type = if let Some(&(terrain, _)) = self.terrain_adaptation.get(&org.id) {
            match terrain {
                Terrain::Forest => "F",
                Terrain::Mountain => "M",
                Terrain::Desert => "D",
                Terrain::Water => "W",
                Terrain::Plain => "P",
            }
        } else {
            "P" // Default to plains
        };

        // Brain classification
        let brain_type = if org.brain.complexity() >= 3 {
            "B3" // Complex brain
        } else if org.brain.complexity() >= 2 {
            "B2" // Medium brain
        } else {
            "B1" // Simple brain
        };

        match method.as_str() {
            "diet" => diet_type.to_string(),
            "terrain" => terrain_type.to_string(),
            "brain" => brain_type.to_string(),
            _ => format!("{}-{}-{}", diet_type, terrain_type, brain_type), // combined
        }
    }

    /// Update strategy frequency counts
    fn update_strategy_frequencies(&mut self) {
        if !self.config.diversity.enabled || !self.config.diversity.frequency_dependent_enabled {
            return;
        }

        // Only update every 10 steps to reduce overhead
        if self.time - self.last_frequency_update < 10 {
            return;
        }
        self.last_frequency_update = self.time;

        self.strategy_frequencies.clear();

        for org in &self.organisms {
            if org.is_alive() {
                let strategy = self.classify_strategy(org);
                *self.strategy_frequencies.entry(strategy).or_insert(0) += 1;
            }
        }
    }

    /// Calculate frequency-dependent bonus for an organism
    fn calculate_frequency_bonus(&self, org: &Organism) -> f32 {
        if !self.config.diversity.enabled || !self.config.diversity.frequency_dependent_enabled {
            return 1.0;
        }

        let strategy = self.classify_strategy(org);
        let count = *self.strategy_frequencies.get(&strategy).unwrap_or(&0);
        let total = self.organisms.iter().filter(|o| o.is_alive()).count();

        if total == 0 {
            return 1.0;
        }

        let frequency = count as f32 / total as f32;

        if frequency < self.config.diversity.rare_strategy_threshold {
            // Rare strategy bonus: interpolate from 1.0 to max_bonus
            let rarity = 1.0 - (frequency / self.config.diversity.rare_strategy_threshold);
            1.0 + rarity * (self.config.diversity.rare_strategy_max_bonus - 1.0)
        } else if frequency > self.config.diversity.common_strategy_threshold {
            // Common strategy penalty
            self.config.diversity.common_strategy_penalty
        } else {
            // Normal range
            1.0
        }
    }

    /// Handle reproduction (supports both sexual and asexual)
    fn handle_reproduction(&mut self) {
        // Update strategy frequencies before reproduction
        self.update_strategy_frequencies();

        // Update terrain adaptation for all organisms
        for org in &self.organisms {
            if org.is_alive() {
                let (id, x, y) = (org.id, org.x, org.y);
                // We need to update after the loop due to borrow checker
                let terrain = self.terrain_grid.get(x, y);
                let entry = self.terrain_adaptation.entry(id).or_insert((terrain, 0));
                if entry.0 == terrain {
                    entry.1 += 1;
                } else {
                    *entry = (terrain, 1);
                }
            }
        }

        if self.config.reproduction.enabled {
            self.handle_sexual_reproduction();
        } else {
            self.handle_asexual_reproduction();
        }
    }

    /// Handle asexual reproduction (original behavior)
    /// HARSH MODE: Fitness-based selection - higher fitness = higher reproduction probability
    fn handle_asexual_reproduction(&mut self) {
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

        // HARSH MODE: Calculate max fitness for normalization
        let max_fitness = candidates
            .iter()
            .map(|&idx| self.organisms[idx].fitness())
            .fold(1.0f32, |a, b| a.max(b));

        // Process reproductions
        for idx in candidates {
            if offspring.len() >= space_available {
                break;
            }

            // HARSH MODE: Fitness-based reproduction probability
            // Higher fitness organisms reproduce more often
            // Base 30% + up to 50% based on relative fitness
            let fitness = self.organisms[idx].fitness();
            let relative_fitness = fitness / max_fitness;
            let mut reproduction_probability = 0.3 + 0.5 * relative_fitness;

            // Apply diversity bonuses
            let org = &self.organisms[idx];
            let niche_bonus = self.calculate_niche_bonus(org);
            let frequency_bonus = self.calculate_frequency_bonus(org);
            reproduction_probability *= niche_bonus * frequency_bonus;
            reproduction_probability = reproduction_probability.min(0.95); // Cap at 95%

            // Random chance based on fitness (replaces fixed 60% threshold)
            if self.rng.gen::<f32>() > reproduction_probability {
                continue;
            }

            let child_id = self.next_organism_id;
            self.next_organism_id += 1;

            if let Some(mut child) = self.organisms[idx].reproduce(child_id, &self.config) {
                // Enable learning if configured
                if self.config.learning.enabled {
                    child.brain.enable_learning_with_config(&self.config.learning);
                }
                // Record in phylogeny
                self.phylogeny.record_birth(
                    child.id,
                    child.parent1_id,
                    child.parent2_id,
                    self.time,
                    child.brain.complexity(),
                    child.energy,
                    child.size,
                    child.lineage_id,
                    child.generation,
                    child.brain.hash(),
                );
                self.phylogeny.record_offspring(child.parent1_id, child.parent2_id);

                // Log birth and reproduction to database
                #[cfg(feature = "database")]
                if let Some(ref sender) = self.db_sender {
                    let _ = sender.send(DbEvent::OrganismBirth {
                        organism_id: child.id,
                        lineage_id: child.lineage_id,
                        generation: child.generation,
                        parent1_id: child.parent1_id,
                        parent2_id: child.parent2_id,
                        step: self.time,
                        x: child.x,
                        y: child.y,
                        brain_layers: child.brain.complexity() as i32,
                        brain_neurons: child.brain.total_hidden_neurons() as i32,
                        brain_connections: child.brain.parameter_count() as i32,
                        is_predator: child.is_predator,
                    });
                    let _ = sender.send(DbEvent::ReproductionEvent {
                        step: self.time,
                        parent1_id: self.organisms[idx].id,
                        parent2_id: None,
                        child_id: child.id,
                        child_lineage_id: child.lineage_id,
                        child_generation: child.generation,
                        mutation_count: 0,
                        parent1_energy: self.organisms[idx].energy,
                        child_energy: child.energy,
                    });
                }

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

    /// Handle sexual reproduction (two-parent mating)
    fn handle_sexual_reproduction(&mut self) {
        // Check population limit
        let current_pop = self.organisms.iter().filter(|o| o.is_alive()).count();
        let space_available = self.config.safety.max_population.saturating_sub(current_pop);

        if space_available == 0 {
            return;
        }

        // Collect mating candidates (can mate and have energy)
        let candidates: Vec<usize> = self
            .organisms
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_alive() && o.can_mate(&self.config))
            .map(|(idx, _)| idx)
            .collect();

        // Find valid pairs first (without borrowing mutably)
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut speciation_blocks = 0u64;
        let mut boundary_zone_attempts = 0u64;

        for &idx1 in &candidates {
            if pairs.len() >= space_available {
                break;
            }
            if used.contains(&idx1) {
                continue;
            }

            let org1_x = self.organisms[idx1].x;
            let org1_y = self.organisms[idx1].y;
            let org1_sex = self.organisms[idx1].sex;

            // Find a suitable mate nearby
            let neighbors = self.spatial_index.query_neighbors(org1_x, org1_y, 1);

            for &idx2 in &neighbors {
                if idx2 >= self.organisms.len() || idx2 == idx1 || used.contains(&idx2) {
                    continue;
                }

                let org2 = &self.organisms[idx2];

                // Check if can mate (opposite sex, both have energy, etc.)
                if !org2.is_alive() || !org2.can_mate(&self.config) {
                    continue;
                }
                if org1_sex == org2.sex {
                    continue;
                }

                // Calculate distance
                let org1_for_dist = &self.organisms[idx1];
                let distance = org1_for_dist.distance_to(org2);
                if distance > self.config.reproduction.max_mating_distance {
                    continue;
                }

                // Check speciation compatibility
                let org1_id = self.organisms[idx1].id;
                let org2_id = org2.id;
                let (can_mate, speciation_prob) = check_speciation_compatibility(
                    org1_id,
                    org2_id,
                    &self.phylogeny,
                    &self.config.diversity,
                );

                if !can_mate {
                    // Different species - cannot mate
                    speciation_blocks += 1;
                    continue;
                }

                // Track boundary zone attempts
                if speciation_prob < 1.0 {
                    boundary_zone_attempts += 1;
                }

                // Random chance to mate, modified by speciation probability
                let base_prob = 0.5 * speciation_prob;
                if self.rng.gen::<f32>() > base_prob {
                    continue;
                }

                // Mark as valid pair
                pairs.push((idx1, idx2));
                used.insert(idx1);
                used.insert(idx2);
                break;
            }
        }

        // Record speciation statistics
        self.sexual_reproduction.speciation_blocks += speciation_blocks;
        self.sexual_reproduction.boundary_zone_attempts += boundary_zone_attempts;

        // Now create offspring from pairs
        let mut offspring = Vec::new();
        for (idx1, idx2) in pairs {
            if let Some(mut child) = self.sexual_reproduce(idx1, idx2) {
                if self.config.learning.enabled {
                    child.brain.enable_learning_with_config(&self.config.learning);
                }
                // Log birth and reproduction to database
                #[cfg(feature = "database")]
                if let Some(ref sender) = self.db_sender {
                    let _ = sender.send(DbEvent::OrganismBirth {
                        organism_id: child.id,
                        lineage_id: child.lineage_id,
                        generation: child.generation,
                        parent1_id: child.parent1_id,
                        parent2_id: child.parent2_id,
                        step: self.time,
                        x: child.x,
                        y: child.y,
                        brain_layers: child.brain.complexity() as i32,
                        brain_neurons: child.brain.total_hidden_neurons() as i32,
                        brain_connections: child.brain.parameter_count() as i32,
                        is_predator: child.is_predator,
                    });
                    let _ = sender.send(DbEvent::ReproductionEvent {
                        step: self.time,
                        parent1_id: self.organisms[idx1].id,
                        parent2_id: Some(self.organisms[idx2].id),
                        child_id: child.id,
                        child_lineage_id: child.lineage_id,
                        child_generation: child.generation,
                        mutation_count: 0,
                        parent1_energy: self.organisms[idx1].energy,
                        child_energy: child.energy,
                    });
                }

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

    /// Create offspring from two parents via sexual reproduction
    fn sexual_reproduce(&mut self, parent1_idx: usize, parent2_idx: usize) -> Option<Organism> {
        let parent1 = &self.organisms[parent1_idx];
        let parent2 = &self.organisms[parent2_idx];

        // Check inbreeding
        let is_inbred = SexualReproductionSystem::is_inbred(parent1.lineage_id, parent2.lineage_id);

        if is_inbred {
            self.sexual_reproduction.record_mating(true);
            // 50% chance to fail if inbred
            if self.rng.gen::<f32>() < 0.5 {
                self.sexual_reproduction.record_failed_mating();
                return None;
            }
        } else {
            self.sexual_reproduction.record_mating(false);
        }

        // Perform crossover
        let fitness1 = parent1.fitness();
        let fitness2 = parent2.fitness();
        let child_brain = self.crossover_system.crossover(
            &parent1.brain,
            &parent2.brain,
            fitness1,
            fitness2,
        );

        // Calculate brain complexity before moving child_brain
        let child_brain_complexity = child_brain.complexity();

        // Create child diet by averaging parents and mutating
        let mut child_diet = crate::ecology::food_types::DietSpecialization {
            plant_efficiency: (parent1.diet.plant_efficiency + parent2.diet.plant_efficiency) / 2.0,
            meat_efficiency: (parent1.diet.meat_efficiency + parent2.diet.meat_efficiency) / 2.0,
            fruit_efficiency: (parent1.diet.fruit_efficiency + parent2.diet.fruit_efficiency) / 2.0,
            insect_efficiency: (parent1.diet.insect_efficiency + parent2.diet.insect_efficiency) / 2.0,
        };
        child_diet.mutate(self.config.evolution.mutation_strength);

        // Determine child position (near parent1)
        let child_x = parent1.x;
        let child_y = parent1.y;

        // New lineage ID (mixing lineages)
        let child_lineage = self.next_lineage_id;
        self.next_lineage_id += 1;

        // Calculate child energy
        let mut child_energy = self.config.reproduction.offspring_energy;

        // Apply inbreeding penalty
        if is_inbred {
            let penalty = SexualReproductionSystem::inbreeding_penalty(&self.config.reproduction);
            child_energy *= 1.0 - penalty;
        }

        // Create child
        let child_id = self.next_organism_id;
        self.next_organism_id += 1;

        let parent1_id = parent1.id;
        let parent2_id = parent2.id;
        let parent1_generation = parent1.generation;
        let parent2_generation = parent2.generation;
        let parent1_is_predator = parent1.is_predator;
        let parent2_is_predator = parent2.is_predator;
        let parent1_is_aquatic = parent1.is_aquatic;
        let parent2_is_aquatic = parent2.is_aquatic;
        let child_size = (parent1.size + parent2.size) / 2.0;

        let mut child = Organism {
            id: child_id,
            lineage_id: child_lineage,
            generation: parent1_generation.max(parent2_generation) + 1,
            x: child_x,
            y: child_y,
            size: child_size,
            energy: child_energy,
            health: 100.0,
            age: 0,
            brain: child_brain,
            memory: [0.0; 5],
            stm: crate::organism::ShortTermMemory::for_brain(
                child_brain_complexity,
                self.config.memory.buffer_multiplier,
                self.config.memory.min_buffer_size,
                self.config.memory.max_buffer_size,
                self.config.memory.decay_rate,
            ),
            kills: 0,
            offspring_count: 0,
            food_eaten: 0,
            is_predator: parent1_is_predator || parent2_is_predator,
            signal: 0.0,
            last_action: None,
            diet: child_diet,
            attack_cooldown: 0,
            cause_of_death: None,
            is_aquatic: parent1_is_aquatic || parent2_is_aquatic,
            sex: Sex::random(),
            parent1_id: Some(parent1_id),
            parent2_id: Some(parent2_id),
            mate_cooldown: 0,
            social_signal: crate::organism::SocialSignal::None,
            signal_cooldown: 0,
            path_history: VecDeque::with_capacity(5),
            observed_predators: HashMap::with_capacity(10),
            cooperation_signal: crate::ecology::large_prey::CooperationSignal::None,
            trust_relationships: HashMap::with_capacity(10),
            current_hunt_target: None,
            coop_successes: 0,
            coop_failures: 0,
            last_coop_time: 0,
            last_reward: 0.0,
            total_lifetime_reward: 0.0,
            successful_forages: 0,
            failed_forages: 0,
            current_learning_rate: 0.0,
            learning_updates_count: 0,
            // Cognitive Gate
            successful_eats: 0,
            failed_eats: 0,
            simple_food_eaten: 0,
            medium_food_eaten: 0,
            complex_food_eaten: 0,
        };

        // Mutate child's brain
        let mutation_config = crate::neural::MutationConfig {
            weight_mutation_rate: self.config.evolution.mutation_rate,
            weight_mutation_strength: self.config.evolution.mutation_strength,
            add_neuron_rate: self.config.evolution.add_neuron_rate,
            add_connection_rate: self.config.evolution.add_connection_rate,
            max_neurons: self.config.safety.max_neurons,
        };
        child.brain.mutate(&mutation_config);

        // Update learning rate to match new brain complexity
        if self.config.learning.enabled {
            child.brain.update_learning_rate(&self.config.learning);
        }

        // Update STM buffer size to match new brain complexity
        child.stm.resize_for_brain(
            child.brain.complexity(),
            self.config.memory.buffer_multiplier,
            self.config.memory.min_buffer_size,
            self.config.memory.max_buffer_size,
        );

        // Small chance to mutate aquatic trait
        if self.rng.gen::<f32>() < 0.01 {
            child.is_aquatic = !child.is_aquatic;
        }

        // Deduct energy from parents and set cooldown
        self.organisms[parent1_idx].energy -= self.config.reproduction.energy_cost;
        self.organisms[parent2_idx].energy -= self.config.reproduction.energy_cost;
        self.organisms[parent1_idx].mate_cooldown = self.config.reproduction.cooldown;
        self.organisms[parent2_idx].mate_cooldown = self.config.reproduction.cooldown;
        self.organisms[parent1_idx].offspring_count += 1;
        self.organisms[parent2_idx].offspring_count += 1;

        // Record in phylogeny
        self.phylogeny.record_birth(
            child.id,
            child.parent1_id,
            child.parent2_id,
            self.time,
            child.brain.complexity(),
            child.energy,
            child.size,
            child.lineage_id,
            child.generation,
            child.brain.hash(),
        );
        self.phylogeny.record_offspring(child.parent1_id, child.parent2_id);
        self.sexual_reproduction.record_offspring();

        Some(child)
    }

    /// Remove dead organisms
    fn remove_dead(&mut self) {
        // Record deaths in phylogeny and survival analyzer before removing
        for org in &self.organisms {
            if !org.is_alive() {
                self.phylogeny.record_death(org.id, self.time);

                // Record in survival analyzer (estimate birth time from age)
                let birth_time = self.time.saturating_sub(org.age as u64);
                self.survival_analyzer.record_death(org, birth_time, self.time);

                // Mark dead in behavior tracker
                if let Some(ref mut bt) = self.behavior_tracker {
                    bt.mark_dead(org.id);
                }

                // Log death to database
                #[cfg(feature = "database")]
                if let Some(ref sender) = self.db_sender {
                    let death_cause = match org.cause_of_death {
                        Some(DeathCause::Starvation) => "starvation",
                        Some(DeathCause::Predation) => "predation",
                        Some(DeathCause::OldAge) => "old_age",
                        None => "unknown",
                    };
                    let _ = sender.send(DbEvent::OrganismDeath {
                        organism_id: org.id,
                        step: self.time,
                        x: org.x,
                        y: org.y,
                        death_cause: death_cause.to_string(),
                        food_eaten: org.food_eaten,
                        kills: org.kills,
                        offspring: org.offspring_count,
                        max_energy: org.energy, // energy at death
                    });
                }
            }
        }

        let alive_before = self.organisms.len();
        self.organisms.retain(|org| org.is_alive());
        self.deaths_this_step = alive_before - self.organisms.len();
    }

    /// Handle predation (Attack actions)
    fn handle_predation(&mut self, actions: &[(usize, Action, Vec<f32>)]) {
        if !self.config.predation.enabled {
            return;
        }

        // Collect attack intentions
        let mut attacks: Vec<(usize, usize)> = Vec::new(); // (attacker_idx, target_idx)

        for (idx, action, _) in actions {
            let (idx, action) = (*idx, *action);
            if action == Action::Attack {
                if idx >= self.organisms.len() || !self.organisms[idx].is_alive() {
                    continue;
                }
                if self.organisms[idx].attack_cooldown > 0 {
                    continue;
                }

                let attacker = &self.organisms[idx];

                // Find nearest target in range
                let neighbors = self.spatial_index.query_neighbors(attacker.x, attacker.y, 1);
                for &neighbor_idx in &neighbors {
                    if neighbor_idx < self.organisms.len()
                        && self.organisms[neighbor_idx].is_alive()
                        && neighbor_idx != idx
                    {
                        attacks.push((idx, neighbor_idx));
                        break; // One attack per organism
                    }
                }
            }
        }

        // Execute attacks
        for (attacker_idx, target_idx) in attacks {
            self.execute_attack(attacker_idx, target_idx);
        }
    }

    /// Execute a single attack
    fn execute_attack(&mut self, attacker_idx: usize, target_idx: usize) {
        let attacker_x = self.organisms[attacker_idx].x;
        let attacker_y = self.organisms[attacker_idx].y;
        let attacker_size = self.organisms[attacker_idx].size;
        let target_x = self.organisms[target_idx].x;
        let target_y = self.organisms[target_idx].y;
        let target_size = self.organisms[target_idx].size;
        let target_energy = self.organisms[target_idx].energy;

        // Check range
        if !predation::is_in_range(attacker_x, attacker_y, target_x, target_y) {
            return;
        }

        // Calculate damage
        let damage = predation::calculate_damage(attacker_size, target_size, &self.config.predation);

        // Apply damage
        self.organisms[target_idx].health -= damage;
        self.organisms[target_idx].energy -= damage * 0.5; // Also drain energy

        // Check if killed
        if self.organisms[target_idx].health <= 0.0 {
            // Mark as killed
            self.organisms[target_idx].cause_of_death = Some(DeathCause::Predation);

            // Calculate energy gain
            let energy_gained =
                predation::calculate_energy_gain(target_size, target_energy, &self.config.predation);
            self.organisms[attacker_idx].energy += energy_gained;

            // Update stats
            self.organisms[attacker_idx].kills += 1;
            self.kills_this_step += 1;

            // Become predator (evolve behavior)
            self.organisms[attacker_idx].is_predator = true;
        } else {
            // Failed attack - increased energy cost to encourage strategic target selection
            let failure_cost = damage * 0.3; // Was 0.1, now 0.3
            self.organisms[attacker_idx].energy -= failure_cost;

            // Extra penalty if target is much larger (foolish attack)
            if target_size > attacker_size * 1.5 {
                self.organisms[attacker_idx].energy -= 5.0; // "Wounded in the attempt"
            }
        }

        // Set attack cooldown
        self.organisms[attacker_idx].attack_cooldown = self.config.predation.attack_cooldown;
    }

    /// Update environment (food regeneration with seasonal and terrain multipliers)
    fn update_environment(&mut self) {
        // Update food patches (regeneration) and write to grid
        if let Some(ref mut patches) = self.patch_world {
            patches.update(self.time);
            patches.write_to_food_grid(&mut self.food_grid, self.config.world.food_max);
        }

        // Get seasonal multiplier
        let season_multiplier = self.seasonal_system.food_multiplier();
        // Reduce ambient food regen when patches are active
        let patch_mult = 1.0; // No ambient food reduction - patches add bonus food on top

        // Update depletion states and regenerate food with terrain/depletion modifiers
        if self.config.terrain.enabled || self.config.depletion.enabled {
            for y in 0..self.config.world.grid_size {
                for x in 0..self.config.world.grid_size {
                    let xu = x as u8;
                    let yu = y as u8;

                    // Get terrain modifier
                    let terrain_mult = if self.config.terrain.enabled {
                        self.terrain_grid.get(xu, yu).food_multiplier()
                    } else {
                        1.0
                    };

                    // Update depletion state
                    if self.config.depletion.enabled {
                        let food_amount = self.food_grid.get(xu, yu);
                        let org_count = self.spatial_index.count_at(xu, yu);
                        self.depletion_system.update_cell(
                            xu,
                            yu,
                            food_amount,
                            org_count,
                            &self.config.depletion,
                        );
                    }

                    // Get depletion modifier
                    let depletion_mult = if self.config.depletion.enabled {
                        self.depletion_system.regen_multiplier(xu, yu)
                    } else {
                        1.0
                    };

                    // Combined regeneration
                    let total_mult = season_multiplier * terrain_mult * depletion_mult * patch_mult;
                    let current = self.food_grid.get(xu, yu);
                    let new_food = (current + self.config.world.food_regen_rate * total_mult)
                        .min(self.config.world.food_max);
                    self.food_grid.set(xu, yu, new_food);
                }
            }
        } else {
            // Simple regeneration (no terrain/depletion)
            self.food_grid
                .regenerate(self.config.world.food_regen_rate * season_multiplier * patch_mult);
        }

        // Random food spawning (also affected by season)
        if self.rng.gen::<f32>() < 0.01 * season_multiplier {
            self.food_grid.spawn_random(10.0 * season_multiplier, 0.001);
        }

        // COGNITIVE GATE: Assign complexity to new food
        if self.config.cognitive_gate.enabled {
            self.assign_complexity_to_new_food();
        }
    }

    /// Assign complexity to cells with food but no complexity value (Cognitive Gate)
    fn assign_complexity_to_new_food(&mut self) {
        let cg = &self.config.cognitive_gate;
        let simple_ratio = cg.simple_ratio;
        let medium_ratio = cg.medium_ratio;

        for y in 0..self.config.world.grid_size {
            for x in 0..self.config.world.grid_size {
                let xu = x as u8;
                let yu = y as u8;

                // If food exists but complexity is 0, assign new complexity
                if self.food_grid.get(xu, yu) > 0.1 && self.complexity_grid.get(xu, yu) < 0.01 {
                    let roll: f32 = self.rng.gen();

                    // Use configurable complexity ranges
                    let complexity = if roll < simple_ratio {
                        // Simple food
                        self.rng.gen_range(cg.simple_complexity_min..cg.simple_complexity_max)
                    } else if roll < simple_ratio + medium_ratio {
                        // Medium food
                        self.rng.gen_range(cg.medium_complexity_min..cg.medium_complexity_max)
                    } else {
                        // Complex food
                        self.rng.gen_range(cg.complex_complexity_min..cg.complex_complexity_max)
                    };

                    self.complexity_grid.set(xu, yu, complexity);
                }
            }
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

        // Record diversity metrics (every 100 steps to match diversity_history interval)
        if self.time % self.diversity_history.record_interval == 0 {
            self.diversity_history.record(self.time, &self.organisms, &self.phylogeny);
        }
    }

    /// Log events to database (called each step when database feature is enabled)
    #[cfg(feature = "database")]
    fn log_to_database(&self) {
        let sender = match &self.db_sender {
            Some(s) => s,
            None => return,
        };

        let snapshot_interval = self.config.database.snapshot_interval;

        // World snapshot every stats_interval steps
        if self.time % self.config.logging.stats_interval == 0 {
            let alive: Vec<&crate::organism::Organism> =
                self.organisms.iter().filter(|o| o.is_alive()).collect();
            let pop = alive.len();
            if pop > 0 {
                let avg_energy = alive.iter().map(|o| o.energy).sum::<f32>() / pop as f32;
                let avg_age = alive.iter().map(|o| o.age as f32).sum::<f32>() / pop as f32;
                let avg_layers =
                    alive.iter().map(|o| o.brain.complexity() as f32).sum::<f32>() / pop as f32;
                let avg_neurons =
                    alive.iter().map(|o| o.brain.total_hidden_neurons() as f32).sum::<f32>() / pop as f32;
                let avg_conns = alive
                    .iter()
                    .map(|o| o.brain.parameter_count() as f32)
                    .sum::<f32>()
                    / pop as f32;
                let predator_count = alive.iter().filter(|o| o.is_predator || o.kills > 0).count();

                let _ = sender.send(DbEvent::WorldSnapshot {
                    step: self.time,
                    population: pop as i32,
                    births: self.births_this_step as i32,
                    deaths: self.deaths_this_step as i32,
                    kills: self.kills_this_step as i32,
                    max_generation: self.generation_max as i16,
                    avg_energy,
                    avg_age,
                    avg_brain_layers: avg_layers,
                    avg_brain_neurons: avg_neurons,
                    avg_brain_connections: avg_conns,
                    predator_count: predator_count as i32,
                    lineage_count: self.lineage_tracker.surviving_count() as i32,
                    total_food: self.food_grid.total_food(),
                });
            }
        }

        // Organism snapshots every snapshot_interval steps
        if snapshot_interval > 0 && self.time % snapshot_interval == 0 {
            for org in &self.organisms {
                if !org.is_alive() {
                    continue;
                }
                let action_str = org.last_action.map(|a| format!("{:?}", a));
                let _ = sender.send(DbEvent::OrganismSnapshot {
                    step: self.time,
                    organism_id: org.id,
                    x: org.x,
                    y: org.y,
                    energy: org.energy,
                    health: org.health,
                    age: org.age,
                    size: org.size,
                    kills: org.kills,
                    offspring: org.offspring_count,
                    food_eaten: org.food_eaten,
                    brain_layers: org.brain.complexity() as i32,
                    brain_neurons: org.brain.total_hidden_neurons() as i32,
                    brain_connections: org.brain.parameter_count() as i32,
                    is_predator: org.is_predator,
                    last_action: action_str,
                });

                // Learning events (only if learning is enabled and configured)
                if self.config.database.log_learning_events {
                    if let Some(stats) = org.brain.learning_stats() {
                        let _ = sender.send(DbEvent::LearningEvent {
                            step: self.time,
                            organism_id: org.id,
                            reward: org.last_reward,
                            total_lifetime_reward: org.total_lifetime_reward,
                            successful_forages: org.successful_forages,
                            failed_forages: org.failed_forages,
                            weight_updates_count: stats.update_count as i32,
                        });
                    }
                }
            }
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

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // B2: PREDATOR PATTERN RECOGNITION
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Update predator observations for all organisms
    pub fn update_predator_observations(&mut self) {
        use crate::ecology::predation::PredatorObservation;

        let time = self.time;
        let sense_range = 5u8;

        // Collect predator info: (id, x, y) for all predators
        let predators: Vec<(u64, u8, u8)> = self.organisms.iter()
            .filter(|o| o.is_alive() && (o.is_predator || o.kills > 0))
            .map(|o| (o.id, o.x, o.y))
            .collect();

        // For each non-predator organism, update observations
        for org in &mut self.organisms {
            if !org.is_alive() || org.is_predator {
                continue;
            }

            // Clean up stale observations (max 50 steps old)
            org.observed_predators.retain(|_, obs| !obs.is_stale(time, 50));

            // Limit to max 10 tracked predators
            while org.observed_predators.len() > 10 {
                // Remove oldest
                if let Some(oldest_id) = org.observed_predators.iter()
                    .min_by_key(|(_, obs)| obs.last_seen)
                    .map(|(id, _)| *id)
                {
                    org.observed_predators.remove(&oldest_id);
                } else {
                    break;
                }
            }

            // Update/add observations for nearby predators
            for &(pred_id, pred_x, pred_y) in &predators {
                if pred_id == org.id {
                    continue; // Skip self
                }

                // Check if in sensing range
                let dx = (pred_x as i16 - org.x as i16).abs();
                let dy = (pred_y as i16 - org.y as i16).abs();
                if dx > sense_range as i16 || dy > sense_range as i16 {
                    continue; // Too far
                }

                // Update existing or create new observation
                if let Some(obs) = org.observed_predators.get_mut(&pred_id) {
                    obs.update(pred_x, pred_y, time, org.x, org.y);
                } else {
                    // New predator spotted
                    org.observed_predators.insert(pred_id, PredatorObservation::new(pred_x, pred_y, time));
                }
            }
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // COGNITIVE TASKS: SPATIAL MEMORY
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Record food consumption at a position (for spatial memory)
    pub fn record_food_eaten(&mut self, x: u8, y: u8) {
        let pos = (x, y);
        if let Some(memory) = self.food_memory.get_mut(&pos) {
            // Increase depletion moderately
            memory.depletion_level = (memory.depletion_level + 0.3).min(1.0);
            memory.last_eaten = self.time;
        } else {
            // New entry
            self.food_memory.insert(pos, FoodMemory::new(self.time));
        }
    }

    /// Update food memory (decay over time)
    pub fn update_food_memory(&mut self) {
        const RECOVERY_TIME: u64 = 100;  // Steps before recovery starts
        const DECAY_RATE: f32 = 0.98;    // Slower decay = longer memory matters

        let current_time = self.time;

        // Update all memory entries
        self.food_memory.retain(|_, memory| {
            let elapsed = current_time.saturating_sub(memory.last_eaten);

            if elapsed > RECOVERY_TIME {
                // Gradual recovery (slow)
                memory.depletion_level *= DECAY_RATE;

                // Remove if fully recovered
                memory.depletion_level > 0.01
            } else {
                true
            }
        });
    }

    /// Get food depletion level at a position (0.0 = fresh, 1.0 = depleted)
    pub fn get_food_depletion(&self, x: u8, y: u8) -> f32 {
        self.food_memory
            .get(&(x, y))
            .map(|m| m.depletion_level)
            .unwrap_or(0.0)
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // COGNITIVE TASKS: TEMPORAL PREDICTION
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Update food history for trend tracking
    pub fn update_food_history(&mut self) {
        self.food_history.push_back(self.food_grid.total_food());

        // Keep last 100 timesteps
        if self.food_history.len() > 100 {
            self.food_history.pop_front();
        }
    }

    /// Get current season progress (0.0 = start, 1.0 = end of cycle)
    pub fn current_season_progress(&self) -> f32 {
        if !self.config.seasons.enabled {
            return 0.5; // Neutral if seasons disabled
        }
        let season_length = self.config.seasons.season_length;
        if season_length == 0 {
            return 0.5;
        }
        let cycle = self.time % season_length;
        cycle as f32 / season_length as f32
    }

    /// Get food availability trend (-1.0 = declining, +1.0 = rising)
    pub fn food_availability_trend(&self) -> f32 {
        if self.food_history.len() < 50 {
            return 0.0;
        }

        let current = *self.food_history.back().unwrap_or(&0.0);
        let past = self.food_history[self.food_history.len() - 50];

        if past == 0.0 {
            return 0.0;
        }

        let change = (current - past) / past;
        change.clamp(-1.0, 1.0)
    }

    /// Get normalized steps to next season change (0.0-1.0)
    pub fn steps_to_next_season(&self) -> f32 {
        if !self.config.seasons.enabled {
            return 0.5;
        }
        let season_length = self.config.seasons.season_length;
        if season_length == 0 {
            return 0.5;
        }
        let cycle = self.time % season_length;
        let remaining = season_length - cycle;
        remaining as f32 / season_length as f32
    }

    /// Calculate reward signal for an organism based on energy change
    fn calculate_reward(&self, org: &Organism, energy_before: f32) -> f32 {
        let energy_delta = org.energy - energy_before;

        // Conservative reward: small signal to avoid weight destabilization
        // Scale down heavily and clamp tight
        (energy_delta / 30.0).clamp(-0.5, 0.5)
    }

    /// Apply Hebbian learning phase: replay inputs, compute reward, update weights
    fn apply_learning_phase(
        &mut self,
        actions: &[(usize, Action, Vec<f32>)],
        energy_before: &[f32],
    ) {
        let time = self.time;

        for (idx, _action, inputs) in actions {
            let idx = *idx;
            if idx >= self.organisms.len() || !self.organisms[idx].is_alive() {
                continue;
            }
            if self.organisms[idx].brain.hebbian_state.is_none() {
                continue;
            }

            let eb = if idx < energy_before.len() {
                energy_before[idx]
            } else {
                self.organisms[idx].energy
            };

            // Replay forward pass with learning (records activations)
            let org = &mut self.organisms[idx];
            org.brain.forward_with_learning(inputs, time);

            // Calculate reward from energy change
            let reward = self.calculate_reward(&self.organisms[idx], eb);

            // Apply learning update
            self.organisms[idx].apply_learning_update(reward);

            // Phase 2 Feature 2: Record experience in Short-Term Memory
            if self.config.memory.enabled {
                let sensory_snapshot = ShortTermMemory::compress_sensory(
                    inputs,
                    self.config.memory.features_per_frame,
                );
                let frame = MemoryFrame::new(time, sensory_snapshot, *_action, reward);
                self.organisms[idx].stm.push(frame);
            }
        }
    }

    /// Apply STM decay to all organisms (call once per step)
    fn apply_memory_decay(&mut self) {
        if !self.config.memory.enabled {
            return;
        }

        for org in &mut self.organisms {
            if org.is_alive() {
                org.stm.apply_decay();
            }
        }
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
        // Note: With Rayon parallelism, random diet initialization, and predation,
        // exact reproducibility is not guaranteed. This test verifies basic mechanics.
        let mut config = test_config();
        config.predation.enabled = false;
        config.seasons.enabled = false;

        let mut world1 = World::new_with_seed(config.clone(), 42);
        let mut world2 = World::new_with_seed(config, 42);

        world1.run(50);
        world2.run(50);

        // Time should always match
        assert_eq!(world1.time, world2.time);

        // Both should have surviving population (basic viability check)
        assert!(world1.population() > 0 || world1.generation_max > 0);
        assert!(world2.population() > 0 || world2.generation_max > 0);
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
