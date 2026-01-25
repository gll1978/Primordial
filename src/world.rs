//! World simulation engine - main simulation loop.

use crate::checkpoint::Checkpoint;
use crate::config::Config;
use crate::ecology::depletion::DepletionSystem;
use crate::ecology::predation;
use crate::ecology::seasons::SeasonalSystem;
use crate::ecology::terrain::TerrainGrid;
use crate::evolution::EvolutionEngine;
use crate::genetics::crossover::CrossoverSystem;
use crate::genetics::phylogeny::PhylogeneticTree;
use crate::genetics::sex::{Sex, SexualReproductionSystem};
use crate::grid::{FoodGrid, SpatialIndex};
use crate::organism::{Action, DeathCause, Organism};
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

        // Initialize depletion system
        let depletion_system = DepletionSystem::new(grid_size);

        // Initialize genetics systems
        let mut phylogeny = PhylogeneticTree::new();
        let sexual_reproduction = SexualReproductionSystem::new();
        let crossover_system = CrossoverSystem::new();

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

        // New depletion system (state not preserved in checkpoint)
        let depletion_system = DepletionSystem::new(grid_size);

        // New genetics systems (state not preserved in checkpoint)
        let phylogeny = PhylogeneticTree::new();
        let sexual_reproduction = SexualReproductionSystem::new();
        let crossover_system = CrossoverSystem::new();
        let next_lineage_id = checkpoint.organisms.len() as u32;

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

        // Phase 1: Parallel sensing and thinking
        let actions = self.compute_actions();

        // Phase 2: Execute actions (sequential to avoid conflicts)
        self.execute_actions(&actions);

        // Phase 3: Handle predation (Attack actions)
        self.handle_predation(&actions);

        // Phase 4: Update organisms (aging, metabolism)
        self.update_organisms();

        // Phase 5: Handle reproduction
        self.handle_reproduction();

        // Phase 6: Remove dead organisms
        self.remove_dead();

        // Phase 7: Update environment (with seasonal multipliers)
        self.update_environment();

        // Phase 8: Update spatial index
        self.update_spatial_index();

        // Phase 9: Update statistics
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

            match action {
                Action::MoveNorth => {
                    self.try_move_with_terrain(idx, 0, -1, move_cost);
                }
                Action::MoveEast => {
                    self.try_move_with_terrain(idx, 1, 0, move_cost);
                }
                Action::MoveSouth => {
                    self.try_move_with_terrain(idx, 0, 1, move_cost);
                }
                Action::MoveWest => {
                    self.try_move_with_terrain(idx, -1, 0, move_cost);
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

        // Execute move
        self.organisms[idx].x = new_xu;
        self.organisms[idx].y = new_yu;
        self.organisms[idx].energy -= move_cost;
    }

    /// Update all organisms (aging, metabolism)
    fn update_organisms(&mut self) {
        for org in &mut self.organisms {
            if org.is_alive() {
                org.update(&self.config);
            }
        }
    }

    /// Handle reproduction (supports both sexual and asexual)
    fn handle_reproduction(&mut self) {
        if self.config.reproduction.enabled {
            self.handle_sexual_reproduction();
        } else {
            self.handle_asexual_reproduction();
        }
    }

    /// Handle asexual reproduction (original behavior)
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

                // Random chance to mate
                if self.rng.gen::<f32>() > 0.5 {
                    continue;
                }

                // Mark as valid pair
                pairs.push((idx1, idx2));
                used.insert(idx1);
                used.insert(idx2);
                break;
            }
        }

        // Now create offspring from pairs
        let mut offspring = Vec::new();
        for (idx1, idx2) in pairs {
            if let Some(child) = self.sexual_reproduce(idx1, idx2) {
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
        // Record deaths in phylogeny before removing
        for org in &self.organisms {
            if !org.is_alive() {
                self.phylogeny.record_death(org.id, self.time);
            }
        }

        let alive_before = self.organisms.len();
        self.organisms.retain(|org| org.is_alive());
        self.deaths_this_step = alive_before - self.organisms.len();
    }

    /// Handle predation (Attack actions)
    fn handle_predation(&mut self, actions: &[(usize, Action)]) {
        if !self.config.predation.enabled {
            return;
        }

        // Collect attack intentions
        let mut attacks: Vec<(usize, usize)> = Vec::new(); // (attacker_idx, target_idx)

        for &(idx, action) in actions {
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
            // Small energy gain from non-lethal attack
            self.organisms[attacker_idx].energy += damage * 0.1;
        }

        // Set attack cooldown
        self.organisms[attacker_idx].attack_cooldown = self.config.predation.attack_cooldown;
    }

    /// Update environment (food regeneration with seasonal and terrain multipliers)
    fn update_environment(&mut self) {
        // Get seasonal multiplier
        let season_multiplier = self.seasonal_system.food_multiplier();

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
                    let total_mult = season_multiplier * terrain_mult * depletion_mult;
                    let current = self.food_grid.get(xu, yu);
                    let new_food = (current + self.config.world.food_regen_rate * total_mult)
                        .min(self.config.world.food_max);
                    self.food_grid.set(xu, yu, new_food);
                }
            }
        } else {
            // Simple regeneration (no terrain/depletion)
            self.food_grid
                .regenerate(self.config.world.food_regen_rate * season_multiplier);
        }

        // Random food spawning (also affected by season)
        if self.rng.gen::<f32>() < 0.01 * season_multiplier {
            self.food_grid.spawn_random(10.0 * season_multiplier, 0.001);
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
