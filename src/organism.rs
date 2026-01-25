//! Organism structure and behavior.

use crate::config::Config;
use crate::ecology::food_types::DietSpecialization;
use crate::genetics::Sex;
use crate::grid::{FoodGrid, SpatialIndex};
use crate::neural::NeuralNet;
use serde::{Deserialize, Serialize};

/// Unique organism identifier
pub type OrganismId = u64;

/// Lineage/family identifier
pub type LineageId = u32;

/// Possible actions an organism can take
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Action {
    MoveNorth,
    MoveEast,
    MoveSouth,
    MoveWest,
    Eat,
    Reproduce,
    Attack,
    Signal(f32),
    Wait,
}

/// Result of an action attempt
#[derive(Debug, Clone, Copy)]
pub enum ActionResult {
    Success,
    Failed(FailReason),
}

/// Reasons why an action might fail
#[derive(Debug, Clone, Copy)]
pub enum FailReason {
    Blocked,
    NoFood,
    NoEnergy,
    NoTarget,
    NotReady,
    OutOfBounds,
}

/// Cause of death tracking
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeathCause {
    Starvation,
    Predation,
    OldAge,
}

/// An organism in the simulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Organism {
    // Identity
    pub id: OrganismId,
    pub lineage_id: LineageId,
    pub generation: u16,

    // Physical state
    pub x: u8,
    pub y: u8,
    pub size: f32,
    pub energy: f32,
    pub health: f32,
    pub age: u32,

    // Neural network brain
    pub brain: NeuralNet,

    // Short-term memory
    pub memory: [f32; 5],

    // Statistics
    pub kills: u16,
    pub offspring_count: u16,
    pub food_eaten: u32,

    // Behavior
    pub is_predator: bool,
    pub signal: f32,

    // Internal state
    pub last_action: Option<Action>,

    // Fase 2: Diet specialization
    pub diet: DietSpecialization,

    // Fase 2: Attack cooldown
    pub attack_cooldown: u32,

    // Fase 2: Death tracking
    pub cause_of_death: Option<DeathCause>,

    // Fase 2 Week 2: Terrain adaptation
    pub is_aquatic: bool,

    // Fase 2 Week 3-4: Sexual reproduction
    pub sex: Sex,
    pub parent1_id: Option<OrganismId>,
    pub parent2_id: Option<OrganismId>,
    pub mate_cooldown: u32,
}

impl Organism {
    /// Create a new organism with default values
    pub fn new(id: OrganismId, lineage_id: LineageId, x: u8, y: u8, config: &Config) -> Self {
        let brain = if config.neural.use_instincts {
            NeuralNet::new_with_instincts(config.neural.n_inputs, config.neural.n_outputs)
        } else {
            NeuralNet::new_minimal(config.neural.n_inputs, config.neural.n_outputs)
        };

        Self {
            id,
            lineage_id,
            generation: 0,
            x,
            y,
            size: 1.0,
            energy: config.organisms.initial_energy,
            health: 100.0,
            age: 0,
            brain,
            memory: [0.0; 5],
            kills: 0,
            offspring_count: 0,
            food_eaten: 0,
            is_predator: false,
            signal: 0.0,
            last_action: None,
            diet: DietSpecialization::random(),
            attack_cooldown: 0,
            cause_of_death: None,
            is_aquatic: false, // Organisms start as land-based
            sex: Sex::random(),
            parent1_id: None,
            parent2_id: None,
            mate_cooldown: 0,
        }
    }

    /// Sense the environment and return input vector for neural network
    pub fn sense(
        &self,
        food_grid: &FoodGrid,
        spatial_index: &SpatialIndex,
        organisms: &[Organism],
        time: u64,
        config: &Config,
    ) -> [f32; 20] {
        let mut inputs = [0.0f32; 20];
        let sense_range = 3u8;

        // Food in 4 directions (normalized)
        let food_scale = config.world.food_max * sense_range as f32;
        inputs[0] = food_grid.sense_direction(self.x, self.y, 0, -1, sense_range) / food_scale; // North
        inputs[1] = food_grid.sense_direction(self.x, self.y, 1, 0, sense_range) / food_scale; // East
        inputs[2] = food_grid.sense_direction(self.x, self.y, 0, 1, sense_range) / food_scale; // South
        inputs[3] = food_grid.sense_direction(self.x, self.y, -1, 0, sense_range) / food_scale; // West

        // Scan for nearby organisms
        let neighbor_indices = spatial_index.query_neighbors(self.x, self.y, sense_range);
        let mut threats = 0;
        let mut potential_mates = 0;

        for &idx in &neighbor_indices {
            if idx < organisms.len() {
                let other = &organisms[idx];
                if other.is_alive() && other.id != self.id {
                    if other.is_predator || other.size > self.size * 1.2 {
                        threats += 1;
                    }
                    if other.energy > config.organisms.reproduction_threshold {
                        potential_mates += 1;
                    }
                }
            }
        }

        inputs[4] = (threats as f32 / 10.0).min(1.0); // Threat level
        inputs[5] = (potential_mates as f32 / 10.0).min(1.0); // Mate availability

        // Internal state (normalized)
        inputs[6] = (self.energy / config.safety.max_energy).clamp(0.0, 1.0);
        inputs[7] = self.health / 100.0;
        inputs[8] = self.size / 5.0;
        inputs[9] = (self.age as f32 / config.safety.max_age as f32).min(1.0);

        // Memory
        inputs[10..15].copy_from_slice(&self.memory);

        // Constants and environmental
        inputs[15] = 1.0; // Bias
        inputs[16] = (time % 1000) as f32 / 1000.0; // Time of day
        inputs[17] = food_grid.get(self.x, self.y) / config.world.food_max; // Food at current position
        inputs[18] = if spatial_index.is_occupied(self.x, self.y) { 1.0 } else { 0.0 };
        inputs[19] = self.signal; // Own signal

        inputs
    }

    /// Process inputs through neural network
    #[inline]
    pub fn think(&mut self, inputs: &[f32; 20]) -> [f32; 10] {
        let outputs = self.brain.forward(inputs);
        let mut result = [0.0f32; 10];
        for (i, &val) in outputs.iter().take(10).enumerate() {
            result[i] = val;
        }
        result
    }

    /// Decide action based on neural network outputs
    pub fn decide_action(&self, outputs: &[f32; 10]) -> Action {
        // Find max output index
        let mut max_idx = 0;
        let mut max_val = outputs[0];

        for (i, &val) in outputs.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        match max_idx {
            0 => Action::MoveNorth,
            1 => Action::MoveEast,
            2 => Action::MoveSouth,
            3 => Action::MoveWest,
            4 => Action::Eat,
            5 => Action::Reproduce,
            6 => Action::Attack,
            7 => Action::Signal(outputs[7]),
            8 => Action::Wait,
            _ => Action::Wait,
        }
    }

    /// Try to move in a direction
    pub fn try_move(
        &mut self,
        dx: i8,
        dy: i8,
        _spatial_index: &SpatialIndex, // Reserved for collision detection
        grid_size: usize,
        move_cost: f32,
    ) -> ActionResult {
        let new_x = (self.x as i16 + dx as i16) as i16;
        let new_y = (self.y as i16 + dy as i16) as i16;

        // Check bounds
        if new_x < 0 || new_x >= grid_size as i16 || new_y < 0 || new_y >= grid_size as i16 {
            return ActionResult::Failed(FailReason::OutOfBounds);
        }

        let new_x = new_x as u8;
        let new_y = new_y as u8;

        // Check if blocked (for now, allow overlap)
        // In future: check spatial_index.is_occupied(new_x, new_y)

        // Apply move cost
        self.energy -= move_cost;

        // Update position
        self.x = new_x;
        self.y = new_y;

        ActionResult::Success
    }

    /// Try to eat food at current position
    pub fn try_eat(&mut self, food_grid: &mut FoodGrid, food_energy: f32) -> ActionResult {
        let available = food_grid.get(self.x, self.y);
        if available > 0.1 {
            let consumed = food_grid.consume(self.x, self.y, food_energy);
            self.energy += consumed;
            self.food_eaten += 1;
            ActionResult::Success
        } else {
            ActionResult::Failed(FailReason::NoFood)
        }
    }

    /// Update organism state (age, metabolism)
    pub fn update(&mut self, config: &Config) {
        self.age += 1;

        // Metabolic cost
        let base_cost = config.organisms.metabolism_base;
        let size_cost = self.size * 0.5;
        let brain_cost = self.brain.complexity() as f32 * 0.01; // Was 0.1 - too high, killed brain evolution

        self.energy -= base_cost + size_cost + brain_cost;

        // Health decay when starving
        if self.energy < 0.0 {
            self.health -= 5.0;
            if self.health <= 0.0 && self.cause_of_death.is_none() {
                self.cause_of_death = Some(DeathCause::Starvation);
            }
        }

        // Age cap
        if self.age >= config.safety.max_age {
            self.health = 0.0;
            if self.cause_of_death.is_none() {
                self.cause_of_death = Some(DeathCause::OldAge);
            }
        }

        // Energy cap
        self.energy = self.energy.min(config.safety.max_energy);

        // Update attack cooldown
        if self.attack_cooldown > 0 {
            self.attack_cooldown -= 1;
        }

        // Update mate cooldown
        if self.mate_cooldown > 0 {
            self.mate_cooldown -= 1;
        }

        // Update memory (shift and decay)
        for i in (1..5).rev() {
            self.memory[i] = self.memory[i - 1] * 0.9;
        }
        self.memory[0] = self.energy / config.safety.max_energy;
    }

    /// Check if organism is alive
    #[inline]
    pub fn is_alive(&self) -> bool {
        self.health > 0.0 && self.energy > -50.0
    }

    /// Check if organism can reproduce (asexual)
    #[inline]
    pub fn can_reproduce(&self, config: &Config) -> bool {
        self.energy >= config.organisms.reproduction_threshold && self.health > 50.0
    }

    /// Check if organism can mate (sexual reproduction)
    #[inline]
    pub fn can_mate(&self, config: &Config) -> bool {
        self.energy >= config.reproduction.min_energy
            && self.health > 50.0
            && self.mate_cooldown == 0
    }

    /// Calculate distance to another organism
    #[inline]
    pub fn distance_to(&self, other: &Organism) -> u8 {
        let dx = (self.x as i16 - other.x as i16).unsigned_abs() as u8;
        let dy = (self.y as i16 - other.y as i16).unsigned_abs() as u8;
        dx.max(dy)
    }

    /// Create offspring from this organism
    pub fn reproduce(&mut self, child_id: OrganismId, config: &Config) -> Option<Organism> {
        if !self.can_reproduce(config) {
            return None;
        }

        // Pay reproduction cost
        self.energy -= config.organisms.reproduction_cost;
        self.offspring_count += 1;

        // Create child
        let mut child = Organism {
            id: child_id,
            lineage_id: self.lineage_id,
            generation: self.generation + 1,
            x: self.x,
            y: self.y,
            size: self.size,
            energy: config.organisms.reproduction_cost * 0.8, // 80% of cost goes to child
            health: 100.0,
            age: 0,
            brain: self.brain.clone(),
            memory: [0.0; 5],
            kills: 0,
            offspring_count: 0,
            food_eaten: 0,
            is_predator: self.is_predator,
            signal: 0.0,
            last_action: None,
            diet: self.diet.clone(),
            attack_cooldown: 0,
            cause_of_death: None,
            is_aquatic: self.is_aquatic,
            sex: Sex::random(),
            parent1_id: Some(self.id),
            parent2_id: None, // Asexual reproduction - single parent
            mate_cooldown: 0,
        };

        // Mutate diet slightly
        child.diet.mutate(config.evolution.mutation_strength);

        // Small chance to mutate aquatic trait
        if rand::random::<f32>() < 0.01 {
            child.is_aquatic = !child.is_aquatic;
        }

        // Mutate child's brain
        let mutation_config = crate::neural::MutationConfig {
            weight_mutation_rate: config.evolution.mutation_rate,
            weight_mutation_strength: config.evolution.mutation_strength,
            add_neuron_rate: config.evolution.add_neuron_rate,
            add_connection_rate: config.evolution.add_connection_rate,
            max_neurons: config.safety.max_neurons,
        };
        child.brain.mutate(&mutation_config);

        // Position child near parent
        let grid_size = config.world.grid_size as u8;
        child.x = (self.x + 1).min(grid_size - 1);

        Some(child)
    }

    /// Get fitness score (for selection/crossover)
    pub fn fitness(&self) -> f32 {
        let survival_score = self.age as f32;
        let reproduction_score = self.offspring_count as f32 * 100.0;
        let food_score = self.food_eaten as f32 * 10.0;

        survival_score + reproduction_score + food_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config::default()
    }

    #[test]
    fn test_organism_creation() {
        let config = test_config();
        let org = Organism::new(1, 1, 10, 10, &config);

        assert_eq!(org.id, 1);
        assert_eq!(org.x, 10);
        assert_eq!(org.y, 10);
        assert!(org.is_alive());
    }

    #[test]
    fn test_organism_sense() {
        let config = test_config();
        let org = Organism::new(1, 1, 40, 40, &config);

        let food_grid = FoodGrid::new(config.world.grid_size, config.world.food_max);
        let spatial_index = SpatialIndex::new(config.world.grid_size);
        let organisms = vec![org.clone()];

        let inputs = org.sense(&food_grid, &spatial_index, &organisms, 0, &config);

        assert_eq!(inputs.len(), 20);
        assert!(inputs.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_organism_think_decide() {
        let config = test_config();
        let mut org = Organism::new(1, 1, 40, 40, &config);

        let inputs = [0.5f32; 20];
        let outputs = org.think(&inputs);
        let action = org.decide_action(&outputs);

        // Should produce a valid action
        match action {
            Action::MoveNorth
            | Action::MoveEast
            | Action::MoveSouth
            | Action::MoveWest
            | Action::Eat
            | Action::Reproduce
            | Action::Attack
            | Action::Signal(_)
            | Action::Wait => {}
        }
    }

    #[test]
    fn test_organism_movement() {
        let config = test_config();
        let mut org = Organism::new(1, 1, 40, 40, &config);
        let spatial_index = SpatialIndex::new(config.world.grid_size);

        let result = org.try_move(1, 0, &spatial_index, config.world.grid_size, config.organisms.move_cost);

        assert!(matches!(result, ActionResult::Success));
        assert_eq!(org.x, 41);
        assert_eq!(org.y, 40);
    }

    #[test]
    fn test_organism_eating() {
        let config = test_config();
        let mut org = Organism::new(1, 1, 40, 40, &config);
        let mut food_grid = FoodGrid::new(config.world.grid_size, config.world.food_max);

        food_grid.set(40, 40, 30.0);
        let initial_energy = org.energy;

        let result = org.try_eat(&mut food_grid, config.organisms.food_energy);

        assert!(matches!(result, ActionResult::Success));
        assert!(org.energy > initial_energy);
    }

    #[test]
    fn test_organism_reproduction() {
        let config = test_config();
        let mut org = Organism::new(1, 1, 40, 40, &config);
        org.energy = 100.0; // Enough to reproduce

        let child = org.reproduce(2, &config);

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.generation, 1);
        assert_eq!(child.lineage_id, org.lineage_id);
        assert_eq!(org.offspring_count, 1);
    }

    #[test]
    fn test_organism_death() {
        let config = test_config();
        let mut org = Organism::new(1, 1, 40, 40, &config);

        org.health = 0.0;
        assert!(!org.is_alive());

        org.health = 50.0;
        org.energy = -100.0;
        assert!(!org.is_alive());
    }
}
