//! Organism structure and behavior.

use crate::config::Config;
use crate::ecology::food_types::DietSpecialization;
use crate::ecology::large_prey::{CooperationSignal, TrustRelationship};
use crate::ecology::predation::PredatorObservation;
use crate::genetics::Sex;
use crate::grid::{FoodGrid, SpatialIndex};
use crate::neural::NeuralNet;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

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
    // Social Communication actions (Cognitive Tasks)
    SignalDanger,   // Warn nearby organisms of predators
    SignalFood,     // Signal rich food source found
    // B3: Multi-Agent Coordination actions
    ProposeCooperation,  // Propose cooperative hunt
    AcceptCooperation,   // Accept cooperation proposal
    RejectCooperation,   // Reject cooperation proposal
    AttackLargePrey,     // Attack large prey (requires 2+ attackers)
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

/// Cognitive inputs for advanced brain evolution
/// These inputs require integration over time/space - forcing hidden layer evolution
#[derive(Debug, Clone, Default)]
pub struct CognitiveInputs {
    // Spatial Memory: Depletion in 8 directions (recently eaten cells)
    pub depletion_n: f32,
    pub depletion_ne: f32,
    pub depletion_e: f32,
    pub depletion_se: f32,
    pub depletion_s: f32,
    pub depletion_sw: f32,
    pub depletion_w: f32,
    pub depletion_nw: f32,

    // Temporal Prediction
    pub season_progress: f32,    // 0.0-1.0, where in season cycle
    pub food_trend: f32,         // -1.0 to +1.0, food rising/falling
    pub time_to_season: f32,     // 0.0-1.0, time until next season change

    // Social Communication
    pub signal_danger: f32,      // 1.0 if nearby organism signals danger
    pub signal_food: f32,        // 1.0 if nearby organism signals food found
    pub signal_help: f32,        // 1.0 if nearby organism signals help needed

    // B1: Sequential Memory (Path Tracking) - 10 new inputs
    pub loop_detected: f32,      // 1.0 if organism visited same position twice recently
    pub oscillation: f32,        // 1.0 if A-B-A pattern detected
    pub path_entropy: f32,       // 0.0-1.0, diversity of movement directions
    pub movement_bias_x: f32,    // -1.0 to +1.0, tendency to move E/W
    pub movement_bias_y: f32,    // -1.0 to +1.0, tendency to move N/S
    // Recent directions (one-hot encoded): N, E, S, W, None
    pub recent_dir_n: f32,       // 1.0 if last move was North
    pub recent_dir_e: f32,       // 1.0 if last move was East
    pub recent_dir_s: f32,       // 1.0 if last move was South
    pub recent_dir_w: f32,       // 1.0 if last move was West
    pub recent_dir_none: f32,    // 1.0 if no recent movement

    // B2: Pattern Recognition (Predator Strategies) - 12 new inputs
    pub pred_movement_variance: f32,     // Predator movement unpredictability (0-1)
    pub pred_speed: f32,                 // Predator average speed (0-1 normalized)
    pub pred_direction_consistency: f32, // How consistent is predator's direction (0-1)
    // Strategy classification (one-hot): Random, Patrol, Chase, Ambush
    pub pred_strategy_random: f32,
    pub pred_strategy_patrol: f32,
    pub pred_strategy_chase: f32,
    pub pred_strategy_ambush: f32,
    pub pred_classification_confidence: f32, // Confidence in classification (0-1)
    pub pred_approach_angle: f32,        // How directly predator approaches (0-1)
    pub pred_relative_speed: f32,        // Speed relative to self (-1 to +1)
    pub pred_time_observed: f32,         // How long we've been tracking (0-1)
    pub pred_threat_level: f32,          // Combined threat assessment (0-1)

    // B3: Multi-Agent Coordination - 15 new inputs
    pub large_prey_nearby: f32,          // 1.0 if large prey in sensing range
    pub large_prey_distance: f32,        // Normalized distance to nearest large prey (0-1)
    pub large_prey_health: f32,          // Normalized health of target (0-1)
    pub large_prey_attackers: f32,       // Current number of attackers (normalized)
    pub large_prey_need: f32,            // How many more attackers needed (0-1)
    pub partner_nearby: f32,             // 1.0 if potential cooperation partner nearby
    pub partner_trust: f32,              // Trust level with nearest partner (-1 to +1)
    pub partner_distance: f32,           // Distance to nearest potential partner (0-1)
    pub cooperation_proposed: f32,       // 1.0 if we received a cooperation proposal
    pub cooperation_active: f32,         // 1.0 if currently cooperating
    pub hunt_success_rate: f32,          // Our recent success rate at cooperative hunts (0-1)
    pub partner_fitness: f32,            // Estimated fitness of potential partner (0-1)
    pub own_attack_power: f32,           // Our own attack capability (0-1)
    pub time_since_last_coop: f32,       // Normalized time since last cooperation (0-1)
    pub prey_escape_urgency: f32,        // How close prey is to escaping (0-1)
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

    // Cognitive Tasks: Social Communication
    pub social_signal: SocialSignal,
    pub signal_cooldown: u32,

    // B1: Sequential Memory - Path History (max 5 positions)
    #[serde(skip)]
    pub path_history: VecDeque<(u8, u8)>,

    // B2: Pattern Recognition - Observed Predators (max 10 tracked)
    #[serde(skip)]
    pub observed_predators: HashMap<OrganismId, PredatorObservation>,

    // B3: Multi-Agent Coordination
    pub cooperation_signal: CooperationSignal,
    #[serde(skip)]
    pub trust_relationships: HashMap<OrganismId, TrustRelationship>,
    pub current_hunt_target: Option<u64>,  // Large prey ID being targeted
    pub coop_successes: u32,               // Lifetime cooperative hunt successes
    pub coop_failures: u32,                // Lifetime cooperative hunt failures
    pub last_coop_time: u64,               // Time of last cooperation

    // Phase 3: Lifetime Learning (Hebbian)
    pub last_reward: f32,
    pub total_lifetime_reward: f32,
    pub successful_forages: u32,
    pub failed_forages: u32,
}

/// Social signal types for communication between organisms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum SocialSignal {
    #[default]
    None,
    Danger,     // Predator nearby - flee!
    FoodFound,  // Rich food source - come here!
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
            social_signal: SocialSignal::None,
            signal_cooldown: 0,
            path_history: VecDeque::with_capacity(5),
            observed_predators: HashMap::with_capacity(10),
            cooperation_signal: CooperationSignal::None,
            trust_relationships: HashMap::with_capacity(10),
            current_hunt_target: None,
            coop_successes: 0,
            coop_failures: 0,
            last_coop_time: 0,
            last_reward: 0.0,
            total_lifetime_reward: 0.0,
            successful_forages: 0,
            failed_forages: 0,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // B1: SEQUENTIAL MEMORY (Path Tracking) Methods
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// Record current position in path history (called after movement)
    pub fn record_movement(&mut self) {
        let pos = (self.x, self.y);

        // Keep max 5 positions
        if self.path_history.len() >= 5 {
            self.path_history.pop_front();
        }
        self.path_history.push_back(pos);
    }

    /// Detect if organism has visited the same position twice in recent history
    pub fn detect_loop(&self) -> bool {
        if self.path_history.len() < 2 {
            return false;
        }

        let current = (self.x, self.y);

        // Check if current position appears in history (excluding most recent)
        for i in 0..self.path_history.len().saturating_sub(1) {
            if self.path_history[i] == current {
                return true;
            }
        }
        false
    }

    /// Detect A-B-A oscillation pattern (back-and-forth movement)
    pub fn detect_oscillation(&self) -> bool {
        if self.path_history.len() < 3 {
            return false;
        }

        let len = self.path_history.len();
        // Check if position[len-1] == position[len-3] (A-B-A pattern)
        self.path_history[len - 1] == self.path_history[len - 3]
    }

    /// Calculate path entropy (diversity of movement directions)
    /// Returns 0.0-1.0, where 1.0 = highly diverse, 0.0 = all same direction
    pub fn calculate_path_entropy(&self) -> f32 {
        if self.path_history.len() < 2 {
            return 0.5; // Neutral if not enough history
        }

        // Count direction changes: N, E, S, W
        let mut dir_counts = [0u32; 4]; // N, E, S, W

        for i in 1..self.path_history.len() {
            let (prev_x, prev_y) = self.path_history[i - 1];
            let (curr_x, curr_y) = self.path_history[i];

            let dx = curr_x as i16 - prev_x as i16;
            let dy = curr_y as i16 - prev_y as i16;

            // Classify direction
            if dy < 0 { dir_counts[0] += 1; } // North
            if dx > 0 { dir_counts[1] += 1; } // East
            if dy > 0 { dir_counts[2] += 1; } // South
            if dx < 0 { dir_counts[3] += 1; } // West
        }

        // Calculate Shannon entropy
        let total: u32 = dir_counts.iter().sum();
        if total == 0 {
            return 0.5;
        }

        let mut entropy = 0.0f32;
        for &count in &dir_counts {
            if count > 0 {
                let p = count as f32 / total as f32;
                entropy -= p * p.log2();
            }
        }

        // Normalize to 0-1 (max entropy for 4 categories is log2(4) = 2)
        (entropy / 2.0).clamp(0.0, 1.0)
    }

    /// Calculate movement bias (tendency to move in X/Y direction)
    /// Returns (bias_x, bias_y) where -1.0 = West/North, +1.0 = East/South
    pub fn calculate_movement_bias(&self) -> (f32, f32) {
        if self.path_history.len() < 2 {
            return (0.0, 0.0);
        }

        let mut total_dx = 0i32;
        let mut total_dy = 0i32;

        for i in 1..self.path_history.len() {
            let (prev_x, prev_y) = self.path_history[i - 1];
            let (curr_x, curr_y) = self.path_history[i];

            total_dx += curr_x as i32 - prev_x as i32;
            total_dy += curr_y as i32 - prev_y as i32;
        }

        let moves = (self.path_history.len() - 1) as f32;
        let bias_x = (total_dx as f32 / moves).clamp(-1.0, 1.0);
        let bias_y = (total_dy as f32 / moves).clamp(-1.0, 1.0);

        (bias_x, bias_y)
    }

    /// Get the last movement direction as one-hot encoded (N, E, S, W, None)
    pub fn get_recent_direction(&self) -> [f32; 5] {
        if self.path_history.len() < 2 {
            return [0.0, 0.0, 0.0, 0.0, 1.0]; // None
        }

        let len = self.path_history.len();
        let (prev_x, prev_y) = self.path_history[len - 2];
        let (curr_x, curr_y) = self.path_history[len - 1];

        let dx = curr_x as i16 - prev_x as i16;
        let dy = curr_y as i16 - prev_y as i16;

        // One-hot encode: [N, E, S, W, None]
        let mut result = [0.0f32; 5];

        if dy < 0 { result[0] = 1.0; } // North
        else if dx > 0 { result[1] = 1.0; } // East
        else if dy > 0 { result[2] = 1.0; } // South
        else if dx < 0 { result[3] = 1.0; } // West
        else { result[4] = 1.0; } // None (no movement)

        result
    }

    /// Sense the environment and return input vector for neural network
    /// Returns 75 inputs: 24 base + 8 memory + 3 temporal + 3 social + 10 sequential + 12 predator + 15 cooperation
    pub fn sense(
        &self,
        food_grid: &FoodGrid,
        spatial_index: &SpatialIndex,
        organisms: &[Organism],
        time: u64,
        config: &Config,
        cognitive: &CognitiveInputs,
    ) -> [f32; 75] {
        let mut inputs = [0.0f32; 75];
        let sense_range = 3u8;

        // Food in 4 directions (normalized)
        let food_scale = config.world.food_max * sense_range as f32;
        inputs[0] = food_grid.sense_direction(self.x, self.y, 0, -1, sense_range) / food_scale; // North
        inputs[1] = food_grid.sense_direction(self.x, self.y, 1, 0, sense_range) / food_scale; // East
        inputs[2] = food_grid.sense_direction(self.x, self.y, 0, 1, sense_range) / food_scale; // South
        inputs[3] = food_grid.sense_direction(self.x, self.y, -1, 0, sense_range) / food_scale; // West

        // Scan for nearby organisms and compute directional threats
        let neighbor_indices = spatial_index.query_neighbors(self.x, self.y, sense_range);
        let mut threats = 0;
        let mut potential_mates = 0;

        // Directional threat accumulators (N, E, S, W)
        let mut threat_north = 0.0f32;
        let mut threat_east = 0.0f32;
        let mut threat_south = 0.0f32;
        let mut threat_west = 0.0f32;

        for &idx in &neighbor_indices {
            if idx < organisms.len() {
                let other = &organisms[idx];
                if other.is_alive() && other.id != self.id {
                    let is_threat = other.is_predator || other.size > self.size * 1.2;

                    if is_threat {
                        threats += 1;

                        // Calculate direction and distance to threat
                        let dx = other.x as i16 - self.x as i16;
                        let dy = other.y as i16 - self.y as i16;
                        let dist = ((dx.abs()).max(dy.abs()) as f32).max(1.0);
                        let threat_intensity = 1.0 / dist; // Closer = more dangerous

                        // Accumulate threat in the appropriate direction
                        if dy < 0 { threat_north += threat_intensity; } // Threat to the north
                        if dx > 0 { threat_east += threat_intensity; }  // Threat to the east
                        if dy > 0 { threat_south += threat_intensity; } // Threat to the south
                        if dx < 0 { threat_west += threat_intensity; }  // Threat to the west
                    }
                    if other.energy > config.organisms.reproduction_threshold {
                        potential_mates += 1;
                    }
                }
            }
        }

        inputs[4] = (threats as f32 / 10.0).min(1.0); // Total threat level (global)
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

        // Directional threat sensors (normalized)
        inputs[20] = (threat_north / 3.0).min(1.0); // Threat from North
        inputs[21] = (threat_east / 3.0).min(1.0);  // Threat from East
        inputs[22] = (threat_south / 3.0).min(1.0); // Threat from South
        inputs[23] = (threat_west / 3.0).min(1.0);  // Threat from West

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // COGNITIVE INPUTS (require hidden layers to process effectively)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        // Spatial Memory: Food depletion in 8 directions (recently eaten = avoid)
        inputs[24] = cognitive.depletion_n;   // North depleted
        inputs[25] = cognitive.depletion_ne;  // NE depleted
        inputs[26] = cognitive.depletion_e;   // East depleted
        inputs[27] = cognitive.depletion_se;  // SE depleted
        inputs[28] = cognitive.depletion_s;   // South depleted
        inputs[29] = cognitive.depletion_sw;  // SW depleted
        inputs[30] = cognitive.depletion_w;   // West depleted
        inputs[31] = cognitive.depletion_nw;  // NW depleted

        // Temporal Prediction: Season awareness
        inputs[32] = cognitive.season_progress;   // Where in season cycle (0-1)
        inputs[33] = cognitive.food_trend;        // Food rising (+1) or falling (-1)
        inputs[34] = cognitive.time_to_season;    // Time until season change (0-1)

        // Social Communication: Nearby signals
        inputs[35] = cognitive.signal_danger;     // Danger signal nearby
        inputs[36] = cognitive.signal_food;       // Food found signal nearby
        inputs[37] = cognitive.signal_help;       // Help needed signal nearby

        // B1: Sequential Memory (Path Tracking)
        inputs[38] = cognitive.loop_detected;     // Loop detected (visited same pos twice)
        inputs[39] = cognitive.oscillation;       // A-B-A oscillation pattern
        inputs[40] = cognitive.path_entropy;      // Movement diversity (0-1)
        inputs[41] = cognitive.movement_bias_x;   // Tendency E/W (-1 to +1)
        inputs[42] = cognitive.movement_bias_y;   // Tendency N/S (-1 to +1)
        inputs[43] = cognitive.recent_dir_n;      // Last move was North
        inputs[44] = cognitive.recent_dir_e;      // Last move was East
        inputs[45] = cognitive.recent_dir_s;      // Last move was South
        inputs[46] = cognitive.recent_dir_w;      // Last move was West
        inputs[47] = cognitive.recent_dir_none;   // No recent movement

        // B2: Pattern Recognition (Predator Strategies)
        inputs[48] = cognitive.pred_movement_variance;      // Predator unpredictability
        inputs[49] = cognitive.pred_speed;                  // Predator speed (normalized)
        inputs[50] = cognitive.pred_direction_consistency;  // Direction consistency
        inputs[51] = cognitive.pred_strategy_random;        // Strategy: Random
        inputs[52] = cognitive.pred_strategy_patrol;        // Strategy: Patrol
        inputs[53] = cognitive.pred_strategy_chase;         // Strategy: Chase
        inputs[54] = cognitive.pred_strategy_ambush;        // Strategy: Ambush
        inputs[55] = cognitive.pred_classification_confidence; // Classification confidence
        inputs[56] = cognitive.pred_approach_angle;         // Approach angle
        inputs[57] = cognitive.pred_relative_speed;         // Relative speed
        inputs[58] = cognitive.pred_time_observed;          // Observation duration
        inputs[59] = cognitive.pred_threat_level;           // Combined threat level

        // B3: Multi-Agent Coordination
        inputs[60] = cognitive.large_prey_nearby;           // Large prey present
        inputs[61] = cognitive.large_prey_distance;         // Distance to large prey
        inputs[62] = cognitive.large_prey_health;           // Large prey health
        inputs[63] = cognitive.large_prey_attackers;        // Current attackers
        inputs[64] = cognitive.large_prey_need;             // Attackers still needed
        inputs[65] = cognitive.partner_nearby;              // Potential partner nearby
        inputs[66] = cognitive.partner_trust;               // Trust with partner
        inputs[67] = cognitive.partner_distance;            // Distance to partner
        inputs[68] = cognitive.cooperation_proposed;        // Received proposal
        inputs[69] = cognitive.cooperation_active;          // Currently cooperating
        inputs[70] = cognitive.hunt_success_rate;           // Our success rate
        inputs[71] = cognitive.partner_fitness;             // Partner's fitness
        inputs[72] = cognitive.own_attack_power;            // Our attack power
        inputs[73] = cognitive.time_since_last_coop;        // Time since last coop
        inputs[74] = cognitive.prey_escape_urgency;         // Prey escape urgency

        inputs
    }

    /// Process inputs through neural network
    #[inline]
    pub fn think(&mut self, inputs: &[f32; 75]) -> [f32; 15] {
        let outputs = self.brain.forward(inputs);
        let mut result = [0.0f32; 15];
        for (i, &val) in outputs.iter().take(15).enumerate() {
            result[i] = val;
        }
        result
    }

    /// Decide action based on neural network outputs (15 outputs)
    pub fn decide_action(&self, outputs: &[f32; 15]) -> Action {
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
            9 => Action::SignalDanger,   // Social communication
            10 => Action::SignalFood,    // Social communication
            11 => Action::ProposeCooperation,  // B3: Cooperation actions
            12 => Action::AcceptCooperation,
            13 => Action::RejectCooperation,
            14 => Action::AttackLargePrey,
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
        let size_cost = self.size * 0.1;

        // Brain complexity BONUS with HARD CAP + PENALTY SYSTEM
        // Prevents runaway growth while rewarding moderate complexity
        // Sweet spot: 5-7 layers
        let brain_complexity = self.brain.complexity();
        let has_learning = self.brain.has_learning();

        let brain_bonus = if brain_complexity == 0 {
            0.0
        } else if !has_learning {
            // NO LEARNING: existing penalty system
            if brain_complexity <= 7 {
                0.03 * brain_complexity as f32
            } else if brain_complexity <= 10 {
                0.21 + 0.01 * (brain_complexity - 7) as f32
            } else if brain_complexity <= 12 {
                0.24
            } else {
                (0.24 - 0.05 * (brain_complexity - 12) as f32).max(-0.30)
            }
        } else {
            // WITH LEARNING: reduced bonus, earlier penalty
            // Learning provides adaptation, not raw size
            if brain_complexity <= 5 {
                0.02 * brain_complexity as f32  // Max 0.10
            } else if brain_complexity <= 8 {
                0.10 + 0.005 * (brain_complexity - 5) as f32  // Max 0.115
            } else if brain_complexity <= 10 {
                0.115  // Capped
            } else {
                // Penalty: 11→0.075, 13→-0.005, 15→-0.085
                (0.115 - 0.04 * (brain_complexity - 10) as f32).max(-0.30)
            }
        };

        // Ensure minimum metabolism of 0.08 to maintain population stability
        self.energy -= (base_cost + size_cost - brain_bonus).max(0.08);

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

        // Update social signal cooldown
        if self.signal_cooldown > 0 {
            self.signal_cooldown -= 1;
            if self.signal_cooldown == 0 {
                self.social_signal = SocialSignal::None;
            }
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
            social_signal: SocialSignal::None,
            signal_cooldown: 0,
            path_history: VecDeque::with_capacity(5),
            observed_predators: HashMap::with_capacity(10),
            cooperation_signal: CooperationSignal::None,
            trust_relationships: HashMap::with_capacity(10),
            current_hunt_target: None,
            coop_successes: 0,
            coop_failures: 0,
            last_coop_time: 0,
            last_reward: 0.0,
            total_lifetime_reward: 0.0,
            successful_forages: 0,
            failed_forages: 0,
        };

        // Mutate diet slightly
        child.diet.mutate(config.evolution.mutation_strength);

        // Small chance to mutate aquatic trait
        if rand::random::<f32>() < 0.01 {
            child.is_aquatic = !child.is_aquatic;
        }

        // Mutate child's brain (predators get boosted structural mutations)
        let predator_boost = if self.is_predator { 1.5 } else { 1.0 };
        let mutation_config = crate::neural::MutationConfig {
            weight_mutation_rate: config.evolution.mutation_rate,
            weight_mutation_strength: config.evolution.mutation_strength,
            add_neuron_rate: config.evolution.add_neuron_rate * predator_boost,
            add_connection_rate: config.evolution.add_connection_rate * predator_boost,
            max_neurons: config.safety.max_neurons,
        };
        child.brain.mutate(&mutation_config);

        // Position child near parent
        let grid_size = config.world.grid_size as u8;
        child.x = (self.x + 1).min(grid_size - 1);

        Some(child)
    }

    /// Apply Hebbian learning update with reward signal
    pub fn apply_learning_update(&mut self, reward: f32) {
        self.last_reward = reward;
        self.total_lifetime_reward += reward;
        if reward > 0.0 {
            self.successful_forages += 1;
        } else if reward < 0.0 {
            self.failed_forages += 1;
        }
        self.brain.learn(reward);
    }

    /// Get fitness score (for selection/crossover)
    /// HARSH MODE: Brain complexity and cooperation are heavily rewarded
    pub fn fitness(&self) -> f32 {
        let survival_score = self.age as f32;
        let reproduction_score = self.offspring_count as f32 * 100.0;
        let food_score = self.food_eaten as f32 * 10.0;
        let predation_score = self.kills as f32 * 50.0; // Reward successful hunting

        // HARSH MODE: Strong bonus for brain complexity
        // +100 fitness per hidden layer (was 50) - makes complex brains highly advantageous
        let brain_complexity = self.brain.complexity() as f32;
        let brain_bonus = brain_complexity * 100.0;

        // HARSH MODE: Bonus for successful cooperation
        // +150 per successful cooperative hunt (was 75) - makes cooperation essential
        let coop_bonus = self.coop_successes as f32 * 150.0;

        // HARSH MODE: Penalty for failed cooperation attempts
        // -50 per failed solo attack on large prey - discourages foolish behavior
        let coop_penalty = self.coop_failures as f32 * 50.0;

        // HARSH MODE: Efficiency bonus based on food eaten vs age
        // Organisms that eat more efficiently get higher fitness
        let efficiency_bonus = if self.age > 0 {
            (self.food_eaten as f32 / self.age as f32) * 50.0
        } else {
            0.0
        };

        let base_fitness = survival_score + reproduction_score + food_score + predation_score;
        let harsh_adjustments = brain_bonus + coop_bonus + efficiency_bonus - coop_penalty;

        (base_fitness + harsh_adjustments).max(0.0)
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
        let cognitive = CognitiveInputs::default();

        let inputs = org.sense(&food_grid, &spatial_index, &organisms, 0, &config, &cognitive);

        assert_eq!(inputs.len(), 75); // 24 base + 8 memory + 3 temporal + 3 social + 10 sequential + 12 predator + 15 cooperation
        assert!(inputs.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_organism_think_decide() {
        let config = test_config();
        let mut org = Organism::new(1, 1, 40, 40, &config);

        let inputs = [0.5f32; 75]; // 75 inputs with cognitive + sequential + predator + cooperation
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
            | Action::Wait
            | Action::SignalDanger
            | Action::SignalFood
            | Action::ProposeCooperation
            | Action::AcceptCooperation
            | Action::RejectCooperation
            | Action::AttackLargePrey => {}
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
