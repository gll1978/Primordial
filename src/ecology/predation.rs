//! Predation system - attack and kill mechanics.
//! Includes B2: Predator Strategy Pattern Recognition.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// B2: PREDATOR STRATEGIES - Pattern Recognition System
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Predator behavior strategies that organisms can recognize
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PredatorStrategy {
    #[default]
    Random,     // Unpredictable movement
    Patrol,     // Fixed pattern (e.g., N-E-S-W cycle)
    Chase,      // Actively pursuing prey
    Ambush,     // Stays still, attacks when close
}

/// Observation data about a predator (used for pattern recognition)
#[derive(Debug, Clone)]
pub struct PredatorObservation {
    /// Recent positions observed (max 10)
    pub positions: VecDeque<(u8, u8)>,
    /// Time of last observation
    pub last_seen: u64,
    /// Accumulated movement variance
    pub movement_variance: f32,
    /// Average speed (cells per step)
    pub average_speed: f32,
    /// Direction consistency (0-1, higher = more consistent direction)
    pub direction_consistency: f32,
    /// Classified strategy
    pub classified_strategy: PredatorStrategy,
    /// Classification confidence (0-1)
    pub confidence: f32,
    /// Time spent stationary
    pub stationary_time: u32,
    /// Is actively chasing this observer
    pub is_chasing: bool,
}

impl PredatorObservation {
    /// Create new observation
    pub fn new(x: u8, y: u8, time: u64) -> Self {
        let mut positions = VecDeque::with_capacity(10);
        positions.push_back((x, y));

        Self {
            positions,
            last_seen: time,
            movement_variance: 0.0,
            average_speed: 0.0,
            direction_consistency: 0.0,
            classified_strategy: PredatorStrategy::Random,
            confidence: 0.0,
            stationary_time: 0,
            is_chasing: false,
        }
    }

    /// Update observation with new position
    pub fn update(&mut self, x: u8, y: u8, time: u64, observer_x: u8, observer_y: u8) {
        // Check if moved
        if let Some(&(last_x, last_y)) = self.positions.back() {
            if last_x == x && last_y == y {
                self.stationary_time += 1;
            } else {
                self.stationary_time = 0;
            }
        }

        // Add new position
        if self.positions.len() >= 10 {
            self.positions.pop_front();
        }
        self.positions.push_back((x, y));
        self.last_seen = time;

        // Recalculate metrics if we have enough data
        if self.positions.len() >= 3 {
            self.calculate_metrics(observer_x, observer_y);
            self.classify_strategy();
        }
    }

    /// Calculate movement metrics from position history
    fn calculate_metrics(&mut self, observer_x: u8, observer_y: u8) {
        if self.positions.len() < 2 {
            return;
        }

        let mut total_distance = 0.0f32;
        let mut dx_sum = 0i32;
        let mut dy_sum = 0i32;
        let mut directions: Vec<(i32, i32)> = Vec::new();

        for i in 1..self.positions.len() {
            let (prev_x, prev_y) = self.positions[i - 1];
            let (curr_x, curr_y) = self.positions[i];

            let dx = curr_x as i32 - prev_x as i32;
            let dy = curr_y as i32 - prev_y as i32;

            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            total_distance += dist;

            dx_sum += dx;
            dy_sum += dy;
            directions.push((dx, dy));
        }

        let moves = (self.positions.len() - 1) as f32;

        // Average speed
        self.average_speed = total_distance / moves;

        // Movement variance (how unpredictable)
        if directions.len() >= 2 {
            let mean_dx = dx_sum as f32 / moves;
            let mean_dy = dy_sum as f32 / moves;

            let mut variance = 0.0f32;
            for (dx, dy) in &directions {
                let diff_x = *dx as f32 - mean_dx;
                let diff_y = *dy as f32 - mean_dy;
                variance += diff_x * diff_x + diff_y * diff_y;
            }
            self.movement_variance = (variance / moves).sqrt();
        }

        // Direction consistency (are they moving in a consistent direction?)
        let total_dx = dx_sum.abs() as f32;
        let total_dy = dy_sum.abs() as f32;
        let total_movement = total_dx + total_dy;

        if total_movement > 0.0 {
            // How much of the total movement was in a consistent direction
            let net_movement = ((dx_sum * dx_sum + dy_sum * dy_sum) as f32).sqrt();
            self.direction_consistency = (net_movement / total_movement).clamp(0.0, 1.0);
        }

        // Check if chasing (moving toward observer)
        if let Some(&(curr_x, curr_y)) = self.positions.back() {
            if self.positions.len() >= 2 {
                let (prev_x, prev_y) = self.positions[self.positions.len() - 2];

                let prev_dist_to_obs = ((prev_x as i32 - observer_x as i32).pow(2)
                    + (prev_y as i32 - observer_y as i32).pow(2)) as f32;
                let curr_dist_to_obs = ((curr_x as i32 - observer_x as i32).pow(2)
                    + (curr_y as i32 - observer_y as i32).pow(2)) as f32;

                // If consistently getting closer
                self.is_chasing = curr_dist_to_obs < prev_dist_to_obs && self.average_speed > 0.5;
            }
        }
    }

    /// Classify the predator's strategy based on observed behavior
    fn classify_strategy(&mut self) {
        // Classification rules:
        // - Ambush: stationary_time > 5 or very low speed
        // - Chase: is_chasing and high speed
        // - Patrol: consistent direction, moderate speed, low variance
        // - Random: high variance, inconsistent direction

        let is_stationary = self.stationary_time > 5 || self.average_speed < 0.1;
        let is_fast = self.average_speed > 0.7;
        let is_consistent = self.direction_consistency > 0.6;
        let is_variable = self.movement_variance > 1.0;

        // Ambush: Stationary predator
        if is_stationary {
            self.classified_strategy = PredatorStrategy::Ambush;
            self.confidence = 0.7 + (self.stationary_time as f32 / 20.0).min(0.3);
            return;
        }

        // Chase: Fast, moving toward observer
        if self.is_chasing && is_fast {
            self.classified_strategy = PredatorStrategy::Chase;
            self.confidence = 0.6 + (self.average_speed / 2.0).min(0.4);
            return;
        }

        // Patrol: Consistent direction, not chasing
        if is_consistent && !self.is_chasing && !is_variable {
            self.classified_strategy = PredatorStrategy::Patrol;
            self.confidence = self.direction_consistency;
            return;
        }

        // Random: High variance or nothing else matched
        self.classified_strategy = PredatorStrategy::Random;
        self.confidence = if is_variable { 0.8 } else { 0.5 };
    }

    /// Get strategy as one-hot encoded array [Random, Patrol, Chase, Ambush]
    pub fn strategy_one_hot(&self) -> [f32; 4] {
        let mut result = [0.0f32; 4];
        match self.classified_strategy {
            PredatorStrategy::Random => result[0] = 1.0,
            PredatorStrategy::Patrol => result[1] = 1.0,
            PredatorStrategy::Chase => result[2] = 1.0,
            PredatorStrategy::Ambush => result[3] = 1.0,
        }
        result
    }

    /// Calculate approach angle relative to observer (0-1, 1 = directly approaching)
    pub fn approach_angle(&self, observer_x: u8, observer_y: u8) -> f32 {
        if self.positions.len() < 2 {
            return 0.0;
        }

        let len = self.positions.len();
        let (prev_x, prev_y) = self.positions[len - 2];
        let (curr_x, curr_y) = self.positions[len - 1];

        // Movement vector
        let mov_dx = curr_x as f32 - prev_x as f32;
        let mov_dy = curr_y as f32 - prev_y as f32;
        let mov_len = (mov_dx * mov_dx + mov_dy * mov_dy).sqrt();

        if mov_len < 0.1 {
            return 0.0; // Not moving
        }

        // Vector toward observer
        let obs_dx = observer_x as f32 - curr_x as f32;
        let obs_dy = observer_y as f32 - curr_y as f32;
        let obs_len = (obs_dx * obs_dx + obs_dy * obs_dy).sqrt();

        if obs_len < 0.1 {
            return 1.0; // Very close
        }

        // Dot product gives cosine of angle
        let dot = (mov_dx * obs_dx + mov_dy * obs_dy) / (mov_len * obs_len);

        // Convert to 0-1 range (1 = directly toward, 0 = directly away)
        ((dot + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Check if observation is stale (too old)
    pub fn is_stale(&self, current_time: u64, max_age: u64) -> bool {
        current_time.saturating_sub(self.last_seen) > max_age
    }
}

/// Result of an attack attempt
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttackResult {
    /// Attack succeeded
    Hit {
        damage: f32,
        killed: bool,
        energy_gained: f32,
    },
    /// No valid target at position
    NoTarget,
    /// Target is too far away
    OutOfRange,
    /// Attack on cooldown
    OnCooldown,
}

/// Predation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredationConfig {
    /// Is predation enabled
    pub enabled: bool,
    /// Damage multiplier based on attacker size
    pub damage_multiplier: f32,
    /// Energy gained per unit of victim size
    pub size_energy_multiplier: f32,
    /// Fraction of victim's stored energy gained
    pub stored_energy_fraction: f32,
    /// Cooldown steps between attacks
    pub attack_cooldown: u32,
}

impl Default for PredationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            damage_multiplier: 15.0,  // Was 0.5 - now ~7 attacks to kill (with size=1.0)
            size_energy_multiplier: 20.0,  // Was 10.0 - more reward for kills
            stored_energy_fraction: 0.5,   // Was 0.3 - get more of victim's energy
            attack_cooldown: 2,            // Was 5 - can attack more often
        }
    }
}

/// Calculate attack damage based on size difference
pub fn calculate_damage(attacker_size: f32, target_size: f32, config: &PredationConfig) -> f32 {
    let size_advantage = attacker_size - target_size;

    if size_advantage < 0.0 {
        // Attacking larger organism - reduced damage
        attacker_size * config.damage_multiplier * 0.5
    } else {
        // Normal damage
        attacker_size * config.damage_multiplier
    }
}

/// Calculate energy gained from a kill
pub fn calculate_energy_gain(
    victim_size: f32,
    victim_energy: f32,
    config: &PredationConfig,
) -> f32 {
    // Energy from victim's size (meat)
    let size_energy = victim_size * config.size_energy_multiplier;

    // Bonus from stored energy (partial)
    let stored_energy = victim_energy.max(0.0) * config.stored_energy_fraction;

    size_energy + stored_energy
}

/// Check if attack can hit (distance check)
pub fn is_in_range(attacker_x: u8, attacker_y: u8, target_x: u8, target_y: u8) -> bool {
    let dx = (attacker_x as i16 - target_x as i16).abs();
    let dy = (attacker_y as i16 - target_y as i16).abs();

    // Must be adjacent (including diagonals)
    dx <= 1 && dy <= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_calculation() {
        let config = PredationConfig::default();

        // Same size: 2.0 * 15.0 = 30.0
        let damage = calculate_damage(2.0, 2.0, &config);
        assert!((damage - 30.0).abs() < 0.01);

        // Larger attacker: 3.0 * 15.0 = 45.0
        let damage = calculate_damage(3.0, 1.0, &config);
        assert!((damage - 45.0).abs() < 0.01);

        // Smaller attacker (reduced): 1.0 * 15.0 * 0.5 = 7.5
        let damage = calculate_damage(1.0, 3.0, &config);
        assert!((damage - 7.5).abs() < 0.01);
    }

    #[test]
    fn test_energy_gain() {
        let config = PredationConfig::default();

        let gain = calculate_energy_gain(2.0, 50.0, &config);
        // 2.0 * 20.0 + 50.0 * 0.5 = 40.0 + 25.0 = 65.0
        assert!((gain - 65.0).abs() < 0.01);
    }

    #[test]
    fn test_range_check() {
        // Adjacent
        assert!(is_in_range(5, 5, 5, 6));
        assert!(is_in_range(5, 5, 6, 5));
        assert!(is_in_range(5, 5, 6, 6)); // Diagonal

        // Same position
        assert!(is_in_range(5, 5, 5, 5));

        // Too far
        assert!(!is_in_range(5, 5, 5, 7));
        assert!(!is_in_range(5, 5, 7, 5));
    }
}
