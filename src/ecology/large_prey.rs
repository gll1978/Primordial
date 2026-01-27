//! B3: Large Prey and Multi-Agent Coordination System
//!
//! Implements cooperative hunting mechanics that require 2+ attackers.

use crate::organism::OrganismId;
use serde::{Deserialize, Serialize};

/// Large prey that requires cooperative hunting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargePrey {
    /// Unique identifier
    pub id: u64,
    /// Position X
    pub x: u8,
    /// Position Y
    pub y: u8,
    /// Current health (starts at 300, 3x normal)
    pub health: f32,
    /// Maximum health
    pub max_health: f32,
    /// Minimum number of attackers needed to damage
    pub attackers_needed: u32,
    /// Steps before the prey escapes (if not enough attackers)
    pub escape_timer: u32,
    /// Maximum escape timer
    pub max_escape_timer: u32,
    /// Energy reward for killing (split among attackers)
    pub reward: f32,
    /// Size (for display and damage calculations)
    pub size: f32,
}

impl LargePrey {
    /// Create new large prey at position
    pub fn new(id: u64, x: u8, y: u8) -> Self {
        Self {
            id,
            x,
            y,
            health: 300.0,
            max_health: 300.0,
            attackers_needed: 2,
            escape_timer: 50,
            max_escape_timer: 50,
            reward: 200.0,  // Base reward
            size: 3.0,      // 3x normal size
        }
    }

    /// Create a "boss" large prey (harder, more reward)
    pub fn new_boss(id: u64, x: u8, y: u8) -> Self {
        Self {
            id,
            x,
            y,
            health: 500.0,
            max_health: 500.0,
            attackers_needed: 3,
            escape_timer: 40,
            max_escape_timer: 40,
            reward: 400.0,
            size: 5.0,
        }
    }

    /// Check if prey is dead
    pub fn is_dead(&self) -> bool {
        self.health <= 0.0
    }

    /// Check if prey has escaped
    pub fn has_escaped(&self) -> bool {
        self.escape_timer == 0
    }

    /// Apply damage from multiple attackers
    /// Returns actual damage dealt (0 if not enough attackers)
    pub fn take_damage(&mut self, damage_per_attacker: f32, num_attackers: u32) -> f32 {
        if num_attackers < self.attackers_needed {
            // Not enough attackers - prey resists and timer counts down
            self.escape_timer = self.escape_timer.saturating_sub(1);
            return 0.0;
        }

        // Reset escape timer when enough attackers
        self.escape_timer = self.max_escape_timer;

        // Apply damage from all attackers
        let total_damage = damage_per_attacker * num_attackers as f32;
        self.health -= total_damage;

        total_damage
    }

    /// Calculate reward per attacker when killed
    pub fn reward_per_attacker(&self, num_attackers: u32) -> f32 {
        if num_attackers == 0 {
            return 0.0;
        }
        // Bonus for exactly meeting the requirement (no freeloaders)
        let efficiency_bonus = if num_attackers == self.attackers_needed { 1.2 } else { 1.0 };
        (self.reward * efficiency_bonus) / num_attackers as f32
    }

    /// Move prey (simple random walk or flee behavior)
    pub fn wander(&mut self, grid_size: u8, rng: &mut impl rand::Rng) {
        let dx: i8 = rng.gen_range(-1..=1);
        let dy: i8 = rng.gen_range(-1..=1);

        let new_x = (self.x as i16 + dx as i16).clamp(0, grid_size as i16 - 1) as u8;
        let new_y = (self.y as i16 + dy as i16).clamp(0, grid_size as i16 - 1) as u8;

        self.x = new_x;
        self.y = new_y;
    }
}

/// Configuration for large prey system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargePreyConfig {
    /// Is the system enabled
    pub enabled: bool,
    /// Maximum number of large prey in the world
    pub max_large_prey: usize,
    /// Spawn chance per step (when below max)
    pub spawn_chance: f32,
    /// Minimum population for large prey to spawn
    pub min_population: usize,
}

impl Default for LargePreyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_large_prey: 3,
            spawn_chance: 0.001, // ~1 spawn per 1000 steps
            min_population: 50,
        }
    }
}

/// Cooperation signal types between organisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CooperationSignal {
    #[default]
    None,
    /// Propose cooperation for hunting
    ProposeCooperation,
    /// Accept cooperation proposal
    AcceptCooperation,
    /// Reject cooperation proposal
    RejectCooperation,
}

/// Trust relationship between two organisms
#[derive(Debug, Clone)]
pub struct TrustRelationship {
    /// Partner organism ID
    pub partner_id: OrganismId,
    /// Trust level (-1.0 to 1.0)
    pub trust_level: f32,
    /// Number of successful cooperations
    pub successes: u32,
    /// Number of failed cooperations (partner didn't help)
    pub failures: u32,
    /// Last interaction time
    pub last_interaction: u64,
}

impl TrustRelationship {
    /// Create new neutral trust relationship
    pub fn new(partner_id: OrganismId, time: u64) -> Self {
        Self {
            partner_id,
            trust_level: 0.0,
            successes: 0,
            failures: 0,
            last_interaction: time,
        }
    }

    /// Update trust after successful cooperation
    pub fn record_success(&mut self, time: u64) {
        self.successes += 1;
        self.trust_level = (self.trust_level + 0.2).min(1.0);
        self.last_interaction = time;
    }

    /// Update trust after failed cooperation (partner didn't help)
    pub fn record_failure(&mut self, time: u64) {
        self.failures += 1;
        self.trust_level = (self.trust_level - 0.3).max(-1.0);
        self.last_interaction = time;
    }

    /// Decay trust over time
    pub fn decay(&mut self, current_time: u64) {
        let elapsed = current_time.saturating_sub(self.last_interaction);
        if elapsed > 100 {
            // Slowly decay toward neutral
            self.trust_level *= 0.99;
        }
    }

    /// Check if relationship is stale
    pub fn is_stale(&self, current_time: u64, max_age: u64) -> bool {
        current_time.saturating_sub(self.last_interaction) > max_age
    }
}

/// Manages cooperation proposals between organisms
#[derive(Debug, Default)]
pub struct CooperationManager {
    /// Active proposals: (proposer_id, target_id, time)
    pub proposals: Vec<(OrganismId, OrganismId, u64)>,
    /// Active cooperation pairs: (org1_id, org2_id, target_prey_id)
    pub active_cooperations: Vec<(OrganismId, OrganismId, u64)>,
}

impl CooperationManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a cooperation proposal
    pub fn propose(&mut self, proposer_id: OrganismId, target_id: OrganismId, time: u64) {
        // Remove any existing proposal from this proposer
        self.proposals.retain(|(p, _, _)| *p != proposer_id);
        self.proposals.push((proposer_id, target_id, time));
    }

    /// Accept a proposal (returns proposer_id if successful)
    pub fn accept(&mut self, accepter_id: OrganismId, prey_id: u64, _time: u64) -> Option<OrganismId> {
        // Find a proposal targeting this accepter
        if let Some(idx) = self.proposals.iter().position(|(_, t, _)| *t == accepter_id) {
            let (proposer_id, _, _) = self.proposals.remove(idx);
            self.active_cooperations.push((proposer_id, accepter_id, prey_id));
            return Some(proposer_id);
        }
        None
    }

    /// Clean up old proposals (expire after 20 steps)
    pub fn cleanup(&mut self, current_time: u64) {
        self.proposals.retain(|(_, _, t)| current_time.saturating_sub(*t) < 20);
        // Active cooperations clean up when prey dies or escapes
    }

    /// Check if organism is in an active cooperation
    pub fn is_cooperating(&self, org_id: OrganismId) -> bool {
        self.active_cooperations.iter().any(|(a, b, _)| *a == org_id || *b == org_id)
    }

    /// Get cooperation partner if any
    pub fn get_partner(&self, org_id: OrganismId) -> Option<OrganismId> {
        for (a, b, _) in &self.active_cooperations {
            if *a == org_id {
                return Some(*b);
            }
            if *b == org_id {
                return Some(*a);
            }
        }
        None
    }

    /// Remove cooperation involving this prey
    pub fn remove_for_prey(&mut self, prey_id: u64) {
        self.active_cooperations.retain(|(_, _, p)| *p != prey_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_prey_creation() {
        let prey = LargePrey::new(1, 10, 10);
        assert_eq!(prey.health, 300.0);
        assert_eq!(prey.attackers_needed, 2);
        assert!(!prey.is_dead());
    }

    #[test]
    fn test_large_prey_damage() {
        let mut prey = LargePrey::new(1, 10, 10);

        // Not enough attackers - no damage
        let damage = prey.take_damage(10.0, 1);
        assert_eq!(damage, 0.0);
        assert_eq!(prey.health, 300.0);
        assert!(prey.escape_timer < prey.max_escape_timer);

        // Enough attackers - damage applied
        prey.escape_timer = prey.max_escape_timer;
        let damage = prey.take_damage(10.0, 2);
        assert_eq!(damage, 20.0);
        assert_eq!(prey.health, 280.0);
    }

    #[test]
    fn test_trust_relationship() {
        let mut trust = TrustRelationship::new(1, 0);
        assert_eq!(trust.trust_level, 0.0);

        trust.record_success(10);
        assert!(trust.trust_level > 0.0);

        trust.record_failure(20);
        assert!(trust.trust_level < 0.2);
    }

    #[test]
    fn test_cooperation_manager() {
        let mut manager = CooperationManager::new();

        manager.propose(1, 2, 0);
        assert_eq!(manager.proposals.len(), 1);

        let proposer = manager.accept(2, 100, 1);
        assert_eq!(proposer, Some(1));
        assert_eq!(manager.active_cooperations.len(), 1);
        assert!(manager.is_cooperating(1));
        assert!(manager.is_cooperating(2));
    }
}
