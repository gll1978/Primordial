//! Sexual reproduction system - sex determination, mating, and inbreeding.

use crate::config::DiversityConfig;
use crate::genetics::phylogeny::PhylogeneticTree;
use serde::{Deserialize, Serialize};

/// Biological sex for sexual reproduction
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    Male,
    Female,
}

impl Sex {
    /// Generate random sex (50/50)
    pub fn random() -> Self {
        if rand::random() {
            Sex::Male
        } else {
            Sex::Female
        }
    }

    /// Get display character
    pub fn char(&self) -> char {
        match self {
            Sex::Male => '♂',
            Sex::Female => '♀',
        }
    }
}

impl Default for Sex {
    fn default() -> Self {
        Sex::random()
    }
}

/// Sexual reproduction system tracking
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SexualReproductionSystem {
    /// Total successful matings
    pub total_matings: u64,
    /// Failed mating attempts
    pub failed_matings: u64,
    /// Matings between related organisms
    pub inbreeding_events: u64,
    /// Total offspring produced
    pub total_offspring: u64,
    /// Mating attempts blocked by speciation
    pub speciation_blocks: u64,
    /// Mating attempts in boundary zone (partial probability)
    pub boundary_zone_attempts: u64,
}

impl SexualReproductionSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if two organisms can mate
    pub fn can_mate(
        sex1: Sex,
        sex2: Sex,
        energy1: f32,
        energy2: f32,
        cooldown1: u32,
        cooldown2: u32,
        distance: u8,
        config: &SexualReproductionConfig,
    ) -> bool {
        // Must be opposite sex
        if sex1 == sex2 {
            return false;
        }

        // Must be adjacent (distance <= 1)
        if distance > 1 {
            return false;
        }

        // Check cooldown
        if cooldown1 > 0 || cooldown2 > 0 {
            return false;
        }

        // Both must have enough energy
        if energy1 < config.min_energy || energy2 < config.min_energy {
            return false;
        }

        true
    }

    /// Detect inbreeding (same lineage)
    #[inline]
    pub fn is_inbred(lineage1: u32, lineage2: u32) -> bool {
        lineage1 == lineage2
    }

    /// Calculate inbreeding fitness penalty
    #[inline]
    pub fn inbreeding_penalty(config: &SexualReproductionConfig) -> f32 {
        config.inbreeding_fitness_cost
    }

    /// Record a successful mating
    pub fn record_mating(&mut self, was_inbred: bool) {
        self.total_matings += 1;
        if was_inbred {
            self.inbreeding_events += 1;
        }
    }

    /// Record a failed mating attempt
    pub fn record_failed_mating(&mut self) {
        self.failed_matings += 1;
    }

    /// Record offspring birth
    pub fn record_offspring(&mut self) {
        self.total_offspring += 1;
    }

    /// Get mating success rate
    pub fn success_rate(&self) -> f32 {
        let total = self.total_matings + self.failed_matings;
        if total == 0 {
            0.0
        } else {
            self.total_matings as f32 / total as f32
        }
    }

    /// Get inbreeding rate
    pub fn inbreeding_rate(&self) -> f32 {
        if self.total_matings == 0 {
            0.0
        } else {
            self.inbreeding_events as f32 / self.total_matings as f32
        }
    }

    /// Record a speciation block (incompatible species)
    pub fn record_speciation_block(&mut self) {
        self.speciation_blocks += 1;
    }

    /// Record a boundary zone mating attempt
    pub fn record_boundary_zone_attempt(&mut self) {
        self.boundary_zone_attempts += 1;
    }

    /// Get speciation block rate
    pub fn speciation_block_rate(&self) -> f32 {
        let total = self.total_matings + self.failed_matings + self.speciation_blocks;
        if total == 0 {
            0.0
        } else {
            self.speciation_blocks as f32 / total as f32
        }
    }
}

/// Check speciation compatibility between two organisms based on genetic distance.
/// Returns (can_mate, probability) where:
/// - can_mate: false if organisms are completely incompatible species
/// - probability: 1.0 for same species, interpolated for boundary zone, 0.0 for different species
pub fn check_speciation_compatibility(
    org1_id: u64,
    org2_id: u64,
    phylogeny: &PhylogeneticTree,
    config: &DiversityConfig,
) -> (bool, f32) {
    if !config.speciation_enabled {
        return (true, 1.0);
    }

    let distance = phylogeny
        .genetic_distance(org1_id, org2_id)
        .unwrap_or(config.max_genetic_distance + 1);

    if distance <= config.min_genetic_distance {
        // Same species - always compatible
        (true, 1.0)
    } else if distance >= config.max_genetic_distance {
        // Different species - never compatible
        (false, 0.0)
    } else {
        // Boundary zone - interpolated probability
        let range = (config.max_genetic_distance - config.min_genetic_distance) as f32;
        let pos = (distance - config.min_genetic_distance) as f32;
        let prob = 1.0 - (pos / range) * (1.0 - config.boundary_mating_probability);
        (true, prob)
    }
}

/// Configuration for sexual reproduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SexualReproductionConfig {
    /// Is sexual reproduction enabled (vs asexual)
    pub enabled: bool,
    /// Minimum energy to mate
    pub min_energy: f32,
    /// Energy cost per parent
    pub energy_cost: f32,
    /// Initial energy for offspring
    pub offspring_energy: f32,
    /// Steps before can mate again
    pub cooldown: u32,
    /// Fitness penalty for inbreeding (0.0-1.0)
    pub inbreeding_fitness_cost: f32,
    /// Maximum distance for mating
    pub max_mating_distance: u8,
}

impl Default for SexualReproductionConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Default to asexual for simpler dynamics
            min_energy: 50.0, // Was 80.0 - too high, same as initial energy!
            energy_cost: 25.0, // Was 40.0 - reduced to match asexual
            offspring_energy: 50.0, // Was 60.0
            cooldown: 30, // Was 50
            inbreeding_fitness_cost: 0.3,
            max_mating_distance: 2, // Was 1 - increase chance of finding mate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SexualReproductionConfig {
        SexualReproductionConfig::default()
    }

    #[test]
    fn test_sex_random() {
        let mut males = 0;
        let mut females = 0;

        for _ in 0..1000 {
            match Sex::random() {
                Sex::Male => males += 1,
                Sex::Female => females += 1,
            }
        }

        // Should be roughly 50/50 (within 10%)
        assert!(males > 400 && males < 600);
        assert!(females > 400 && females < 600);
    }

    #[test]
    fn test_can_mate_opposite_sex() {
        let config = test_config();

        // Opposite sex should be able to mate
        assert!(SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            100.0,
            100.0,
            0,
            0,
            1,
            &config
        ));

        // Same sex should not mate
        assert!(!SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Male,
            100.0,
            100.0,
            0,
            0,
            1,
            &config
        ));

        assert!(!SexualReproductionSystem::can_mate(
            Sex::Female,
            Sex::Female,
            100.0,
            100.0,
            0,
            0,
            1,
            &config
        ));
    }

    #[test]
    fn test_can_mate_distance() {
        let config = test_config();

        // Adjacent should work
        assert!(SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            100.0,
            100.0,
            0,
            0,
            1,
            &config
        ));

        // Same cell should work
        assert!(SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            100.0,
            100.0,
            0,
            0,
            0,
            &config
        ));

        // Too far should fail
        assert!(!SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            100.0,
            100.0,
            0,
            0,
            2,
            &config
        ));
    }

    #[test]
    fn test_can_mate_energy() {
        let config = test_config();

        // Enough energy
        assert!(SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            100.0,
            100.0,
            0,
            0,
            1,
            &config
        ));

        // One too low (below min_energy of 50.0)
        assert!(!SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            30.0, // Below min_energy threshold
            100.0,
            0,
            0,
            1,
            &config
        ));

        // Both too low (below min_energy of 50.0)
        assert!(!SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            30.0,
            30.0,
            0,
            0,
            1,
            &config
        ));
    }

    #[test]
    fn test_can_mate_cooldown() {
        let config = test_config();

        // No cooldown
        assert!(SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            100.0,
            100.0,
            0,
            0,
            1,
            &config
        ));

        // One on cooldown
        assert!(!SexualReproductionSystem::can_mate(
            Sex::Male,
            Sex::Female,
            100.0,
            100.0,
            10,
            0,
            1,
            &config
        ));
    }

    #[test]
    fn test_inbreeding_detection() {
        assert!(SexualReproductionSystem::is_inbred(42, 42));
        assert!(!SexualReproductionSystem::is_inbred(42, 43));
    }

    #[test]
    fn test_mating_stats() {
        let mut system = SexualReproductionSystem::new();

        system.record_mating(false);
        system.record_mating(false);
        system.record_mating(true); // inbred
        system.record_failed_mating();

        assert_eq!(system.total_matings, 3);
        assert_eq!(system.failed_matings, 1);
        assert_eq!(system.inbreeding_events, 1);
        assert!((system.success_rate() - 0.75).abs() < 0.01);
        assert!((system.inbreeding_rate() - 0.333).abs() < 0.01);
    }
}
