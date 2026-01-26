//! Predation system - attack and kill mechanics.

use serde::{Deserialize, Serialize};

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
