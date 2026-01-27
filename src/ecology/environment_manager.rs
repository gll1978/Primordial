//! Manages environment transitions for procedural environments.
//!
//! Triggers patch reshuffling based on generation count or step count,
//! forcing organisms to generalize rather than memorize specific layouts.

use serde::{Deserialize, Serialize};

/// Environment variation settings
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub enabled: bool,
    pub reshuffle_interval: u64,
    #[serde(default)]
    pub reshuffle_interval_steps: Option<u64>,
    pub base_seed: u64,
    pub variation_level: f32,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reshuffle_interval: 50,
            reshuffle_interval_steps: None,
            base_seed: 42,
            variation_level: 0.5,
        }
    }
}

/// Manages environment transitions
pub struct EnvironmentManager {
    config: EnvironmentConfig,
    current_seed: u64,
    last_reshuffle_gen: u64,
    last_reshuffle_step: u64,
    reshuffle_count: u32,
}

impl EnvironmentManager {
    pub fn new(config: EnvironmentConfig) -> Self {
        Self {
            current_seed: config.base_seed,
            config,
            last_reshuffle_gen: 0,
            last_reshuffle_step: 0,
            reshuffle_count: 0,
        }
    }

    /// Check if should reshuffle based on generation
    pub fn should_reshuffle_gen(&self, current_generation: u64) -> bool {
        if !self.config.enabled {
            return false;
        }
        current_generation.saturating_sub(self.last_reshuffle_gen) >= self.config.reshuffle_interval
    }

    /// Check if should reshuffle based on steps
    pub fn should_reshuffle_step(&self, current_step: u64) -> bool {
        if !self.config.enabled {
            return false;
        }
        if let Some(interval) = self.config.reshuffle_interval_steps {
            current_step.saturating_sub(self.last_reshuffle_step) >= interval
        } else {
            false
        }
    }

    /// Get next seed and update state
    pub fn next_seed(&mut self, current_gen: u64, current_step: u64) -> u64 {
        self.reshuffle_count += 1;
        self.last_reshuffle_gen = current_gen;
        self.last_reshuffle_step = current_step;

        self.current_seed = self.config.base_seed
            .wrapping_add(self.reshuffle_count as u64)
            .wrapping_mul(6364136223846793005);

        log::info!(
            "Environment reshuffle #{} at gen {} step {} (seed: {})",
            self.reshuffle_count,
            current_gen,
            current_step,
            self.current_seed
        );

        self.current_seed
    }

    pub fn current_seed(&self) -> u64 {
        self.current_seed
    }

    pub fn reshuffle_count(&self) -> u32 {
        self.reshuffle_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshuffle_timing() {
        let config = EnvironmentConfig {
            enabled: true,
            reshuffle_interval: 10,
            ..Default::default()
        };
        let mgr = EnvironmentManager::new(config);
        assert!(!mgr.should_reshuffle_gen(5));
        assert!(mgr.should_reshuffle_gen(10));
        assert!(mgr.should_reshuffle_gen(15));
    }

    #[test]
    fn test_next_seed_deterministic() {
        let config = EnvironmentConfig {
            enabled: true,
            base_seed: 42,
            ..Default::default()
        };
        let mut mgr = EnvironmentManager::new(config);
        let s1 = mgr.next_seed(50, 1000);
        // Same base_seed + same count should give same result
        let config2 = EnvironmentConfig {
            enabled: true,
            base_seed: 42,
            ..Default::default()
        };
        let mut mgr2 = EnvironmentManager::new(config2);
        let s2 = mgr2.next_seed(50, 1000);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_disabled() {
        let config = EnvironmentConfig::default(); // enabled: false
        let mgr = EnvironmentManager::new(config);
        assert!(!mgr.should_reshuffle_gen(1000));
    }
}
