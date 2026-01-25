//! Resource depletion system - tracks over-exploitation of areas.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Depletion state for a cell
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DepletionState {
    /// Normal state
    Healthy,
    /// Being depleted (food < threshold)
    Depleting,
    /// Fully depleted (no regeneration)
    Depleted,
    /// Recovering (slowly regenerating)
    Recovering,
}

/// Resource depletion tracking system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DepletionSystem {
    /// Grid of depletion states
    pub states: Vec<Vec<DepletionState>>,
    /// Set of depleted cell coordinates
    pub depleted_cells: HashSet<(u8, u8)>,
    /// Grid size
    pub grid_size: usize,
    /// Total depleted cells count
    pub total_depleted: usize,
    /// Steps since depletion started (for recovery timing)
    pub depletion_timers: Vec<Vec<u32>>,
}

impl DepletionSystem {
    /// Create a new depletion system
    pub fn new(grid_size: usize) -> Self {
        Self {
            states: vec![vec![DepletionState::Healthy; grid_size]; grid_size],
            depleted_cells: HashSet::new(),
            grid_size,
            total_depleted: 0,
            depletion_timers: vec![vec![0; grid_size]; grid_size],
        }
    }

    /// Update depletion state for a cell based on food level and organism count
    pub fn update_cell(
        &mut self,
        x: u8,
        y: u8,
        food_amount: f32,
        organism_count: usize,
        config: &DepletionConfig,
    ) {
        if !config.enabled {
            return;
        }

        let x_idx = x as usize;
        let y_idx = y as usize;

        if x_idx >= self.grid_size || y_idx >= self.grid_size {
            return;
        }

        let current_state = self.states[y_idx][x_idx];

        // State machine for depletion
        let new_state = match current_state {
            DepletionState::Healthy => {
                // Check if over-exploited
                if organism_count >= config.organism_threshold
                    && food_amount < config.food_threshold
                {
                    DepletionState::Depleting
                } else {
                    DepletionState::Healthy
                }
            }

            DepletionState::Depleting => {
                self.depletion_timers[y_idx][x_idx] += 1;

                if self.depletion_timers[y_idx][x_idx] >= config.depletion_time {
                    // Fully depleted
                    self.depleted_cells.insert((x, y));
                    self.total_depleted += 1;
                    DepletionState::Depleted
                } else if organism_count < config.organism_threshold / 2 {
                    // Pressure released - back to healthy
                    self.depletion_timers[y_idx][x_idx] = 0;
                    DepletionState::Healthy
                } else {
                    DepletionState::Depleting
                }
            }

            DepletionState::Depleted => {
                self.depletion_timers[y_idx][x_idx] += 1;

                // Check if can start recovering
                if organism_count == 0
                    && self.depletion_timers[y_idx][x_idx] >= config.recovery_delay
                {
                    DepletionState::Recovering
                } else {
                    DepletionState::Depleted
                }
            }

            DepletionState::Recovering => {
                self.depletion_timers[y_idx][x_idx] += 1;

                // Full recovery takes time
                if self.depletion_timers[y_idx][x_idx] >= config.recovery_delay + config.recovery_time
                {
                    self.depleted_cells.remove(&(x, y));
                    self.total_depleted = self.total_depleted.saturating_sub(1);
                    self.depletion_timers[y_idx][x_idx] = 0;
                    DepletionState::Healthy
                } else if organism_count > 0 {
                    // Organisms returned too soon - back to depleted
                    DepletionState::Depleted
                } else {
                    DepletionState::Recovering
                }
            }
        };

        self.states[y_idx][x_idx] = new_state;
    }

    /// Get depletion state at position
    #[inline]
    pub fn get_state(&self, x: u8, y: u8) -> DepletionState {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.states[y][x]
        } else {
            DepletionState::Depleted // Out of bounds = depleted
        }
    }

    /// Check if cell is depleted (no food regeneration)
    #[inline]
    pub fn is_depleted(&self, x: u8, y: u8) -> bool {
        matches!(
            self.get_state(x, y),
            DepletionState::Depleted | DepletionState::Depleting
        )
    }

    /// Get food regeneration multiplier (0.0 for depleted, 1.0 for healthy)
    pub fn regen_multiplier(&self, x: u8, y: u8) -> f32 {
        match self.get_state(x, y) {
            DepletionState::Healthy => 1.0,
            DepletionState::Depleting => 0.5,
            DepletionState::Depleted => 0.0,
            DepletionState::Recovering => 0.3,
        }
    }

    /// Calculate ecological pressure (fraction of depleted cells)
    pub fn ecological_pressure(&self) -> f32 {
        self.total_depleted as f32 / (self.grid_size * self.grid_size) as f32
    }

    /// Get count of cells in each state
    pub fn state_counts(&self) -> std::collections::HashMap<DepletionState, usize> {
        let mut counts = std::collections::HashMap::new();
        for row in &self.states {
            for &state in row {
                *counts.entry(state).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Reset all depletion (useful for testing)
    pub fn reset(&mut self) {
        for row in &mut self.states {
            for state in row {
                *state = DepletionState::Healthy;
            }
        }
        for row in &mut self.depletion_timers {
            for timer in row {
                *timer = 0;
            }
        }
        self.depleted_cells.clear();
        self.total_depleted = 0;
    }
}

/// Depletion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepletionConfig {
    /// Is depletion tracking enabled
    pub enabled: bool,
    /// Minimum organisms in cell to trigger depletion
    pub organism_threshold: usize,
    /// Food level below which depletion starts
    pub food_threshold: f32,
    /// Steps of over-exploitation before full depletion
    pub depletion_time: u32,
    /// Steps before recovery can begin
    pub recovery_delay: u32,
    /// Steps for full recovery
    pub recovery_time: u32,
}

impl Default for DepletionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            organism_threshold: 5,
            food_threshold: 5.0,
            depletion_time: 100,
            recovery_delay: 200,
            recovery_time: 300,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DepletionConfig {
        DepletionConfig {
            enabled: true,
            organism_threshold: 3,
            food_threshold: 5.0,
            depletion_time: 10,
            recovery_delay: 20,
            recovery_time: 30,
        }
    }

    #[test]
    fn test_depletion_system_creation() {
        let system = DepletionSystem::new(50);
        assert_eq!(system.grid_size, 50);
        assert_eq!(system.total_depleted, 0);
        assert_eq!(system.get_state(25, 25), DepletionState::Healthy);
    }

    #[test]
    fn test_depletion_progression() {
        let mut system = DepletionSystem::new(50);
        let config = test_config();

        // Initial state
        assert_eq!(system.get_state(10, 10), DepletionState::Healthy);

        // Over-exploit: high organisms, low food
        for _ in 0..5 {
            system.update_cell(10, 10, 2.0, 5, &config);
        }
        assert_eq!(system.get_state(10, 10), DepletionState::Depleting);

        // Continue until depleted
        for _ in 0..10 {
            system.update_cell(10, 10, 1.0, 5, &config);
        }
        assert_eq!(system.get_state(10, 10), DepletionState::Depleted);
        assert!(system.is_depleted(10, 10));
    }

    #[test]
    fn test_recovery() {
        let mut system = DepletionSystem::new(50);
        let mut config = test_config();
        config.depletion_time = 5;
        config.recovery_delay = 20;  // Increased to ensure Depleted state persists
        config.recovery_time = 20;

        // Deplete (5 iterations in Depleting, 5 in Depleted)
        for _ in 0..10 {
            system.update_cell(10, 10, 1.0, 5, &config);
        }
        assert_eq!(system.get_state(10, 10), DepletionState::Depleted);

        // Wait until recovery starts (timer at 10, need to reach 20 for recovery_delay)
        for _ in 0..15 {
            system.update_cell(10, 10, 1.0, 0, &config);
        }
        // Timer is now 25, which is >= recovery_delay (20), so Recovering
        assert_eq!(system.get_state(10, 10), DepletionState::Recovering);

        // Complete recovery (need to reach recovery_delay + recovery_time = 40)
        for _ in 0..20 {
            system.update_cell(10, 10, 1.0, 0, &config);
        }
        // Timer is now 45, which is >= 40, so Healthy
        assert_eq!(system.get_state(10, 10), DepletionState::Healthy);
    }

    #[test]
    fn test_regen_multiplier() {
        let mut system = DepletionSystem::new(50);

        assert_eq!(system.regen_multiplier(10, 10), 1.0);

        system.states[10][10] = DepletionState::Depleting;
        assert_eq!(system.regen_multiplier(10, 10), 0.5);

        system.states[10][10] = DepletionState::Depleted;
        assert_eq!(system.regen_multiplier(10, 10), 0.0);

        system.states[10][10] = DepletionState::Recovering;
        assert_eq!(system.regen_multiplier(10, 10), 0.3);
    }

    #[test]
    fn test_ecological_pressure() {
        let mut system = DepletionSystem::new(10); // 100 cells

        assert_eq!(system.ecological_pressure(), 0.0);

        // Deplete 10 cells
        system.total_depleted = 10;
        assert!((system.ecological_pressure() - 0.1).abs() < 0.01);
    }
}
