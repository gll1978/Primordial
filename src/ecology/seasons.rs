//! Seasonal variation system.

use serde::{Deserialize, Serialize};

/// The four seasons
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}

impl Season {
    /// Get food regeneration multiplier for this season
    pub fn food_multiplier(&self) -> f32 {
        match self {
            Season::Spring => 1.5,
            Season::Summer => 1.0,
            Season::Autumn => 0.7,
            Season::Winter => 0.3,
        }
    }

    /// Get fruit availability multiplier (fruits peak in autumn)
    pub fn fruit_multiplier(&self) -> f32 {
        match self {
            Season::Spring => 0.5,
            Season::Summer => 1.0,
            Season::Autumn => 2.0,
            Season::Winter => 0.1,
        }
    }

    /// Get insect availability multiplier
    pub fn insect_multiplier(&self) -> f32 {
        match self {
            Season::Spring => 1.5,
            Season::Summer => 2.0,
            Season::Autumn => 0.5,
            Season::Winter => 0.1,
        }
    }

    /// Get energy per food multiplier (Phase 2 Feature 4)
    /// Winter food is rare but more valuable (animals store more fat)
    /// Based on Notion specs: Spring: 30, Summer: 25, Autumn: 28, Winter: 35
    /// Normalized to Summer = 1.0
    pub fn energy_multiplier(&self) -> f32 {
        match self {
            Season::Spring => 1.2,  // 30/25 = 1.2
            Season::Summer => 1.0,  // baseline
            Season::Autumn => 1.12, // 28/25 = 1.12
            Season::Winter => 1.4,  // 35/25 = 1.4 (rare but valuable)
        }
    }

    /// Get current season based on world time
    pub fn from_time(time: u64, season_length: u64) -> Season {
        if season_length == 0 {
            return Season::Summer; // Default to summer if no seasons
        }
        let season_index = (time / season_length) % 4;
        match season_index {
            0 => Season::Spring,
            1 => Season::Summer,
            2 => Season::Autumn,
            _ => Season::Winter,
        }
    }

    /// Days until next season
    pub fn days_until_next(time: u64, season_length: u64) -> u64 {
        if season_length == 0 {
            return 0;
        }
        season_length - (time % season_length)
    }

    /// Get next season
    pub fn next(&self) -> Season {
        match self {
            Season::Spring => Season::Summer,
            Season::Summer => Season::Autumn,
            Season::Autumn => Season::Winter,
            Season::Winter => Season::Spring,
        }
    }

    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Season::Spring => "Spring",
            Season::Summer => "Summer",
            Season::Autumn => "Autumn",
            Season::Winter => "Winter",
        }
    }
}

/// Seasonal system manager
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeasonalSystem {
    pub current_season: Season,
    pub season_length: u64,
    pub enabled: bool,
}

impl SeasonalSystem {
    pub fn new(config: &SeasonsConfig) -> Self {
        Self {
            current_season: Season::Spring,
            season_length: config.season_length,
            enabled: config.enabled,
        }
    }

    /// Update season based on current time
    pub fn update(&mut self, time: u64) {
        if self.enabled && self.season_length > 0 {
            self.current_season = Season::from_time(time, self.season_length);
        }
    }

    /// Get current food multiplier
    pub fn food_multiplier(&self) -> f32 {
        if self.enabled {
            self.current_season.food_multiplier()
        } else {
            1.0
        }
    }

    /// Get fruit multiplier
    pub fn fruit_multiplier(&self) -> f32 {
        if self.enabled {
            self.current_season.fruit_multiplier()
        } else {
            1.0
        }
    }

    /// Get insect multiplier
    pub fn insect_multiplier(&self) -> f32 {
        if self.enabled {
            self.current_season.insect_multiplier()
        } else {
            1.0
        }
    }

    /// Get energy per food multiplier (Phase 2 Feature 4)
    /// Food gives more energy in winter (rare but valuable)
    pub fn energy_multiplier(&self) -> f32 {
        if self.enabled {
            self.current_season.energy_multiplier()
        } else {
            1.0
        }
    }
}

/// Seasons configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonsConfig {
    /// Is seasonal variation enabled
    pub enabled: bool,
    /// Steps per season (1000 = 4000 steps per year)
    pub season_length: u64,
}

impl Default for SeasonsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            season_length: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_season_cycle() {
        assert_eq!(Season::from_time(0, 1000), Season::Spring);
        assert_eq!(Season::from_time(500, 1000), Season::Spring);
        assert_eq!(Season::from_time(1000, 1000), Season::Summer);
        assert_eq!(Season::from_time(2000, 1000), Season::Autumn);
        assert_eq!(Season::from_time(3000, 1000), Season::Winter);
        assert_eq!(Season::from_time(4000, 1000), Season::Spring); // Cycle
    }

    #[test]
    fn test_food_multipliers() {
        assert_eq!(Season::Spring.food_multiplier(), 1.5);
        assert_eq!(Season::Summer.food_multiplier(), 1.0);
        assert_eq!(Season::Autumn.food_multiplier(), 0.7);
        assert_eq!(Season::Winter.food_multiplier(), 0.3);
    }

    #[test]
    fn test_days_until_next() {
        assert_eq!(Season::days_until_next(0, 1000), 1000);
        assert_eq!(Season::days_until_next(500, 1000), 500);
        assert_eq!(Season::days_until_next(999, 1000), 1);
    }

    #[test]
    fn test_seasonal_system() {
        let config = SeasonsConfig::default();
        let mut system = SeasonalSystem::new(&config);

        assert_eq!(system.current_season, Season::Spring);
        assert!((system.food_multiplier() - 1.5).abs() < 0.01);

        system.update(1000);
        assert_eq!(system.current_season, Season::Summer);
        assert!((system.food_multiplier() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_disabled_seasons() {
        let config = SeasonsConfig {
            enabled: false,
            season_length: 1000,
        };
        let system = SeasonalSystem::new(&config);

        assert!((system.food_multiplier() - 1.0).abs() < 0.01); // Always 1.0 when disabled
    }

    #[test]
    fn test_energy_multipliers() {
        // Phase 2 Feature 4: Seasonal energy per food
        // Winter food is rare but more valuable
        assert!((Season::Spring.energy_multiplier() - 1.2).abs() < 0.01);
        assert!((Season::Summer.energy_multiplier() - 1.0).abs() < 0.01);
        assert!((Season::Autumn.energy_multiplier() - 1.12).abs() < 0.01);
        assert!((Season::Winter.energy_multiplier() - 1.4).abs() < 0.01);
    }

    #[test]
    fn test_seasonal_system_energy() {
        let config = SeasonsConfig::default();
        let mut system = SeasonalSystem::new(&config);

        // Spring
        assert!((system.energy_multiplier() - 1.2).abs() < 0.01);

        // Summer
        system.update(1000);
        assert!((system.energy_multiplier() - 1.0).abs() < 0.01);

        // Autumn
        system.update(2000);
        assert!((system.energy_multiplier() - 1.12).abs() < 0.01);

        // Winter
        system.update(3000);
        assert!((system.energy_multiplier() - 1.4).abs() < 0.01);
    }
}
