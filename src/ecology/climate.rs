//! Climate system for realistic temperature and humidity simulation.
//!
//! This module provides environmental temperature and humidity grids that vary
//! based on terrain, season, and position (latitude simulation).

use serde::{Deserialize, Serialize};
use super::seasons::Season;
use super::terrain::{Terrain, TerrainGrid};

/// Configuration for the climate system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateConfig {
    /// Enable climate system
    pub enabled: bool,
    /// Base temperature (0.0 = cold, 1.0 = hot), 0.5 = temperate
    pub base_temperature: f32,
    /// How much temperature varies with seasons (0.0-0.5)
    pub seasonal_variation: f32,
    /// Temperature reduction for mountains (0.0-0.5)
    pub altitude_cooling: f32,
    /// Humidity bonus near water (0.0-0.5)
    pub water_humidity_bonus: f32,
    /// Humidity penalty for deserts (0.0-0.5)
    pub desert_humidity_penalty: f32,
    /// Latitude effect strength (north = colder, south = warmer)
    pub latitude_effect: f32,
    /// Steps between climate updates (for performance)
    pub update_interval: u64,
}

impl Default for ClimateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_temperature: 0.5,       // Temperate baseline
            seasonal_variation: 0.4,     // Strong seasonal effect
            altitude_cooling: 0.3,       // Mountains are cooler
            water_humidity_bonus: 0.4,   // Near water is humid
            desert_humidity_penalty: 0.3, // Deserts are dry
            latitude_effect: 0.3,        // North/south gradient
            update_interval: 100,        // Update every 100 steps
        }
    }
}

/// Climate system managing temperature and humidity grids
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClimateSystem {
    /// Temperature grid: -1.0 (very cold) to 1.0 (very hot)
    pub temperature_grid: Vec<Vec<f32>>,
    /// Humidity grid: 0.0 (very dry) to 1.0 (very humid)
    pub humidity_grid: Vec<Vec<f32>>,
    /// Configuration
    config: ClimateConfig,
    /// Grid size
    grid_size: usize,
    /// Last update time
    last_update: u64,
}

impl ClimateSystem {
    /// Create a new climate system
    pub fn new(grid_size: usize, terrain_grid: &TerrainGrid, config: &ClimateConfig) -> Self {
        let mut system = Self {
            temperature_grid: vec![vec![0.5; grid_size]; grid_size],
            humidity_grid: vec![vec![0.5; grid_size]; grid_size],
            config: config.clone(),
            grid_size,
            last_update: 0,
        };

        // Initialize grids based on terrain (Season::Summer as baseline)
        system.recalculate(terrain_grid, Season::Summer);

        system
    }

    /// Check if climate system is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Update climate based on current season (called periodically)
    pub fn update(&mut self, time: u64, terrain_grid: &TerrainGrid, season: Season) {
        if !self.config.enabled {
            return;
        }

        // Only update at intervals for performance
        if time.saturating_sub(self.last_update) < self.config.update_interval {
            return;
        }

        self.recalculate(terrain_grid, season);
        self.last_update = time;
    }

    /// Force recalculate all temperature and humidity values
    pub fn recalculate(&mut self, terrain_grid: &TerrainGrid, season: Season) {
        if !self.config.enabled {
            return;
        }

        for y in 0..self.grid_size {
            for x in 0..self.grid_size {
                let terrain = terrain_grid.get(x as u8, y as u8);
                let near_water = self.check_near_water(x, y, terrain_grid);

                self.temperature_grid[y][x] = self.calculate_temperature(x, y, terrain, season);
                self.humidity_grid[y][x] = self.calculate_humidity(x, y, terrain, near_water, season);
            }
        }
    }

    /// Calculate temperature at a cell based on: season, altitude, latitude
    fn calculate_temperature(&self, _x: usize, y: usize, terrain: Terrain, season: Season) -> f32 {
        let mut temp = self.config.base_temperature;

        // Seasonal effect
        temp += match season {
            Season::Summer => self.config.seasonal_variation,
            Season::Winter => -self.config.seasonal_variation,
            Season::Spring => self.config.seasonal_variation * 0.3,
            Season::Autumn => -self.config.seasonal_variation * 0.3,
        };

        // Altitude effect (mountains are colder)
        if terrain == Terrain::Mountain {
            temp -= self.config.altitude_cooling;
        }

        // Water is slightly cooler in summer, warmer in winter (thermal mass)
        if terrain == Terrain::Water {
            match season {
                Season::Summer => temp -= 0.1,
                Season::Winter => temp += 0.1,
                _ => {}
            }
        }

        // Latitude effect (y=0 is north/cold, y=max is south/warm)
        let latitude_factor = (y as f32 / self.grid_size as f32) - 0.5; // -0.5 to +0.5
        temp += latitude_factor * self.config.latitude_effect;

        // Desert is hot during day (simplified as warmer overall)
        if terrain == Terrain::Desert {
            temp += 0.15;
        }

        // Forest provides shade (slightly cooler)
        if terrain == Terrain::Forest {
            temp -= 0.05;
        }

        temp.clamp(-1.0, 1.0)
    }

    /// Calculate humidity at a cell based on: terrain, proximity to water, season
    fn calculate_humidity(&self, _x: usize, _y: usize, terrain: Terrain, near_water: bool, season: Season) -> f32 {
        let mut humidity = 0.5; // Base humidity

        // Terrain effects
        match terrain {
            Terrain::Water => humidity = 1.0,
            Terrain::Forest => humidity += 0.2,
            Terrain::Desert => humidity -= self.config.desert_humidity_penalty,
            Terrain::Mountain => humidity -= 0.1,
            Terrain::Plain => {}
        }

        // Proximity to water bonus
        if near_water && terrain != Terrain::Water {
            humidity += self.config.water_humidity_bonus;
        }

        // Seasonal effects
        match season {
            Season::Spring => humidity += 0.1, // Rainy season
            Season::Summer => humidity -= 0.1, // Dry heat
            Season::Autumn => humidity += 0.05,
            Season::Winter => humidity -= 0.05, // Cold and dry
        }

        humidity.clamp(0.0, 1.0)
    }

    /// Check if a cell is adjacent to water
    fn check_near_water(&self, x: usize, y: usize, terrain_grid: &TerrainGrid) -> bool {
        let check_radius = 2; // Check within 2 cells

        for dy in 0..=check_radius * 2 {
            for dx in 0..=check_radius * 2 {
                let nx = x as i32 + dx as i32 - check_radius as i32;
                let ny = y as i32 + dy as i32 - check_radius as i32;

                if nx >= 0 && ny >= 0 && (nx as usize) < self.grid_size && (ny as usize) < self.grid_size {
                    if terrain_grid.get(nx as u8, ny as u8) == Terrain::Water {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Get temperature at position (-1.0 to 1.0)
    #[inline]
    pub fn get_temperature(&self, x: u8, y: u8) -> f32 {
        if !self.config.enabled {
            return 0.5; // Neutral if disabled
        }
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.temperature_grid[y][x]
        } else {
            0.5 // Default neutral
        }
    }

    /// Get humidity at position (0.0 to 1.0)
    #[inline]
    pub fn get_humidity(&self, x: u8, y: u8) -> f32 {
        if !self.config.enabled {
            return 0.5; // Neutral if disabled
        }
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.humidity_grid[y][x]
        } else {
            0.5 // Default neutral
        }
    }

    /// Get food regeneration multiplier based on local climate
    /// High humidity = faster regen, extreme temperatures = slower regen
    pub fn get_food_regen_multiplier(&self, x: u8, y: u8) -> f32 {
        if !self.config.enabled {
            return 1.0;
        }

        let temp = self.get_temperature(x, y);
        let humidity = self.get_humidity(x, y);

        // Humidity effect: 0.0 humidity = 0.7x, 1.0 humidity = 1.3x
        let humidity_factor = 0.7 + humidity * 0.6;

        // Temperature effect: optimal around 0.3-0.7, penalty at extremes
        let temp_factor = if temp < -0.5 {
            0.5 // Very cold: slow growth
        } else if temp < 0.0 {
            0.7 + (temp + 0.5) * 0.6 // Cold: reduced
        } else if temp <= 0.5 {
            1.0 // Optimal range
        } else if temp <= 0.8 {
            1.0 - (temp - 0.5) * 0.5 // Getting too hot
        } else {
            0.6 // Very hot: reduced growth
        };

        (humidity_factor * temp_factor).clamp(0.3, 1.5)
    }

    /// Get metabolic cost modifier based on temperature
    /// Cold = higher metabolism (keeping warm), hot = slightly higher (cooling)
    pub fn get_metabolism_modifier(&self, x: u8, y: u8) -> f32 {
        if !self.config.enabled {
            return 1.0;
        }

        let temp = self.get_temperature(x, y);

        // Optimal temperature around 0.3-0.5
        if temp < -0.3 {
            // Very cold: +30% metabolism cost
            1.3
        } else if temp < 0.2 {
            // Cold: +15% metabolism cost
            1.15
        } else if temp <= 0.6 {
            // Optimal: no penalty
            1.0
        } else if temp <= 0.8 {
            // Hot: +10% metabolism cost
            1.1
        } else {
            // Very hot: +20% metabolism cost
            1.2
        }
    }

    /// Get movement speed modifier based on temperature
    /// Extreme cold slows movement, extreme heat also slows
    pub fn get_movement_modifier(&self, x: u8, y: u8) -> f32 {
        if !self.config.enabled {
            return 1.0;
        }

        let temp = self.get_temperature(x, y);

        if temp < -0.5 {
            0.7 // Very cold: 30% slower
        } else if temp < 0.0 {
            0.85 // Cold: 15% slower
        } else if temp <= 0.7 {
            1.0 // Normal speed
        } else if temp <= 0.9 {
            0.9 // Hot: 10% slower
        } else {
            0.8 // Very hot: 20% slower
        }
    }

    /// Check if temperature is suitable for reproduction
    /// Extreme temperatures reduce reproduction success
    pub fn get_reproduction_modifier(&self, x: u8, y: u8) -> f32 {
        if !self.config.enabled {
            return 1.0;
        }

        let temp = self.get_temperature(x, y);

        if temp < -0.4 || temp > 0.8 {
            0.5 // Extreme temperatures: 50% reproduction penalty
        } else if temp < 0.0 || temp > 0.7 {
            0.75 // Suboptimal: 25% penalty
        } else {
            1.0 // Optimal for reproduction
        }
    }

    /// Get grid size
    pub fn grid_size(&self) -> usize {
        self.grid_size
    }

    /// Flatten temperature grid for snapshot (row-major)
    pub fn flatten_temperature(&self) -> Vec<f32> {
        self.temperature_grid.iter().flatten().copied().collect()
    }

    /// Flatten humidity grid for snapshot (row-major)
    pub fn flatten_humidity(&self) -> Vec<f32> {
        self.humidity_grid.iter().flatten().copied().collect()
    }

    /// Calculate average temperature across the grid
    pub fn average_temperature(&self) -> f32 {
        if !self.config.enabled || self.grid_size == 0 {
            return 0.5;
        }
        let sum: f32 = self.temperature_grid.iter().flatten().sum();
        let count = (self.grid_size * self.grid_size) as f32;
        sum / count
    }

    /// Calculate average humidity across the grid
    pub fn average_humidity(&self) -> f32 {
        if !self.config.enabled || self.grid_size == 0 {
            return 0.5;
        }
        let sum: f32 = self.humidity_grid.iter().flatten().sum();
        let count = (self.grid_size * self.grid_size) as f32;
        sum / count
    }

    /// Get global food regeneration multiplier based on season and time of day
    /// (for logging/statistics, not position-specific)
    pub fn get_global_food_regen_multiplier(&self, season: Season, is_daytime: bool) -> f32 {
        // Base seasonal multiplier
        let season_mult = match season {
            Season::Spring => 0.6,
            Season::Summer => 0.8,
            Season::Autumn => 0.56,
            Season::Winter => 0.1,
        };

        // Day/night modifier
        let daynight_mult = if is_daytime { 1.0 } else { 0.6 };

        season_mult * daynight_mult
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_terrain(grid_size: usize) -> TerrainGrid {
        let mut grid = TerrainGrid::new(grid_size);
        // Set some variety
        grid.set(5, 5, Terrain::Water);
        grid.set(10, 10, Terrain::Mountain);
        grid.set(15, 15, Terrain::Desert);
        grid.set(20, 20, Terrain::Forest);
        grid
    }

    #[test]
    fn test_climate_creation() {
        let terrain = make_test_terrain(50);
        let config = ClimateConfig::default();
        let climate = ClimateSystem::new(50, &terrain, &config);

        assert_eq!(climate.grid_size(), 50);
        assert!(climate.is_enabled());
    }

    #[test]
    fn test_temperature_range() {
        let terrain = make_test_terrain(50);
        let config = ClimateConfig::default();
        let climate = ClimateSystem::new(50, &terrain, &config);

        for y in 0..50 {
            for x in 0..50 {
                let temp = climate.get_temperature(x, y);
                assert!(temp >= -1.0 && temp <= 1.0, "Temperature out of range: {}", temp);
            }
        }
    }

    #[test]
    fn test_humidity_range() {
        let terrain = make_test_terrain(50);
        let config = ClimateConfig::default();
        let climate = ClimateSystem::new(50, &terrain, &config);

        for y in 0..50 {
            for x in 0..50 {
                let humidity = climate.get_humidity(x, y);
                assert!(humidity >= 0.0 && humidity <= 1.0, "Humidity out of range: {}", humidity);
            }
        }
    }

    #[test]
    fn test_seasonal_temperature_variation() {
        let terrain = make_test_terrain(50);
        let config = ClimateConfig::default();
        let mut climate = ClimateSystem::new(50, &terrain, &config);

        // Get summer temperature
        climate.recalculate(&terrain, Season::Summer);
        let summer_temp = climate.get_temperature(25, 25);

        // Get winter temperature
        climate.recalculate(&terrain, Season::Winter);
        let winter_temp = climate.get_temperature(25, 25);

        // Summer should be warmer than winter
        assert!(summer_temp > winter_temp, "Summer ({}) should be warmer than winter ({})", summer_temp, winter_temp);
    }

    #[test]
    fn test_mountain_cooling() {
        let mut terrain = TerrainGrid::new(50);
        terrain.set(25, 25, Terrain::Mountain);
        terrain.set(30, 30, Terrain::Plain);

        let config = ClimateConfig::default();
        let climate = ClimateSystem::new(50, &terrain, &config);

        let mountain_temp = climate.get_temperature(25, 25);
        let plain_temp = climate.get_temperature(30, 30);

        // Mountain should be cooler
        assert!(mountain_temp < plain_temp, "Mountain ({}) should be cooler than plain ({})", mountain_temp, plain_temp);
    }

    #[test]
    fn test_water_humidity() {
        let mut terrain = TerrainGrid::new(50);
        terrain.set(25, 25, Terrain::Water);
        terrain.set(40, 40, Terrain::Desert);

        let config = ClimateConfig::default();
        let climate = ClimateSystem::new(50, &terrain, &config);

        let water_humidity = climate.get_humidity(25, 25);
        let desert_humidity = climate.get_humidity(40, 40);

        // Water should be more humid than desert
        assert!(water_humidity > desert_humidity, "Water ({}) should be more humid than desert ({})", water_humidity, desert_humidity);
    }

    #[test]
    fn test_food_regen_multiplier() {
        let terrain = make_test_terrain(50);
        let config = ClimateConfig::default();
        let climate = ClimateSystem::new(50, &terrain, &config);

        let multiplier = climate.get_food_regen_multiplier(25, 25);
        assert!(multiplier >= 0.3 && multiplier <= 1.5, "Food regen multiplier out of range: {}", multiplier);
    }

    #[test]
    fn test_disabled_climate() {
        let terrain = make_test_terrain(50);
        let config = ClimateConfig {
            enabled: false,
            ..Default::default()
        };
        let climate = ClimateSystem::new(50, &terrain, &config);

        // All values should be neutral when disabled
        assert!((climate.get_temperature(25, 25) - 0.5).abs() < 0.01);
        assert!((climate.get_humidity(25, 25) - 0.5).abs() < 0.01);
        assert!((climate.get_food_regen_multiplier(25, 25) - 1.0).abs() < 0.01);
        assert!((climate.get_metabolism_modifier(25, 25) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_latitude_gradient() {
        let terrain = TerrainGrid::new(50);
        let config = ClimateConfig::default();
        let climate = ClimateSystem::new(50, &terrain, &config);

        let north_temp = climate.get_temperature(25, 5);  // North
        let south_temp = climate.get_temperature(25, 45); // South

        // South should be warmer (higher y = warmer)
        assert!(south_temp > north_temp, "South ({}) should be warmer than north ({})", south_temp, north_temp);
    }
}
