//! Dynamic obstacle system for Phase 2 Feature 4.
//!
//! Creates obstacles that can move, appear, and disappear over time,
//! forcing organisms to adapt their navigation strategies.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for dynamic obstacles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicObstacleConfig {
    /// Enable dynamic obstacles
    pub enabled: bool,
    /// Maximum number of obstacles
    pub max_obstacles: usize,
    /// Initial number of obstacles
    pub initial_obstacles: usize,
    /// Probability of obstacle moving each step (0.01 = 1%)
    pub movement_chance: f32,
    /// Probability of new obstacle spawning each step
    pub spawn_rate: f32,
    /// Probability of obstacle despawning each step
    pub despawn_rate: f32,
    /// Minimum lifetime before despawn is possible (steps)
    pub min_lifetime: u64,
}

impl Default for DynamicObstacleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_obstacles: 50,
            initial_obstacles: 20,
            movement_chance: 0.01,
            spawn_rate: 0.001,
            despawn_rate: 0.0005,
            min_lifetime: 100,
        }
    }
}

/// A single dynamic obstacle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicObstacle {
    pub x: u8,
    pub y: u8,
    pub created_at: u64,
    pub last_moved: u64,
    pub times_moved: u32,
}

impl DynamicObstacle {
    pub fn new(x: u8, y: u8, time: u64) -> Self {
        Self {
            x,
            y,
            created_at: time,
            last_moved: time,
            times_moved: 0,
        }
    }

    /// Get obstacle age in steps
    pub fn age(&self, current_time: u64) -> u64 {
        current_time.saturating_sub(self.created_at)
    }
}

/// System managing all dynamic obstacles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicObstacleSystem {
    pub obstacles: Vec<DynamicObstacle>,
    pub config: DynamicObstacleConfig,
    pub grid_size: u8,
    /// Bitmap for fast collision check (flattened 2D array)
    #[serde(skip)]
    obstacle_map: Vec<bool>,
}

impl DynamicObstacleSystem {
    /// Create a new system with initial obstacles
    pub fn new(config: &DynamicObstacleConfig, grid_size: u8, rng: &mut impl Rng) -> Self {
        let mut system = Self {
            obstacles: Vec::with_capacity(config.max_obstacles),
            config: config.clone(),
            grid_size,
            obstacle_map: vec![false; (grid_size as usize) * (grid_size as usize)],
        };

        if config.enabled {
            // Spawn initial obstacles
            for _ in 0..config.initial_obstacles {
                system.spawn_obstacle(rng, 0);
            }
        }

        system
    }

    /// Rebuild obstacle map from obstacles list (call after deserialization)
    pub fn rebuild_map(&mut self) {
        self.obstacle_map = vec![false; (self.grid_size as usize) * (self.grid_size as usize)];
        for obs in &self.obstacles {
            let idx = obs.y as usize * self.grid_size as usize + obs.x as usize;
            if idx < self.obstacle_map.len() {
                self.obstacle_map[idx] = true;
            }
        }
    }

    /// Check if a cell has an obstacle
    #[inline]
    pub fn is_blocked(&self, x: u8, y: u8) -> bool {
        let idx = y as usize * self.grid_size as usize + x as usize;
        idx < self.obstacle_map.len() && self.obstacle_map[idx]
    }

    /// Update the obstacle map when an obstacle moves
    fn update_map(&mut self, old_x: u8, old_y: u8, new_x: u8, new_y: u8) {
        let old_idx = old_y as usize * self.grid_size as usize + old_x as usize;
        let new_idx = new_y as usize * self.grid_size as usize + new_x as usize;

        if old_idx < self.obstacle_map.len() {
            self.obstacle_map[old_idx] = false;
        }
        if new_idx < self.obstacle_map.len() {
            self.obstacle_map[new_idx] = true;
        }
    }

    /// Spawn a new obstacle at a random position
    pub fn spawn_obstacle(&mut self, rng: &mut impl Rng, time: u64) -> bool {
        if self.obstacles.len() >= self.config.max_obstacles {
            return false;
        }

        // Try to find an empty spot
        for _ in 0..50 {
            let x = rng.gen_range(0..self.grid_size);
            let y = rng.gen_range(0..self.grid_size);

            if !self.is_blocked(x, y) {
                let idx = y as usize * self.grid_size as usize + x as usize;
                if idx < self.obstacle_map.len() {
                    self.obstacle_map[idx] = true;
                }
                self.obstacles.push(DynamicObstacle::new(x, y, time));
                return true;
            }
        }

        false
    }

    /// Remove an obstacle by index
    fn remove_obstacle(&mut self, idx: usize) {
        if idx < self.obstacles.len() {
            let obs = self.obstacles.swap_remove(idx);
            let map_idx = obs.y as usize * self.grid_size as usize + obs.x as usize;
            if map_idx < self.obstacle_map.len() {
                self.obstacle_map[map_idx] = false;
            }
        }
    }

    /// Move an obstacle to a random adjacent cell
    fn move_obstacle(&mut self, idx: usize, rng: &mut impl Rng, time: u64) -> bool {
        if idx >= self.obstacles.len() {
            return false;
        }

        let obs = &self.obstacles[idx];
        let old_x = obs.x;
        let old_y = obs.y;

        // Try random directions
        let directions: [(i8, i8); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
        let mut shuffled = directions;
        for i in (1..4).rev() {
            let j = rng.gen_range(0..=i);
            shuffled.swap(i, j);
        }

        for (dx, dy) in shuffled {
            let new_x = old_x.saturating_add_signed(dx);
            let new_y = old_y.saturating_add_signed(dy);

            // Check bounds
            if new_x >= self.grid_size || new_y >= self.grid_size {
                continue;
            }

            // Check not blocked
            if self.is_blocked(new_x, new_y) {
                continue;
            }

            // Move obstacle
            self.update_map(old_x, old_y, new_x, new_y);
            let obs = &mut self.obstacles[idx];
            obs.x = new_x;
            obs.y = new_y;
            obs.last_moved = time;
            obs.times_moved += 1;
            return true;
        }

        false
    }

    /// Update the obstacle system for one step
    pub fn update(&mut self, time: u64, rng: &mut impl Rng) {
        if !self.config.enabled {
            return;
        }

        // Process obstacles in reverse order for safe removal
        let mut i = self.obstacles.len();
        while i > 0 {
            i -= 1;

            // Check despawn
            let age = self.obstacles[i].age(time);
            if age >= self.config.min_lifetime && rng.gen::<f32>() < self.config.despawn_rate {
                self.remove_obstacle(i);
                continue;
            }

            // Check movement
            if rng.gen::<f32>() < self.config.movement_chance {
                self.move_obstacle(i, rng, time);
            }
        }

        // Check spawn
        if rng.gen::<f32>() < self.config.spawn_rate {
            self.spawn_obstacle(rng, time);
        }
    }

    /// Get number of obstacles
    pub fn count(&self) -> usize {
        self.obstacles.len()
    }

    /// Get statistics
    pub fn stats(&self, time: u64) -> ObstacleStats {
        let total_age: u64 = self.obstacles.iter().map(|o| o.age(time)).sum();
        let total_moves: u32 = self.obstacles.iter().map(|o| o.times_moved).sum();

        ObstacleStats {
            count: self.obstacles.len(),
            avg_age: if self.obstacles.is_empty() { 0.0 } else { total_age as f32 / self.obstacles.len() as f32 },
            total_moves,
        }
    }
}

/// Statistics about dynamic obstacles
#[derive(Debug, Clone)]
pub struct ObstacleStats {
    pub count: usize,
    pub avg_age: f32,
    pub total_moves: u32,
}

impl std::fmt::Display for ObstacleStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Obstacles: {} (avg age {:.0}, {} moves)",
            self.count, self.avg_age, self.total_moves
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obstacle_creation() {
        let config = DynamicObstacleConfig::default();
        let mut rng = rand::thread_rng();
        let system = DynamicObstacleSystem::new(&config, 50, &mut rng);

        assert_eq!(system.count(), config.initial_obstacles);
    }

    #[test]
    fn test_obstacle_blocking() {
        let config = DynamicObstacleConfig {
            enabled: true,
            initial_obstacles: 1,
            max_obstacles: 10,
            ..Default::default()
        };
        let mut rng = rand::thread_rng();
        let system = DynamicObstacleSystem::new(&config, 50, &mut rng);

        // At least one cell should be blocked
        let blocked_count: usize = (0..50)
            .flat_map(|y| (0..50).map(move |x| (x, y)))
            .filter(|&(x, y)| system.is_blocked(x, y))
            .count();

        assert_eq!(blocked_count, 1);
    }

    #[test]
    fn test_obstacle_spawn() {
        let config = DynamicObstacleConfig {
            enabled: true,
            initial_obstacles: 0,
            max_obstacles: 10,
            ..Default::default()
        };
        let mut rng = rand::thread_rng();
        let mut system = DynamicObstacleSystem::new(&config, 50, &mut rng);

        assert_eq!(system.count(), 0);

        let spawned = system.spawn_obstacle(&mut rng, 0);
        assert!(spawned);
        assert_eq!(system.count(), 1);
    }

    #[test]
    fn test_obstacle_movement() {
        let config = DynamicObstacleConfig {
            enabled: true,
            initial_obstacles: 1,
            max_obstacles: 10,
            movement_chance: 1.0, // Always move
            ..Default::default()
        };
        let mut rng = rand::thread_rng();
        let mut system = DynamicObstacleSystem::new(&config, 50, &mut rng);

        let initial_x = system.obstacles[0].x;
        let initial_y = system.obstacles[0].y;

        // Update multiple times to ensure movement
        for t in 1..10 {
            system.update(t, &mut rng);
        }

        let moved = system.obstacles[0].x != initial_x || system.obstacles[0].y != initial_y;
        assert!(moved || system.obstacles[0].times_moved > 0);
    }

    #[test]
    fn test_obstacle_despawn() {
        let config = DynamicObstacleConfig {
            enabled: true,
            initial_obstacles: 10,
            max_obstacles: 10,
            despawn_rate: 1.0, // Always despawn
            min_lifetime: 0,   // No minimum
            ..Default::default()
        };
        let mut rng = rand::thread_rng();
        let mut system = DynamicObstacleSystem::new(&config, 50, &mut rng);

        assert_eq!(system.count(), 10);

        system.update(1, &mut rng);

        // Some obstacles should have despawned
        assert!(system.count() < 10);
    }

    #[test]
    fn test_max_obstacles() {
        let config = DynamicObstacleConfig {
            enabled: true,
            initial_obstacles: 5,
            max_obstacles: 5,
            ..Default::default()
        };
        let mut rng = rand::thread_rng();
        let mut system = DynamicObstacleSystem::new(&config, 50, &mut rng);

        // Should not spawn beyond max
        let spawned = system.spawn_obstacle(&mut rng, 0);
        assert!(!spawned);
        assert_eq!(system.count(), 5);
    }

    #[test]
    fn test_disabled_system() {
        let config = DynamicObstacleConfig {
            enabled: false,
            ..Default::default()
        };
        let mut rng = rand::thread_rng();
        let system = DynamicObstacleSystem::new(&config, 50, &mut rng);

        assert_eq!(system.count(), 0);
    }
}
