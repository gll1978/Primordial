//! Food patch system for foraging memory pressure.
//!
//! Creates localized food patches that deplete when eaten and regenerate over time,
//! forcing organisms to develop spatial memory to remember patch locations.

use serde::{Deserialize, Serialize};

/// Configuration for the food patch system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchConfig {
    pub patch_count: usize,
    pub initial_capacity: f32,
    pub depletion_rate: f32,
    pub regeneration_rate: f32,
    pub regeneration_time: u64,
    pub min_distance: u8,
    pub patch_radius: u8,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            patch_count: 8,
            initial_capacity: 80.0,
            depletion_rate: 5.0,
            regeneration_rate: 0.5,
            regeneration_time: 200,
            min_distance: 10,
            patch_radius: 2,
        }
    }
}

/// A single food patch with depletion/regeneration
#[derive(Debug, Clone)]
pub struct FoodPatch {
    pub x: u8,
    pub y: u8,
    pub capacity: f32,
    pub max_capacity: f32,
    pub depleted_at: Option<u64>,
    pub times_depleted: u32,
    pub times_visited: u64,
}

impl FoodPatch {
    pub fn new(x: u8, y: u8, max_capacity: f32) -> Self {
        Self {
            x,
            y,
            capacity: max_capacity,
            max_capacity,
            depleted_at: None,
            times_depleted: 0,
            times_visited: 0,
        }
    }

    /// Deplete the patch by a given amount
    pub fn deplete(&mut self, amount: f32, time: u64) {
        self.capacity = (self.capacity - amount).max(0.0);
        self.times_visited += 1;
        if self.capacity <= 0.0 && self.depleted_at.is_none() {
            self.depleted_at = Some(time);
            self.times_depleted += 1;
        }
    }

    /// Try to regenerate the patch
    pub fn regenerate(&mut self, time: u64, regen_time: u64, regen_rate: f32) {
        if let Some(depleted_time) = self.depleted_at {
            if time >= depleted_time + regen_time {
                self.capacity = (self.capacity + regen_rate).min(self.max_capacity);
                if self.capacity >= self.max_capacity * 0.5 {
                    self.depleted_at = None;
                }
            }
        } else if self.capacity < self.max_capacity {
            self.capacity = (self.capacity + regen_rate).min(self.max_capacity);
        }
    }

    /// Whether the patch has food available
    pub fn has_food(&self) -> bool {
        self.capacity > 1.0
    }

    /// Manhattan distance from this patch to a point
    pub fn distance_to(&self, x: u8, y: u8) -> u8 {
        let dx = (self.x as i16 - x as i16).unsigned_abs() as u8;
        let dy = (self.y as i16 - y as i16).unsigned_abs() as u8;
        dx.saturating_add(dy)
    }
}

/// Manages all food patches in the world
pub struct PatchWorld {
    pub patches: Vec<FoodPatch>,
    pub config: PatchConfig,
    pub grid_size: u8,
}

impl PatchWorld {
    /// Create a new PatchWorld with randomly placed patches
    pub fn new(config: &PatchConfig, grid_size: u8, rng: &mut impl rand::Rng) -> Self {
        let mut patches = Vec::with_capacity(config.patch_count);

        for _ in 0..config.patch_count {
            let mut attempts = 0;
            loop {
                let x = rng.gen_range(config.patch_radius..grid_size.saturating_sub(config.patch_radius));
                let y = rng.gen_range(config.patch_radius..grid_size.saturating_sub(config.patch_radius));

                // Check min distance from existing patches
                let too_close = patches.iter().any(|p: &FoodPatch| {
                    p.distance_to(x, y) < config.min_distance
                });

                if !too_close || attempts > 100 {
                    patches.push(FoodPatch::new(x, y, config.initial_capacity));
                    break;
                }
                attempts += 1;
            }
        }

        Self {
            patches,
            config: config.clone(),
            grid_size,
        }
    }

    /// Update all patches (regeneration)
    pub fn update(&mut self, time: u64) {
        for patch in &mut self.patches {
            patch.regenerate(time, self.config.regeneration_time, self.config.regeneration_rate);
        }
    }

    /// Find the nearest patch within radius to a given position
    pub fn get_nearest_patch(&self, x: u8, y: u8, max_dist: u8) -> Option<usize> {
        let mut best: Option<(usize, u8)> = None;
        for (i, patch) in self.patches.iter().enumerate() {
            let dist = patch.distance_to(x, y);
            if dist <= max_dist {
                if best.is_none() || dist < best.unwrap().1 {
                    best = Some((i, dist));
                }
            }
        }
        best.map(|(i, _)| i)
    }

    /// Write patch food signals onto the food grid
    pub fn write_to_food_grid(&self, food_grid: &mut crate::grid::FoodGrid, food_max: f32) {
        for patch in &self.patches {
            if !patch.has_food() {
                continue;
            }
            let radius = self.config.patch_radius as i16;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = patch.x as i16 + dx;
                    let ny = patch.y as i16 + dy;
                    if nx < 0 || ny < 0 || nx >= self.grid_size as i16 || ny >= self.grid_size as i16 {
                        continue;
                    }
                    let manhattan = (dx.abs() + dy.abs()) as f32;
                    let max_manhattan = (radius * 2) as f32;
                    let falloff = 1.0 - (manhattan / max_manhattan).min(1.0);
                    let food_amount = patch.capacity * falloff * 0.5;

                    let cx = nx as u8;
                    let cy = ny as u8;
                    let current = food_grid.get(cx, cy);
                    let new_val = (current + food_amount).min(food_max);
                    food_grid.set(cx, cy, new_val);
                }
            }
        }
    }

    /// Get aggregate stats
    pub fn stats(&self) -> PatchStats {
        let active = self.patches.iter().filter(|p| p.has_food()).count();
        let depleted = self.patches.len() - active;
        let total_capacity: f32 = self.patches.iter().map(|p| p.capacity).sum();
        let total_max: f32 = self.patches.iter().map(|p| p.max_capacity).sum();
        let total_visits: u64 = self.patches.iter().map(|p| p.times_visited).sum();
        let total_depletions: u32 = self.patches.iter().map(|p| p.times_depleted).sum();

        PatchStats {
            active_patches: active,
            depleted_patches: depleted,
            total_capacity,
            capacity_ratio: if total_max > 0.0 { total_capacity / total_max } else { 0.0 },
            total_visits,
            total_depletions,
        }
    }
}

/// Aggregate statistics for food patches
#[derive(Debug, Clone)]
pub struct PatchStats {
    pub active_patches: usize,
    pub depleted_patches: usize,
    pub total_capacity: f32,
    pub capacity_ratio: f32,
    pub total_visits: u64,
    pub total_depletions: u32,
}

impl std::fmt::Display for PatchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Patches: {}/{} active, capacity {:.0}%, visits {}, depletions {}",
            self.active_patches,
            self.active_patches + self.depleted_patches,
            self.capacity_ratio * 100.0,
            self.total_visits,
            self.total_depletions,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_food_patch_depletion() {
        let mut patch = FoodPatch::new(10, 10, 50.0);
        assert!(patch.has_food());
        for _ in 0..20 {
            patch.deplete(5.0, 0);
        }
        assert!(!patch.has_food());
        assert_eq!(patch.times_depleted, 1);
    }

    #[test]
    fn test_food_patch_regeneration() {
        let mut patch = FoodPatch::new(10, 10, 50.0);
        patch.deplete(50.0, 100);
        assert!(!patch.has_food());
        // Before regen time
        patch.regenerate(200, 200, 5.0);
        assert!(!patch.has_food());
        // After regen time
        for t in 300..320 {
            patch.regenerate(t, 200, 5.0);
        }
        assert!(patch.has_food());
    }

    #[test]
    fn test_patch_world_creation() {
        let config = PatchConfig::default();
        let mut rng = rand::thread_rng();
        let world = PatchWorld::new(&config, 100, &mut rng);
        assert_eq!(world.patches.len(), config.patch_count);
    }

    #[test]
    fn test_get_nearest_patch() {
        let config = PatchConfig { patch_count: 1, min_distance: 0, ..Default::default() };
        let mut rng = rand::thread_rng();
        let world = PatchWorld::new(&config, 100, &mut rng);
        let px = world.patches[0].x;
        let py = world.patches[0].y;
        assert!(world.get_nearest_patch(px, py, 5).is_some());
        assert!(world.get_nearest_patch(px.wrapping_add(100), py.wrapping_add(100), 2).is_none());
    }

    #[test]
    fn test_patch_distance() {
        let patch = FoodPatch::new(50, 50, 100.0);
        assert_eq!(patch.distance_to(50, 50), 0);
        assert_eq!(patch.distance_to(52, 53), 5);
    }
}
