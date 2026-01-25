//! Multi-terrain system with movement costs and food modifiers.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Terrain types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Terrain {
    Plain,
    Forest,
    Mountain,
    Desert,
    Water,
}

impl Terrain {
    /// Movement cost multiplier (higher = harder to traverse)
    pub fn movement_cost(&self, is_aquatic: bool) -> f32 {
        match self {
            Terrain::Plain => 1.0,
            Terrain::Forest => 1.3,
            Terrain::Mountain => 2.5,
            Terrain::Desert => 1.8,
            Terrain::Water => {
                if is_aquatic {
                    0.5 // Aquatic organisms move easily in water
                } else {
                    10.0 // Non-aquatic can barely move in water
                }
            }
        }
    }

    /// Food availability multiplier
    pub fn food_multiplier(&self) -> f32 {
        match self {
            Terrain::Plain => 1.0,
            Terrain::Forest => 1.5,    // Abundant food
            Terrain::Mountain => 0.4,  // Sparse
            Terrain::Desert => 0.2,    // Very sparse
            Terrain::Water => 0.8,     // Fish/algae
        }
    }

    /// Vision range modifier (added to base vision)
    pub fn vision_modifier(&self) -> i8 {
        match self {
            Terrain::Plain => 0,
            Terrain::Forest => -1,    // Reduced visibility
            Terrain::Mountain => 2,   // Can see farther
            Terrain::Desert => 1,     // Clear view
            Terrain::Water => -1,     // Limited visibility
        }
    }

    /// Is this terrain passable for a given organism?
    pub fn is_passable(&self, is_aquatic: bool) -> bool {
        match self {
            Terrain::Water => is_aquatic,
            _ => true,
        }
    }

    /// Generate random terrain based on probabilities
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..100) {
            0..=45 => Terrain::Plain,     // 45%
            46..=70 => Terrain::Forest,   // 25%
            71..=85 => Terrain::Mountain, // 15%
            86..=95 => Terrain::Desert,   // 10%
            _ => Terrain::Water,          // 5%
        }
    }

    /// Get display character for visualization
    pub fn char(&self) -> char {
        match self {
            Terrain::Plain => '.',
            Terrain::Forest => 'T',
            Terrain::Mountain => '^',
            Terrain::Desert => '~',
            Terrain::Water => 'W',
        }
    }

    /// Get color code for visualization (ANSI)
    pub fn color_code(&self) -> &'static str {
        match self {
            Terrain::Plain => "\x1b[32m",    // Green
            Terrain::Forest => "\x1b[92m",   // Bright green
            Terrain::Mountain => "\x1b[90m", // Gray
            Terrain::Desert => "\x1b[33m",   // Yellow
            Terrain::Water => "\x1b[34m",    // Blue
        }
    }
}

/// Terrain grid for the world
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TerrainGrid {
    pub grid: Vec<Vec<Terrain>>,
    pub grid_size: usize,
}

impl TerrainGrid {
    /// Create a new terrain grid filled with plains
    pub fn new(grid_size: usize) -> Self {
        Self {
            grid: vec![vec![Terrain::Plain; grid_size]; grid_size],
            grid_size,
        }
    }

    /// Generate random terrain with clustering (more realistic)
    pub fn generate_clustered(&mut self, seed: Option<u64>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // First pass: random terrain
        for y in 0..self.grid_size {
            for x in 0..self.grid_size {
                self.grid[y][x] = Terrain::random();
            }
        }

        // Second pass: clustering (neighbors influence each other)
        let old_grid = self.grid.clone();
        for y in 1..self.grid_size - 1 {
            for x in 1..self.grid_size - 1 {
                // 60% chance to match a random neighbor
                if rng.gen::<f32>() < 0.6 {
                    let neighbors = [
                        old_grid[y - 1][x],
                        old_grid[y + 1][x],
                        old_grid[y][x - 1],
                        old_grid[y][x + 1],
                    ];
                    let idx = rng.gen_range(0..4);
                    self.grid[y][x] = neighbors[idx];
                }
            }
        }

        // Third pass: create some water bodies (lakes/rivers)
        self.create_water_bodies(&mut rng);
    }

    /// Create connected water bodies
    fn create_water_bodies(&mut self, rng: &mut impl Rng) {
        // Create 1-3 lakes
        let num_lakes = rng.gen_range(1..=3);

        for _ in 0..num_lakes {
            let center_x = rng.gen_range(5..self.grid_size - 5);
            let center_y = rng.gen_range(5..self.grid_size - 5);
            let radius = rng.gen_range(2..=5);

            for dy in 0..=radius * 2 {
                for dx in 0..=radius * 2 {
                    let x = center_x + dx - radius;
                    let y = center_y + dy - radius;

                    if x < self.grid_size && y < self.grid_size {
                        let dist = ((dx as i32 - radius as i32).pow(2)
                            + (dy as i32 - radius as i32).pow(2)) as f32;
                        if dist.sqrt() <= radius as f32 {
                            self.grid[y][x] = Terrain::Water;
                        }
                    }
                }
            }
        }
    }

    /// Generate terrain with a central mountain range (for isolation experiments)
    pub fn generate_with_barrier(&mut self, vertical: bool) {
        // First generate normal terrain
        self.generate_clustered(None);

        // Then add a mountain barrier
        let mid = self.grid_size / 2;
        let width = 3;

        if vertical {
            // Vertical barrier (East-West split)
            for y in 0..self.grid_size {
                for dx in 0..width {
                    let x = mid - width / 2 + dx;
                    if x < self.grid_size {
                        self.grid[y][x] = Terrain::Mountain;
                    }
                }
            }
        } else {
            // Horizontal barrier (North-South split)
            for x in 0..self.grid_size {
                for dy in 0..width {
                    let y = mid - width / 2 + dy;
                    if y < self.grid_size {
                        self.grid[y][x] = Terrain::Mountain;
                    }
                }
            }
        }
    }

    /// Get terrain at position
    #[inline]
    pub fn get(&self, x: u8, y: u8) -> Terrain {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.grid[y][x]
        } else {
            Terrain::Mountain // Out of bounds = impassable
        }
    }

    /// Set terrain at position
    #[inline]
    pub fn set(&mut self, x: u8, y: u8, terrain: Terrain) {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.grid[y][x] = terrain;
        }
    }

    /// Count terrain types
    pub fn terrain_counts(&self) -> std::collections::HashMap<Terrain, usize> {
        let mut counts = std::collections::HashMap::new();
        for row in &self.grid {
            for &terrain in row {
                *counts.entry(terrain).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Get average food multiplier for the entire grid
    pub fn average_food_multiplier(&self) -> f32 {
        let total: f32 = self
            .grid
            .iter()
            .flatten()
            .map(|t| t.food_multiplier())
            .sum();
        total / (self.grid_size * self.grid_size) as f32
    }
}

/// Terrain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainConfig {
    /// Is terrain variation enabled
    pub enabled: bool,
    /// Use clustered generation (more realistic)
    pub clustered: bool,
    /// Create a barrier for isolation experiments
    pub barrier: bool,
    /// Barrier is vertical (true) or horizontal (false)
    pub barrier_vertical: bool,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            clustered: true,
            barrier: false,
            barrier_vertical: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_costs() {
        assert_eq!(Terrain::Plain.movement_cost(false), 1.0);
        assert_eq!(Terrain::Mountain.movement_cost(false), 2.5);
        assert_eq!(Terrain::Water.movement_cost(false), 10.0);
        assert_eq!(Terrain::Water.movement_cost(true), 0.5); // Aquatic
    }

    #[test]
    fn test_terrain_food() {
        assert!(Terrain::Forest.food_multiplier() > Terrain::Plain.food_multiplier());
        assert!(Terrain::Desert.food_multiplier() < Terrain::Plain.food_multiplier());
    }

    #[test]
    fn test_terrain_grid_creation() {
        let grid = TerrainGrid::new(50);
        assert_eq!(grid.grid_size, 50);
        assert_eq!(grid.get(25, 25), Terrain::Plain);
    }

    #[test]
    fn test_terrain_generation() {
        let mut grid = TerrainGrid::new(50);
        grid.generate_clustered(Some(42));

        // Should have variety
        let counts = grid.terrain_counts();
        assert!(counts.len() >= 3, "Should have at least 3 terrain types");
    }

    #[test]
    fn test_terrain_barrier() {
        let mut grid = TerrainGrid::new(50);
        grid.generate_with_barrier(true);

        // Check middle column is mountains
        let mid = 25;
        let mountain_count: usize = (0..50)
            .filter(|&y| grid.get(mid as u8, y as u8) == Terrain::Mountain)
            .count();

        assert!(mountain_count > 40, "Should have mountain barrier");
    }

    #[test]
    fn test_passability() {
        assert!(Terrain::Plain.is_passable(false));
        assert!(Terrain::Mountain.is_passable(false));
        assert!(!Terrain::Water.is_passable(false));
        assert!(Terrain::Water.is_passable(true)); // Aquatic can pass
    }
}
