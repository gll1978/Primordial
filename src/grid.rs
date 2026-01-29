//! Spatial grid and indexing for fast neighbor queries.

use serde::{Deserialize, Serialize};

/// Spatial index for fast organism lookups by position
#[derive(Clone, Debug)]
pub struct SpatialIndex {
    grid_size: usize,
    /// cells[y][x] contains indices of organisms at that position
    cells: Vec<Vec<Vec<usize>>>,
}

impl SpatialIndex {
    /// Create a new spatial index for the given grid size
    pub fn new(grid_size: usize) -> Self {
        Self {
            grid_size,
            cells: vec![vec![Vec::new(); grid_size]; grid_size],
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        for row in &mut self.cells {
            for cell in row {
                cell.clear();
            }
        }
    }

    /// Insert an organism index at the given position
    #[inline]
    pub fn insert(&mut self, x: u8, y: u8, org_idx: usize) {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.cells[y][x].push(org_idx);
        }
    }

    /// Get all organism indices at a specific cell
    #[inline]
    pub fn get(&self, x: u8, y: u8) -> &[usize] {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            &self.cells[y][x]
        } else {
            &[]
        }
    }

    /// Query all organisms within a radius (Manhattan distance)
    pub fn query_radius(&self, x: u8, y: u8, radius: u8) -> Vec<usize> {
        let mut results = Vec::new();

        let x_min = x.saturating_sub(radius) as usize;
        let x_max = (x as usize).saturating_add(radius as usize).min(self.grid_size - 1);
        let y_min = y.saturating_sub(radius) as usize;
        let y_max = (y as usize).saturating_add(radius as usize).min(self.grid_size - 1);

        for dy in y_min..=y_max {
            for dx in x_min..=x_max {
                results.extend_from_slice(&self.cells[dy][dx]);
            }
        }

        results
    }

    /// Query all organisms within a radius, excluding the center cell
    pub fn query_neighbors(&self, x: u8, y: u8, radius: u8) -> Vec<usize> {
        let mut results = Vec::new();

        let x_min = x.saturating_sub(radius) as usize;
        let x_max = (x as usize).saturating_add(radius as usize).min(self.grid_size - 1);
        let y_min = y.saturating_sub(radius) as usize;
        let y_max = (y as usize).saturating_add(radius as usize).min(self.grid_size - 1);
        let center_x = x as usize;
        let center_y = y as usize;

        for dy in y_min..=y_max {
            for dx in x_min..=x_max {
                if dx != center_x || dy != center_y {
                    results.extend_from_slice(&self.cells[dy][dx]);
                }
            }
        }

        results
    }

    /// Check if a cell is occupied
    #[inline]
    pub fn is_occupied(&self, x: u8, y: u8) -> bool {
        !self.get(x, y).is_empty()
    }

    /// Count organisms in a cell
    #[inline]
    pub fn count_at(&self, x: u8, y: u8) -> usize {
        self.get(x, y).len()
    }
}

/// Food grid for the environment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoodGrid {
    grid_size: usize,
    cells: Vec<Vec<f32>>,
    max_food: f32,
}

impl FoodGrid {
    /// Create a new food grid
    pub fn new(grid_size: usize, max_food: f32) -> Self {
        Self {
            grid_size,
            cells: vec![vec![0.0; grid_size]; grid_size],
            max_food,
        }
    }

    /// Initialize with random food distribution
    pub fn initialize(&mut self, density: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for row in &mut self.cells {
            for cell in row {
                if rng.gen::<f32>() < density {
                    *cell = rng.gen_range(0.0..self.max_food);
                }
            }
        }
    }

    /// Get food amount at position
    #[inline]
    pub fn get(&self, x: u8, y: u8) -> f32 {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.cells[y][x]
        } else {
            0.0
        }
    }

    /// Set food amount at position
    #[inline]
    pub fn set(&mut self, x: u8, y: u8, amount: f32) {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.cells[y][x] = amount.clamp(0.0, self.max_food);
        }
    }

    /// Consume food at position, returns amount consumed
    #[inline]
    pub fn consume(&mut self, x: u8, y: u8, max_amount: f32) -> f32 {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            let available = self.cells[y][x];
            let consumed = available.min(max_amount);
            self.cells[y][x] -= consumed;
            consumed
        } else {
            0.0
        }
    }

    /// Regenerate food across the grid
    pub fn regenerate(&mut self, rate: f32) {
        for row in &mut self.cells {
            for cell in row {
                *cell = (*cell + rate).min(self.max_food);
            }
        }
    }

    /// Spawn random food
    pub fn spawn_random(&mut self, amount: f32, probability: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for row in &mut self.cells {
            for cell in row {
                if rng.gen::<f32>() < probability {
                    *cell = (*cell + amount).min(self.max_food);
                }
            }
        }
    }

    /// Sense food in a direction (sum of food in cells in that direction)
    pub fn sense_direction(&self, x: u8, y: u8, dx: i8, dy: i8, range: u8) -> f32 {
        let mut total = 0.0;
        let mut cx = x as i16;
        let mut cy = y as i16;

        for _ in 0..range {
            cx += dx as i16;
            cy += dy as i16;

            if cx >= 0 && cx < self.grid_size as i16 && cy >= 0 && cy < self.grid_size as i16 {
                total += self.cells[cy as usize][cx as usize];
            }
        }

        total
    }

    /// Get total food in the grid
    pub fn total_food(&self) -> f32 {
        self.cells.iter().flatten().sum()
    }

    /// Get grid size
    #[inline]
    pub fn size(&self) -> usize {
        self.grid_size
    }
}

// ============================================================================
// COGNITIVE GATE SYSTEM - Food Complexity
// ============================================================================

/// Food complexity tier - determines brain requirement for eating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoodTier {
    /// Simple food: accessible to 0-2 layer brains (complexity 0.0-0.3)
    Simple,
    /// Medium food: requires 3-6 layer brains (complexity 0.3-0.7)
    Medium,
    /// Complex food: requires 7+ layer brains (complexity 0.7-1.0)
    Complex,
}

impl FoodTier {
    /// Get energy value for this food tier (default values)
    pub fn energy(&self) -> f32 {
        match self {
            FoodTier::Simple => 23.0,
            FoodTier::Medium => 27.0,
            FoodTier::Complex => 32.0,
        }
    }

    /// Get energy with configurable values
    pub fn energy_with_config(&self, simple: f32, medium: f32, complex: f32) -> f32 {
        match self {
            FoodTier::Simple => simple,
            FoodTier::Medium => medium,
            FoodTier::Complex => complex,
        }
    }

    /// Get complexity range for this tier (default values with overlap)
    pub fn complexity_range(&self) -> (f32, f32) {
        match self {
            FoodTier::Simple => (0.0, 0.25),
            FoodTier::Medium => (0.20, 0.60),
            FoodTier::Complex => (0.55, 1.0),
        }
    }

    /// Generate a random complexity value within configurable range
    pub fn random_complexity_with_range(&self, min: f32, max: f32) -> f32 {
        use rand::Rng;
        rand::thread_rng().gen_range(min..max)
    }

    /// Generate a random complexity value within this tier's default range
    pub fn random_complexity(&self) -> f32 {
        use rand::Rng;
        let (min, max) = self.complexity_range();
        rand::thread_rng().gen_range(min..max)
    }

    /// Get tier from complexity value (using configurable thresholds)
    pub fn from_complexity_with_thresholds(complexity: f32, simple_max: f32, medium_max: f32) -> Self {
        if complexity < simple_max {
            FoodTier::Simple
        } else if complexity < medium_max {
            FoodTier::Medium
        } else {
            FoodTier::Complex
        }
    }

    /// Get tier from complexity value (default thresholds)
    pub fn from_complexity(complexity: f32) -> Self {
        Self::from_complexity_with_thresholds(complexity, 0.25, 0.60)
    }

    /// Roll for a random food tier based on distribution (40/40/20)
    pub fn random_tier() -> Self {
        use rand::Rng;
        let roll: f32 = rand::thread_rng().gen();
        if roll < 0.40 {
            FoodTier::Simple
        } else if roll < 0.80 {
            FoodTier::Medium
        } else {
            FoodTier::Complex
        }
    }
}

/// Grid storing food complexity values (parallel to FoodGrid)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplexityGrid {
    grid_size: usize,
    /// Complexity values 0.0-1.0 per cell
    cells: Vec<Vec<f32>>,
}

impl ComplexityGrid {
    /// Create a new complexity grid initialized to 0 (simple food)
    pub fn new(grid_size: usize) -> Self {
        Self {
            grid_size,
            cells: vec![vec![0.0; grid_size]; grid_size],
        }
    }

    /// Get complexity at position
    #[inline]
    pub fn get(&self, x: u8, y: u8) -> f32 {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.cells[y][x]
        } else {
            0.0
        }
    }

    /// Get food tier at position
    #[inline]
    pub fn get_tier(&self, x: u8, y: u8) -> FoodTier {
        FoodTier::from_complexity(self.get(x, y))
    }

    /// Set complexity at position
    #[inline]
    pub fn set(&mut self, x: u8, y: u8, complexity: f32) {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.cells[y][x] = complexity.clamp(0.0, 1.0);
        }
    }

    /// Set food tier at position (generates random complexity within tier range)
    #[inline]
    pub fn set_tier(&mut self, x: u8, y: u8, tier: FoodTier) {
        self.set(x, y, tier.random_complexity());
    }

    /// Clear complexity when food is consumed
    #[inline]
    pub fn clear(&mut self, x: u8, y: u8) {
        let x = x as usize;
        let y = y as usize;
        if x < self.grid_size && y < self.grid_size {
            self.cells[y][x] = 0.0;
        }
    }

    /// Initialize with random complexity distribution matching 40/40/20
    pub fn initialize(&mut self, food_grid: &FoodGrid) {
        for y in 0..self.grid_size {
            for x in 0..self.grid_size {
                if food_grid.get(x as u8, y as u8) > 0.1 {
                    let tier = FoodTier::random_tier();
                    self.cells[y][x] = tier.random_complexity();
                } else {
                    self.cells[y][x] = 0.0;
                }
            }
        }
    }

    /// Get grid size
    #[inline]
    pub fn size(&self) -> usize {
        self.grid_size
    }

    /// Count food by tier
    pub fn count_by_tier(&self, food_grid: &FoodGrid) -> (usize, usize, usize) {
        let mut simple = 0;
        let mut medium = 0;
        let mut complex = 0;

        for y in 0..self.grid_size {
            for x in 0..self.grid_size {
                if food_grid.get(x as u8, y as u8) > 0.1 {
                    match self.get_tier(x as u8, y as u8) {
                        FoodTier::Simple => simple += 1,
                        FoodTier::Medium => medium += 1,
                        FoodTier::Complex => complex += 1,
                    }
                }
            }
        }

        (simple, medium, complex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_index_basic() {
        let mut index = SpatialIndex::new(80);
        index.insert(10, 20, 0);
        index.insert(10, 20, 1);
        index.insert(11, 20, 2);

        assert_eq!(index.get(10, 20).len(), 2);
        assert_eq!(index.get(11, 20).len(), 1);
        assert_eq!(index.get(12, 20).len(), 0);
    }

    #[test]
    fn test_spatial_query_radius() {
        let mut index = SpatialIndex::new(80);
        index.insert(10, 10, 0);
        index.insert(11, 10, 1);
        index.insert(10, 11, 2);
        index.insert(20, 20, 3); // Far away

        let results = index.query_radius(10, 10, 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(!results.contains(&3));
    }

    #[test]
    fn test_food_grid() {
        let mut grid = FoodGrid::new(80, 50.0);
        grid.set(10, 10, 25.0);

        assert_eq!(grid.get(10, 10), 25.0);

        let consumed = grid.consume(10, 10, 10.0);
        assert_eq!(consumed, 10.0);
        assert_eq!(grid.get(10, 10), 15.0);
    }

    #[test]
    fn test_food_sense_direction() {
        let mut grid = FoodGrid::new(80, 50.0);
        grid.set(10, 10, 10.0);
        grid.set(11, 10, 20.0);
        grid.set(12, 10, 15.0);

        // Sense east from (9, 10)
        let sensed = grid.sense_direction(9, 10, 1, 0, 3);
        assert_eq!(sensed, 45.0); // 10 + 20 + 15
    }

    #[test]
    fn test_food_regeneration() {
        let mut grid = FoodGrid::new(80, 50.0);
        grid.set(10, 10, 20.0);
        grid.regenerate(5.0);

        assert_eq!(grid.get(10, 10), 25.0);
    }
}
