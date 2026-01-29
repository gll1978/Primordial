//! Multi-terrain system with movement costs and food modifiers.
//!
//! Supports both clustered random generation and realistic biome-based generation
//! using Simplex noise for elevation and humidity maps.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

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

// ============================================================================
// SIMPLEX NOISE IMPLEMENTATION
// ============================================================================

/// 2D Simplex noise generator for procedural terrain generation
pub struct SimplexNoise {
    perm: [u8; 512],
}

impl SimplexNoise {
    /// Create a new SimplexNoise generator with a seed
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut perm = [0u8; 512];

        // Initialize permutation table
        let mut p: Vec<u8> = (0..=255).collect();

        // Fisher-Yates shuffle
        for i in (1..256).rev() {
            let j = rng.gen_range(0..=i);
            p.swap(i, j);
        }

        // Duplicate for overflow handling
        for i in 0..512 {
            perm[i] = p[i & 255];
        }

        Self { perm }
    }

    /// Sample 2D simplex noise at (x, y)
    /// Returns value in range [-1, 1]
    pub fn sample(&self, x: f64, y: f64) -> f64 {
        const F2: f64 = 0.5 * (1.732050808 - 1.0); // (sqrt(3) - 1) / 2
        const G2: f64 = (3.0 - 1.732050808) / 6.0; // (3 - sqrt(3)) / 6

        // Skew input space to determine which simplex cell we're in
        let s = (x + y) * F2;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;

        let t = (i + j) as f64 * G2;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);

        // Determine which simplex we're in
        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

        let x1 = x0 - i1 as f64 + G2;
        let y1 = y0 - j1 as f64 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        // Hash coordinates of corners
        let ii = (i & 255) as usize;
        let jj = (j & 255) as usize;

        let gi0 = self.perm[ii + self.perm[jj] as usize] as usize % 12;
        let gi1 = self.perm[ii + i1 as usize + self.perm[jj + j1 as usize] as usize] as usize % 12;
        let gi2 = self.perm[ii + 1 + self.perm[jj + 1] as usize] as usize % 12;

        // Calculate contributions from each corner
        let n0 = self.corner_contribution(x0, y0, gi0);
        let n1 = self.corner_contribution(x1, y1, gi1);
        let n2 = self.corner_contribution(x2, y2, gi2);

        // Scale to [-1, 1]
        70.0 * (n0 + n1 + n2)
    }

    fn corner_contribution(&self, x: f64, y: f64, gi: usize) -> f64 {
        // Gradient vectors for 2D
        const GRAD2: [[f64; 2]; 12] = [
            [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
            [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0],
            [0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, -1.0],
        ];

        let t = 0.5 - x * x - y * y;
        if t < 0.0 {
            0.0
        } else {
            let t = t * t;
            t * t * (GRAD2[gi][0] * x + GRAD2[gi][1] * y)
        }
    }

    /// Generate fractal Brownian motion (fBm) for more natural-looking noise
    /// octaves: number of noise layers
    /// persistence: amplitude decay per octave (0.5 typical)
    /// lacunarity: frequency increase per octave (2.0 typical)
    pub fn fbm(&self, x: f64, y: f64, octaves: u32, persistence: f64, lacunarity: f64) -> f64 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..octaves {
            value += amplitude * self.sample(x * frequency, y * frequency);
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        value / max_value
    }
}

// ============================================================================
// TERRAIN GENERATION CONFIGURATION
// ============================================================================

/// Configuration for realistic terrain generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainGenerationConfig {
    /// Scale for elevation noise (smaller = larger features)
    pub elevation_scale: f64,
    /// Scale for humidity noise
    pub humidity_scale: f64,
    /// Threshold for mountains (elevation > this = mountain)
    pub mountain_threshold: f64,
    /// Threshold for water (elevation < this = water candidate)
    pub water_threshold: f64,
    /// Number of procedural mountain ridges to generate
    pub mountain_ridge_count: usize,
    /// Number of lakes to generate
    pub lake_count: usize,
    /// Minimum lake radius
    pub lake_min_radius: usize,
    /// Maximum lake radius
    pub lake_max_radius: usize,
    /// Number of rivers to generate
    pub river_count: usize,
    /// Number of smoothing passes for terrain transitions
    pub smoothing_passes: usize,
    /// Number of octaves for fBm noise
    pub noise_octaves: u32,
    /// Persistence for fBm (amplitude decay)
    pub noise_persistence: f64,
    /// Lacunarity for fBm (frequency increase)
    pub noise_lacunarity: f64,
}

impl Default for TerrainGenerationConfig {
    fn default() -> Self {
        Self {
            elevation_scale: 0.05,
            humidity_scale: 0.04,
            mountain_threshold: 0.7,
            water_threshold: 0.25,
            mountain_ridge_count: 2,
            lake_count: 3,
            lake_min_radius: 2,
            lake_max_radius: 5,
            river_count: 2,
            smoothing_passes: 2,
            noise_octaves: 4,
            noise_persistence: 0.5,
            noise_lacunarity: 2.0,
        }
    }
}

// ============================================================================
// TERRAIN MAPS FOR GENERATION
// ============================================================================

/// Intermediate maps used during terrain generation
struct TerrainMaps {
    elevation: Vec<Vec<f64>>,
    humidity: Vec<Vec<f64>>,
    size: usize,
}

impl TerrainMaps {
    fn new(size: usize) -> Self {
        Self {
            elevation: vec![vec![0.0; size]; size],
            humidity: vec![vec![0.0; size]; size],
            size,
        }
    }

    fn get_elevation(&self, x: usize, y: usize) -> f64 {
        if x < self.size && y < self.size {
            self.elevation[y][x]
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    fn set_elevation(&mut self, x: usize, y: usize, val: f64) {
        if x < self.size && y < self.size {
            self.elevation[y][x] = val;
        }
    }

    #[allow(dead_code)]
    fn get_humidity(&self, x: usize, y: usize) -> f64 {
        if x < self.size && y < self.size {
            self.humidity[y][x]
        } else {
            0.0
        }
    }
}

// ============================================================================
// TERRAIN GRID
// ============================================================================

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

    // ========================================================================
    // REALISTIC TERRAIN GENERATION
    // ========================================================================

    /// Generate realistic terrain using Simplex noise for biomes and features
    pub fn generate_realistic(&mut self, seed: u64, config: &TerrainGenerationConfig) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let elevation_noise = SimplexNoise::new(seed);
        let humidity_noise = SimplexNoise::new(seed.wrapping_add(12345));

        // Step 1: Generate base elevation and humidity maps
        let mut maps = TerrainMaps::new(self.grid_size);
        self.generate_noise_maps(&elevation_noise, &humidity_noise, &mut maps, config);

        // Step 2: Generate mountain ridges as barriers
        self.generate_mountain_ridges(&mut maps, &mut rng, config);

        // Step 3: Assign biomes based on elevation × humidity table
        self.assign_biomes(&maps, config);

        // Step 4: Generate water systems (lakes in depressions, rivers from mountains)
        self.generate_water_systems(&maps, &mut rng, config);

        // Step 5: Post-processing - enforce adjacency rules and smooth transitions
        for _ in 0..config.smoothing_passes {
            self.smooth_terrain();
        }
        self.enforce_adjacency_rules();
    }

    /// Generate elevation and humidity noise maps
    fn generate_noise_maps(
        &self,
        elevation_noise: &SimplexNoise,
        humidity_noise: &SimplexNoise,
        maps: &mut TerrainMaps,
        config: &TerrainGenerationConfig,
    ) {
        for y in 0..self.grid_size {
            for x in 0..self.grid_size {
                // Elevation with fBm for natural variation
                let elev = elevation_noise.fbm(
                    x as f64 * config.elevation_scale,
                    y as f64 * config.elevation_scale,
                    config.noise_octaves,
                    config.noise_persistence,
                    config.noise_lacunarity,
                );
                // Normalize from [-1, 1] to [0, 1]
                maps.elevation[y][x] = (elev + 1.0) / 2.0;

                // Humidity with fBm
                let humid = humidity_noise.fbm(
                    x as f64 * config.humidity_scale,
                    y as f64 * config.humidity_scale,
                    config.noise_octaves,
                    config.noise_persistence,
                    config.noise_lacunarity,
                );
                maps.humidity[y][x] = (humid + 1.0) / 2.0;
            }
        }
    }

    /// Generate curved mountain ridges as natural barriers
    fn generate_mountain_ridges(
        &self,
        maps: &mut TerrainMaps,
        rng: &mut ChaCha8Rng,
        config: &TerrainGenerationConfig,
    ) {
        for _ in 0..config.mountain_ridge_count {
            // Start point on edge
            let (mut x, mut y): (f64, f64);
            let vertical = rng.gen_bool(0.5);

            if vertical {
                x = rng.gen_range(self.grid_size / 4..self.grid_size * 3 / 4) as f64;
                y = 0.0;
            } else {
                x = 0.0;
                y = rng.gen_range(self.grid_size / 4..self.grid_size * 3 / 4) as f64;
            }

            // Direction with slight curve
            let base_angle = if vertical { std::f64::consts::PI / 2.0 } else { 0.0 };
            let mut angle = base_angle + rng.gen_range(-0.3..0.3);
            let curve_rate = rng.gen_range(-0.02..0.02);

            // Draw ridge with varying width
            let ridge_length = self.grid_size as f64 * 1.2;
            let step_count = (ridge_length / 0.5) as usize;

            for _ in 0..step_count {
                let ix = x as usize;
                let iy = y as usize;

                if ix >= self.grid_size || iy >= self.grid_size {
                    break;
                }

                // Ridge width varies
                let width = rng.gen_range(1..=3);
                for dy in 0..width {
                    for dx in 0..width {
                        let px = ix.saturating_add(dx).min(self.grid_size - 1);
                        let py = iy.saturating_add(dy).min(self.grid_size - 1);
                        // Boost elevation to force mountain
                        maps.elevation[py][px] = maps.elevation[py][px].max(0.85);
                    }
                }

                // Move along ridge
                x += angle.cos() * 0.5;
                y += angle.sin() * 0.5;
                angle += curve_rate + rng.gen_range(-0.05..0.05);
            }
        }
    }

    /// Assign biomes based on elevation × humidity table
    fn assign_biomes(&mut self, maps: &TerrainMaps, config: &TerrainGenerationConfig) {
        for y in 0..self.grid_size {
            for x in 0..self.grid_size {
                let elevation = maps.elevation[y][x];
                let humidity = maps.humidity[y][x];

                // Biome assignment table
                self.grid[y][x] = if elevation > config.mountain_threshold {
                    // High elevation = Mountain (regardless of humidity)
                    Terrain::Mountain
                } else if elevation < config.water_threshold {
                    // Low elevation = Water candidate (if humid enough) or desert
                    if humidity > 0.4 {
                        Terrain::Water
                    } else {
                        Terrain::Desert
                    }
                } else {
                    // Mid elevation: depends on humidity
                    if humidity < 0.33 {
                        Terrain::Desert
                    } else if humidity < 0.66 {
                        Terrain::Plain
                    } else {
                        Terrain::Forest
                    }
                };
            }
        }
    }

    /// Generate water systems: lakes in depressions and rivers from mountains
    fn generate_water_systems(
        &mut self,
        maps: &TerrainMaps,
        rng: &mut ChaCha8Rng,
        config: &TerrainGenerationConfig,
    ) {
        // Find depression points (local minima in elevation)
        let mut depressions = self.find_depressions(maps);

        // Sort by elevation (lowest first) and take top candidates
        depressions.sort_by(|a, b| {
            maps.get_elevation(a.0, a.1)
                .partial_cmp(&maps.get_elevation(b.0, b.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create lakes at depressions
        let lake_count = config.lake_count.min(depressions.len());
        for i in 0..lake_count {
            let (cx, cy) = depressions[i];
            let radius = rng.gen_range(config.lake_min_radius..=config.lake_max_radius);
            self.flood_fill_lake(cx, cy, radius, maps);
        }

        // Generate rivers from mountain peaks
        self.generate_rivers(maps, rng, config);
    }

    /// Find local minima in elevation map (depression points)
    fn find_depressions(&self, maps: &TerrainMaps) -> Vec<(usize, usize)> {
        let mut depressions = Vec::new();

        for y in 2..self.grid_size - 2 {
            for x in 2..self.grid_size - 2 {
                let elev = maps.get_elevation(x, y);

                // Check if this is a local minimum (lower than all neighbors)
                let is_minimum = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
                    .iter()
                    .all(|(dx, dy)| {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        maps.get_elevation(nx, ny) >= elev
                    });

                if is_minimum && elev < 0.4 {
                    depressions.push((x, y));
                }
            }
        }

        depressions
    }

    /// Flood fill to create a lake at a depression
    fn flood_fill_lake(&mut self, cx: usize, cy: usize, radius: usize, maps: &TerrainMaps) {
        let base_elev = maps.get_elevation(cx, cy);
        let threshold = base_elev + 0.1; // Fill up to this level

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((cx, cy));
        visited.insert((cx, cy));

        let mut filled = 0;
        let max_fill = radius * radius * 4; // Limit lake size

        while let Some((x, y)) = queue.pop_front() {
            if filled >= max_fill {
                break;
            }

            // Only fill if elevation is below threshold and not mountain
            if maps.get_elevation(x, y) <= threshold && self.grid[y][x] != Terrain::Mountain {
                self.grid[y][x] = Terrain::Water;
                filled += 1;

                // Add neighbors
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;

                    if nx < self.grid_size && ny < self.grid_size && !visited.contains(&(nx, ny)) {
                        visited.insert((nx, ny));
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
    }

    /// Generate rivers flowing from mountains to low areas
    fn generate_rivers(&mut self, maps: &TerrainMaps, rng: &mut ChaCha8Rng, config: &TerrainGenerationConfig) {
        // Find mountain peaks to start rivers
        let mut peaks: Vec<(usize, usize)> = Vec::new();

        for y in 5..self.grid_size - 5 {
            for x in 5..self.grid_size - 5 {
                if self.grid[y][x] == Terrain::Mountain {
                    let elev = maps.get_elevation(x, y);
                    if elev > 0.8 {
                        peaks.push((x, y));
                    }
                }
            }
        }

        if peaks.is_empty() {
            return;
        }

        // Generate rivers
        for _ in 0..config.river_count {
            if peaks.is_empty() {
                break;
            }

            let idx = rng.gen_range(0..peaks.len());
            let (start_x, start_y) = peaks.remove(idx);

            self.trace_river(start_x, start_y, maps);
        }
    }

    /// Trace a river path from start following steepest descent
    fn trace_river(&mut self, start_x: usize, start_y: usize, maps: &TerrainMaps) {
        let mut x = start_x;
        let mut y = start_y;
        let mut visited = HashSet::new();

        for _ in 0..self.grid_size * 2 {
            visited.insert((x, y));

            // Find lowest neighbor
            let mut lowest = (x, y);
            let mut lowest_elev = maps.get_elevation(x, y);

            for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)] {
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;

                if nx < self.grid_size && ny < self.grid_size && !visited.contains(&(nx, ny)) {
                    let elev = maps.get_elevation(nx, ny);
                    if elev < lowest_elev {
                        lowest_elev = elev;
                        lowest = (nx, ny);
                    }
                }
            }

            // Stop if we can't go lower or hit water
            if lowest == (x, y) || self.grid[y][x] == Terrain::Water {
                break;
            }

            // Place water (river tile)
            if self.grid[y][x] != Terrain::Mountain {
                self.grid[y][x] = Terrain::Water;
            }

            x = lowest.0;
            y = lowest.1;
        }
    }

    /// Smooth terrain transitions for more natural appearance
    fn smooth_terrain(&mut self) {
        let old_grid = self.grid.clone();

        for y in 1..self.grid_size - 1 {
            for x in 1..self.grid_size - 1 {
                // Count neighbor terrain types
                let mut counts: HashMap<Terrain, usize> = HashMap::new();

                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        *counts.entry(old_grid[ny][nx]).or_insert(0) += 1;
                    }
                }

                // If current terrain is isolated (only 1-2 of same type), change to majority
                let current = old_grid[y][x];
                let current_count = *counts.get(&current).unwrap_or(&0);

                if current_count <= 2 {
                    // Find majority (excluding water to preserve lakes)
                    if let Some((&majority, _)) = counts
                        .iter()
                        .filter(|(&t, _)| t != Terrain::Water || current == Terrain::Water)
                        .max_by_key(|(_, &count)| count)
                    {
                        if majority != current && *counts.get(&majority).unwrap_or(&0) >= 5 {
                            self.grid[y][x] = majority;
                        }
                    }
                }
            }
        }
    }

    /// Enforce adjacency rules: no desert directly next to water
    fn enforce_adjacency_rules(&mut self) {
        let mut changes = Vec::new();

        for y in 0..self.grid_size {
            for x in 0..self.grid_size {
                if self.grid[y][x] == Terrain::Desert {
                    // Check if adjacent to water
                    let adjacent_to_water = [(-1i32, 0), (1, 0), (0, -1), (0, 1)]
                        .iter()
                        .any(|(dx, dy)| {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && ny >= 0 && (nx as usize) < self.grid_size && (ny as usize) < self.grid_size {
                                self.grid[ny as usize][nx as usize] == Terrain::Water
                            } else {
                                false
                            }
                        });

                    if adjacent_to_water {
                        // Convert desert to plain (transition zone)
                        changes.push((x, y, Terrain::Plain));
                    }
                }
            }
        }

        for (x, y, terrain) in changes {
            self.grid[y][x] = terrain;
        }
    }

    /// Find a valid spawn position for an organism
    /// If is_aquatic is true, finds water; otherwise finds non-water passable terrain
    pub fn find_valid_spawn_position(&self, is_aquatic: bool, rng: &mut impl Rng) -> Option<(u8, u8)> {
        // Collect all valid positions
        let mut valid_positions = Vec::new();

        for y in 0..self.grid_size {
            for x in 0..self.grid_size {
                let terrain = self.grid[y][x];
                let is_valid = if is_aquatic {
                    terrain == Terrain::Water
                } else {
                    terrain.is_passable(false) // Non-water passable terrain
                };

                if is_valid {
                    valid_positions.push((x as u8, y as u8));
                }
            }
        }

        if valid_positions.is_empty() {
            None
        } else {
            Some(valid_positions[rng.gen_range(0..valid_positions.len())])
        }
    }

    /// Find valid spawn position near a specific location
    /// Useful for spawning offspring near parents
    pub fn find_valid_spawn_near(&self, cx: u8, cy: u8, radius: u8, is_aquatic: bool, rng: &mut impl Rng) -> Option<(u8, u8)> {
        let mut valid_positions = Vec::new();

        let min_x = cx.saturating_sub(radius) as usize;
        let max_x = (cx as usize).saturating_add(radius as usize).min(self.grid_size - 1);
        let min_y = cy.saturating_sub(radius) as usize;
        let max_y = (cy as usize).saturating_add(radius as usize).min(self.grid_size - 1);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let terrain = self.grid[y][x];
                let is_valid = if is_aquatic {
                    terrain == Terrain::Water
                } else {
                    terrain.is_passable(false)
                };

                if is_valid {
                    valid_positions.push((x as u8, y as u8));
                }
            }
        }

        if valid_positions.is_empty() {
            // Fall back to global search
            self.find_valid_spawn_position(is_aquatic, rng)
        } else {
            Some(valid_positions[rng.gen_range(0..valid_positions.len())])
        }
    }

    /// Check if position is adjacent to water (for aquatic mutation checks)
    pub fn is_adjacent_to_water(&self, x: u8, y: u8) -> bool {
        let x = x as i32;
        let y = y as i32;

        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let ny = y + dy;

            if nx >= 0 && ny >= 0 && (nx as usize) < self.grid_size && (ny as usize) < self.grid_size {
                if self.grid[ny as usize][nx as usize] == Terrain::Water {
                    return true;
                }
            }
        }

        false
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

// ============================================================================
// TERRAIN CONFIGURATION
// ============================================================================

/// Terrain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainConfig {
    /// Is terrain variation enabled
    pub enabled: bool,
    /// Use clustered generation (more realistic than pure random)
    pub clustered: bool,
    /// Create a barrier for isolation experiments
    pub barrier: bool,
    /// Barrier is vertical (true) or horizontal (false)
    pub barrier_vertical: bool,
    /// Use realistic biome-based generation (overrides clustered if true)
    #[serde(default)]
    pub realistic: bool,
    /// Configuration for realistic generation
    #[serde(default)]
    pub generation: TerrainGenerationConfig,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            clustered: false,     // Ignored when realistic is true
            barrier: false,
            barrier_vertical: true,
            realistic: true,      // Realistic biome generation enabled by default
            generation: TerrainGenerationConfig::default(),
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

    #[test]
    fn test_simplex_noise() {
        let noise = SimplexNoise::new(42);

        // Test that noise is in valid range
        for x in 0..10 {
            for y in 0..10 {
                let val = noise.sample(x as f64 * 0.1, y as f64 * 0.1);
                assert!(val >= -1.0 && val <= 1.0, "Noise value out of range: {}", val);
            }
        }

        // Test determinism
        let val1 = noise.sample(0.5, 0.5);
        let val2 = noise.sample(0.5, 0.5);
        assert_eq!(val1, val2, "Noise should be deterministic");
    }

    #[test]
    fn test_realistic_generation() {
        let mut grid = TerrainGrid::new(80);
        let config = TerrainGenerationConfig::default();
        grid.generate_realistic(42, &config);

        let counts = grid.terrain_counts();

        // Should have all terrain types
        assert!(counts.len() >= 4, "Should have at least 4 terrain types");

        // Should have some of each
        assert!(counts.get(&Terrain::Mountain).unwrap_or(&0) > &0, "Should have mountains");
        assert!(counts.get(&Terrain::Water).unwrap_or(&0) > &0, "Should have water");
        assert!(counts.get(&Terrain::Plain).unwrap_or(&0) > &0, "Should have plains");
    }

    #[test]
    fn test_adjacency_rules() {
        let mut grid = TerrainGrid::new(80);
        let config = TerrainGenerationConfig::default();
        grid.generate_realistic(42, &config);

        // Check no desert directly adjacent to water
        for y in 0..grid.grid_size {
            for x in 0..grid.grid_size {
                if grid.grid[y][x] == Terrain::Desert {
                    for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && ny >= 0 && (nx as usize) < grid.grid_size && (ny as usize) < grid.grid_size {
                            assert_ne!(
                                grid.grid[ny as usize][nx as usize],
                                Terrain::Water,
                                "Desert at ({}, {}) should not be adjacent to water at ({}, {})",
                                x, y, nx, ny
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_find_valid_spawn() {
        let mut grid = TerrainGrid::new(50);
        let config = TerrainGenerationConfig::default();
        grid.generate_realistic(42, &config);

        let mut rng = ChaCha8Rng::seed_from_u64(123);

        // Should find land position for non-aquatic
        let land_pos = grid.find_valid_spawn_position(false, &mut rng);
        assert!(land_pos.is_some(), "Should find land spawn position");
        if let Some((x, y)) = land_pos {
            assert_ne!(grid.get(x, y), Terrain::Water, "Non-aquatic should not spawn in water");
        }

        // Should find water position for aquatic
        let water_pos = grid.find_valid_spawn_position(true, &mut rng);
        assert!(water_pos.is_some(), "Should find water spawn position");
        if let Some((x, y)) = water_pos {
            assert_eq!(grid.get(x, y), Terrain::Water, "Aquatic should spawn in water");
        }
    }

    #[test]
    fn test_is_adjacent_to_water() {
        let mut grid = TerrainGrid::new(10);
        // Create a small lake
        grid.set(5, 5, Terrain::Water);

        // Adjacent cells should return true
        assert!(grid.is_adjacent_to_water(4, 5));
        assert!(grid.is_adjacent_to_water(6, 5));
        assert!(grid.is_adjacent_to_water(5, 4));
        assert!(grid.is_adjacent_to_water(5, 6));

        // Diagonal and far cells should return false
        assert!(!grid.is_adjacent_to_water(4, 4));
        assert!(!grid.is_adjacent_to_water(0, 0));
    }
}
