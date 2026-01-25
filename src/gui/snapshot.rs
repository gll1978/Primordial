//! Snapshot structures for GUI communication.
//!
//! These are lightweight copies of simulation state, optimized for
//! fast transfer between simulation and render threads.

use crate::ecology::terrain::Terrain;
use crate::stats::Stats;

/// Lightweight view of an organism for rendering
#[derive(Clone, Debug)]
pub struct OrganismView {
    pub id: u64,
    pub x: u8,
    pub y: u8,
    pub energy: f32,
    pub is_predator: bool,
    pub is_aquatic: bool,
    pub generation: u16,
    pub size: f32,
}

/// Detailed organism info for the selected organism panel
#[derive(Clone, Debug)]
pub struct OrganismDetail {
    pub id: u64,
    pub x: u8,
    pub y: u8,
    pub energy: f32,
    pub health: f32,
    pub age: u32,
    pub generation: u16,
    pub lineage_id: u32,
    pub is_predator: bool,
    pub is_aquatic: bool,
    pub kills: u16,
    pub offspring_count: u16,
    pub food_eaten: u32,
    pub size: f32,
    pub memory: [f32; 5],
    pub brain_layers: Vec<LayerView>,
}

/// Simplified layer representation for brain visualization
#[derive(Clone, Debug)]
pub struct LayerView {
    /// Input dimension
    pub inputs: usize,
    /// Output dimension
    pub outputs: usize,
    /// Flattened weights (row-major)
    pub weights: Vec<f32>,
    /// Biases
    pub biases: Vec<f32>,
}

/// Complete world snapshot for rendering
#[derive(Clone, Debug)]
pub struct WorldSnapshot {
    /// Current simulation time
    pub time: u64,
    /// Statistics
    pub stats: Stats,
    /// All organisms (lightweight view)
    pub organisms: Vec<OrganismView>,
    /// Flattened food grid (row-major, size x size)
    pub food_grid: Vec<f32>,
    /// Flattened terrain grid (row-major, encoded as u8)
    pub terrain_grid: Vec<u8>,
    /// Grid dimension
    pub grid_size: usize,
    /// Currently selected organism (if any)
    pub selected_organism: Option<OrganismDetail>,
}

impl WorldSnapshot {
    /// Create a snapshot from the current world state
    pub fn from_world(world: &crate::World, selected_id: Option<u64>) -> Self {
        let grid_size = world.config.world.grid_size;

        // Convert organisms to lightweight views
        let organisms: Vec<OrganismView> = world
            .organisms
            .iter()
            .filter(|o| o.is_alive())
            .map(|o| OrganismView {
                id: o.id,
                x: o.x,
                y: o.y,
                energy: o.energy,
                is_predator: o.is_predator,
                is_aquatic: o.is_aquatic,
                generation: o.generation,
                size: o.size,
            })
            .collect();

        // Flatten food grid
        let mut food_grid = Vec::with_capacity(grid_size * grid_size);
        for y in 0..grid_size {
            for x in 0..grid_size {
                food_grid.push(world.food_grid.get(x as u8, y as u8));
            }
        }

        // Flatten terrain grid
        let mut terrain_grid = Vec::with_capacity(grid_size * grid_size);
        for y in 0..grid_size {
            for x in 0..grid_size {
                let terrain = world.terrain_grid.get(x as u8, y as u8);
                terrain_grid.push(terrain_to_u8(terrain));
            }
        }

        // Get selected organism details if requested
        let selected_organism = selected_id.and_then(|id| {
            world
                .organisms
                .iter()
                .find(|o| o.id == id && o.is_alive())
                .map(|o| OrganismDetail {
                    id: o.id,
                    x: o.x,
                    y: o.y,
                    energy: o.energy,
                    health: o.health,
                    age: o.age,
                    generation: o.generation,
                    lineage_id: o.lineage_id,
                    is_predator: o.is_predator,
                    is_aquatic: o.is_aquatic,
                    kills: o.kills,
                    offspring_count: o.offspring_count,
                    food_eaten: o.food_eaten,
                    size: o.size,
                    memory: o.memory,
                    brain_layers: o
                        .brain
                        .layers
                        .iter()
                        .map(|layer| {
                            let shape = layer.weights.shape();
                            LayerView {
                                inputs: shape[0],
                                outputs: shape[1],
                                weights: layer.weights.iter().copied().collect(),
                                biases: layer.biases.iter().copied().collect(),
                            }
                        })
                        .collect(),
                })
        });

        Self {
            time: world.time,
            stats: world.stats.clone(),
            organisms,
            food_grid,
            terrain_grid,
            grid_size,
            selected_organism,
        }
    }
}

/// Convert terrain enum to u8 for compact storage
#[inline]
pub fn terrain_to_u8(terrain: Terrain) -> u8 {
    match terrain {
        Terrain::Plain => 0,
        Terrain::Forest => 1,
        Terrain::Mountain => 2,
        Terrain::Desert => 3,
        Terrain::Water => 4,
    }
}

/// Convert u8 back to terrain enum
#[inline]
pub fn u8_to_terrain(value: u8) -> Terrain {
    match value {
        0 => Terrain::Plain,
        1 => Terrain::Forest,
        2 => Terrain::Mountain,
        3 => Terrain::Desert,
        4 => Terrain::Water,
        _ => Terrain::Plain,
    }
}

/// Get terrain color as RGB tuple for rendering
#[inline]
pub fn terrain_color(terrain: Terrain) -> (u8, u8, u8) {
    match terrain {
        Terrain::Plain => (144, 238, 144),   // Light green
        Terrain::Forest => (34, 139, 34),     // Forest green
        Terrain::Mountain => (139, 137, 137), // Gray
        Terrain::Desert => (238, 213, 145),   // Sandy
        Terrain::Water => (65, 105, 225),     // Royal blue
    }
}
