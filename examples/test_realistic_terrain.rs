//! Test example for realistic terrain generation.
//!
//! Run with: cargo run --example test_realistic_terrain

use primordial::config::Config;
use primordial::ecology::{TerrainConfig, TerrainGenerationConfig, Terrain};
use primordial::world::World;

fn main() {
    println!("=== Test Realistic Terrain Generation ===\n");

    // Create config with realistic terrain enabled
    let mut config = Config::default();
    config.terrain = TerrainConfig {
        enabled: true,
        clustered: false,
        barrier: false,
        barrier_vertical: false,
        realistic: true,
        generation: TerrainGenerationConfig {
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
        },
    };
    config.world.grid_size = 60;
    config.organisms.initial_population = 50;

    // Create world
    println!("Creating world with realistic terrain...");
    let world = World::new_with_seed(config.clone(), 42);

    // Analyze terrain
    let counts = world.terrain_grid.terrain_counts();
    println!("\nTerrain distribution:");
    for (terrain, count) in &counts {
        let percent = (*count as f32 / (config.world.grid_size * config.world.grid_size) as f32) * 100.0;
        println!("  {:?}: {} ({:.1}%)", terrain, count, percent);
    }

    // Verify organisms are not in water
    println!("\nOrganism spawn verification:");
    let mut in_water = 0;
    let mut in_valid = 0;
    for org in &world.organisms {
        let terrain = world.terrain_grid.get(org.x, org.y);
        if terrain == Terrain::Water {
            if org.is_aquatic {
                in_valid += 1; // Aquatic in water is OK
            } else {
                in_water += 1;
                println!("  WARNING: Non-aquatic organism {} at ({}, {}) in water!", org.id, org.x, org.y);
            }
        } else {
            in_valid += 1;
        }
    }
    println!("  Valid spawns: {}", in_valid);
    println!("  Invalid spawns (terrestrial in water): {}", in_water);

    // Check adjacency rules
    println!("\nAdjacency rules verification:");
    let mut violations = 0;
    for y in 0..config.world.grid_size {
        for x in 0..config.world.grid_size {
            if world.terrain_grid.grid[y][x] == Terrain::Desert {
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < config.world.grid_size && (ny as usize) < config.world.grid_size {
                        if world.terrain_grid.grid[ny as usize][nx as usize] == Terrain::Water {
                            violations += 1;
                        }
                    }
                }
            }
        }
    }
    println!("  Desert-Water adjacency violations: {}", violations);

    // Print terrain map (small section)
    println!("\nTerrain map (top-left 30x15):");
    for y in 0..15.min(config.world.grid_size) {
        let mut row = String::new();
        for x in 0..30.min(config.world.grid_size) {
            let terrain = world.terrain_grid.grid[y][x];
            let ch = match terrain {
                Terrain::Plain => '.',
                Terrain::Forest => 'F',
                Terrain::Mountain => '^',
                Terrain::Desert => '~',
                Terrain::Water => 'W',
            };
            row.push(ch);
        }
        println!("  {}", row);
    }

    // Run simulation for a few steps
    println!("\nRunning simulation for 100 steps...");
    let mut world = world;
    for _ in 0..100 {
        world.step();
    }

    // Check organism positions after simulation
    let alive_count = world.organisms.iter().filter(|o| o.is_alive()).count();
    let mut in_water_after = 0;
    for org in world.organisms.iter().filter(|o| o.is_alive()) {
        let terrain = world.terrain_grid.get(org.x, org.y);
        if terrain == Terrain::Water && !org.is_aquatic {
            in_water_after += 1;
        }
    }

    println!("After 100 steps:");
    println!("  Alive organisms: {}", alive_count);
    println!("  Non-aquatic in water: {}", in_water_after);

    println!("\n=== Test Complete ===");
}
