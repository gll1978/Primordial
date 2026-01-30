//! Benchmark/headless simulation runner with database support

use primordial::config::Config;
use primordial::World;
use std::env;
use std::time::Instant;

#[cfg(feature = "database")]
use primordial::database::Database;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = env::args().collect();

    let grid_size: usize = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);

    let max_steps: u64 = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    log::info!("=== PRIMORDIAL Benchmark ===");
    log::info!("Grid size: {}", grid_size);
    log::info!("Max steps: {}", max_steps);

    // Load and modify config
    let mut config = Config::from_file("config.yaml").unwrap_or_default();
    config.world.grid_size = grid_size;

    // Scale population proportionally to grid area
    let base_grid = 80.0;
    let scale_factor = (grid_size as f64 * grid_size as f64) / (base_grid * base_grid);
    config.organisms.initial_population = (500.0 * scale_factor) as usize;
    config.safety.max_population = (900.0 * scale_factor) as usize;

    log::info!("Initial population: {}", config.organisms.initial_population);
    log::info!("Max population: {}", config.safety.max_population);

    // Create world
    let mut world = World::new(config.clone());
    log::info!("World created with {} organisms", world.organisms.len());

    // Setup database if enabled
    #[cfg(feature = "database")]
    let _db = if config.database.enabled {
        let config_json = serde_json::to_string(&config).unwrap_or_default();
        match Database::new(&config.database.url, &config_json, None) {
            Ok(db) => {
                log::info!("Database connected: run_id = {}", db.run_id);
                world.set_db_sender(db.sender_clone());
                Some(db)
            }
            Err(e) => {
                log::error!("Database connection failed: {}", e);
                None
            }
        }
    } else {
        log::info!("Database disabled in config");
        None
    };

    // Run simulation
    let start = Instant::now();
    let mut last_report = Instant::now();
    let report_interval = std::time::Duration::from_secs(10);

    while world.time < max_steps && !world.is_extinct() {
        world.step();

        // Progress report every 10 seconds
        if last_report.elapsed() >= report_interval {
            let elapsed = start.elapsed().as_secs_f64();
            let steps_per_sec = world.time as f64 / elapsed;
            let eta_secs = (max_steps - world.time) as f64 / steps_per_sec;

            log::info!(
                "Step {}/{} ({:.1}%) - Pop: {} - Gen: {} - {:.0} steps/s - ETA: {:.0}s",
                world.time,
                max_steps,
                (world.time as f64 / max_steps as f64) * 100.0,
                world.organisms.len(),
                world.stats.generation_max,
                steps_per_sec,
                eta_secs
            );
            last_report = Instant::now();
        }
    }

    let elapsed = start.elapsed();
    log::info!("=== Simulation Complete ===");
    log::info!("Total steps: {}", world.time);
    log::info!("Final population: {}", world.organisms.len());
    log::info!("Max generation: {}", world.stats.generation_max);
    log::info!("Elapsed time: {:.2}s", elapsed.as_secs_f64());
    log::info!("Average speed: {:.0} steps/s", world.time as f64 / elapsed.as_secs_f64());

    if world.is_extinct() {
        log::warn!("Population went extinct at step {}", world.time);
    }

    // Database will be flushed when _db is dropped
    #[cfg(feature = "database")]
    if let Some(db) = _db {
        log::info!("Flushing database...");
        drop(db);
        log::info!("Database flushed");
    }
}
