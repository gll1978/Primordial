//! Database module for tracking individual organisms and simulation events.
//!
//! Uses a channel-based architecture: the main simulation thread sends events
//! via `mpsc::channel` to a dedicated DB writer thread running a tokio runtime.

use std::sync::mpsc;
use std::thread;

use chrono::Utc;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use uuid::Uuid;

/// Events sent from the simulation to the DB writer thread
#[derive(Debug)]
pub enum DbEvent {
    OrganismBirth {
        organism_id: u64,
        lineage_id: u32,
        generation: u16,
        parent1_id: Option<u64>,
        parent2_id: Option<u64>,
        step: u64,
        x: u8,
        y: u8,
        brain_layers: i32,
        brain_neurons: i32,
        brain_connections: i32,
        is_predator: bool,
    },
    OrganismDeath {
        organism_id: u64,
        step: u64,
        x: u8,
        y: u8,
        death_cause: String,
        food_eaten: u32,
        kills: u16,
        offspring: u16,
        max_energy: f32,
    },
    OrganismSnapshot {
        step: u64,
        organism_id: u64,
        x: u8,
        y: u8,
        energy: f32,
        health: f32,
        age: u32,
        size: f32,
        kills: u16,
        offspring: u16,
        food_eaten: u32,
        brain_layers: i32,
        brain_neurons: i32,
        brain_connections: i32,
        is_predator: bool,
        last_action: Option<String>,
    },
    ReproductionEvent {
        step: u64,
        parent1_id: u64,
        parent2_id: Option<u64>,
        child_id: u64,
        child_lineage_id: u32,
        child_generation: u16,
        mutation_count: i32,
        parent1_energy: f32,
        child_energy: f32,
    },
    WorldSnapshot {
        step: u64,
        population: i32,
        births: i32,
        deaths: i32,
        kills: i32,
        max_generation: i16,
        avg_energy: f32,
        avg_age: f32,
        avg_brain_layers: f32,
        avg_brain_neurons: f32,
        avg_brain_connections: f32,
        predator_count: i32,
        lineage_count: i32,
        total_food: f32,
    },
    LearningEvent {
        step: u64,
        organism_id: u64,
        reward: f32,
        total_lifetime_reward: f32,
        successful_forages: u32,
        failed_forages: u32,
        weight_updates_count: i32,
    },
    /// Finalize: update run record and shut down
    Shutdown {
        total_steps: u64,
        final_population: i32,
        max_generation: i32,
    },
}

/// Database handle for the simulation.
/// Sends events to a background writer thread via channel.
pub struct Database {
    sender: mpsc::Sender<DbEvent>,
    writer_handle: Option<thread::JoinHandle<()>>,
    pub run_id: Uuid,
}

impl Database {
    /// Connect to the database and start the writer thread.
    /// This blocks briefly to create the pool and insert the run record.
    pub fn new(database_url: &str, config_json: &str, seed: Option<u64>) -> Result<Self, String> {
        let url = database_url.to_string();
        let config_json = config_json.to_string();

        // Create a oneshot-style channel for the run_id
        let (run_id_tx, run_id_rx) = mpsc::channel::<Result<Uuid, String>>();
        let (sender, receiver) = mpsc::channel::<DbEvent>();

        let writer_handle = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime for DB writer");

            rt.block_on(async move {
                // Connect
                let pool = match PgPoolOptions::new()
                    .max_connections(5)
                    .connect(&url)
                    .await
                {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = run_id_tx.send(Err(format!("DB connection failed: {}", e)));
                        return;
                    }
                };

                // Create run record
                let run_id = Uuid::new_v4();
                let config_value: serde_json::Value =
                    serde_json::from_str(&config_json).unwrap_or(serde_json::Value::Null);

                if let Err(e) = sqlx::query(
                    "INSERT INTO simulation_runs (run_id, started_at, config, seed) VALUES ($1, $2, $3, $4)"
                )
                .bind(run_id)
                .bind(Utc::now())
                .bind(&config_value)
                .bind(seed.map(|s| s as i64))
                .execute(&pool)
                .await
                {
                    let _ = run_id_tx.send(Err(format!("Failed to create run: {}", e)));
                    return;
                }

                let _ = run_id_tx.send(Ok(run_id));

                // Process events
                Self::event_loop(pool, run_id, receiver).await;
            });
        });

        // Wait for run_id from the writer thread
        let run_id = run_id_rx
            .recv()
            .map_err(|e| format!("Writer thread died: {}", e))?
            .map_err(|e| e)?;

        Ok(Self {
            sender,
            writer_handle: Some(writer_handle),
            run_id,
        })
    }

    async fn event_loop(pool: PgPool, run_id: Uuid, receiver: mpsc::Receiver<DbEvent>) {
        // Batch buffer
        let mut birth_batch: Vec<DbEvent> = Vec::with_capacity(256);
        let mut death_batch: Vec<DbEvent> = Vec::with_capacity(256);
        let mut snapshot_batch: Vec<DbEvent> = Vec::with_capacity(1024);
        let mut repro_batch: Vec<DbEvent> = Vec::with_capacity(128);
        let mut learning_batch: Vec<DbEvent> = Vec::with_capacity(256);

        loop {
            // Drain all available events
            match receiver.recv() {
                Ok(event) => {
                    match event {
                        DbEvent::Shutdown { total_steps, final_population, max_generation } => {
                            // Flush remaining batches
                            Self::flush_births(&pool, run_id, &mut birth_batch).await;
                            Self::flush_deaths(&pool, run_id, &mut death_batch).await;
                            Self::flush_snapshots(&pool, run_id, &mut snapshot_batch).await;
                            Self::flush_reproductions(&pool, run_id, &mut repro_batch).await;
                            Self::flush_learning(&pool, run_id, &mut learning_batch).await;

                            // Update run record
                            let _ = sqlx::query(
                                "UPDATE simulation_runs SET ended_at = $1, total_steps = $2, final_population = $3, max_generation = $4 WHERE run_id = $5"
                            )
                            .bind(Utc::now())
                            .bind(total_steps as i64)
                            .bind(final_population)
                            .bind(max_generation)
                            .bind(run_id)
                            .execute(&pool)
                            .await;

                            break;
                        }
                        DbEvent::OrganismBirth { .. } => birth_batch.push(event),
                        DbEvent::OrganismDeath { .. } => death_batch.push(event),
                        DbEvent::OrganismSnapshot { .. } => snapshot_batch.push(event),
                        DbEvent::ReproductionEvent { .. } => repro_batch.push(event),
                        DbEvent::WorldSnapshot { .. } => {
                            // World snapshots are infrequent, write immediately
                            Self::write_world_snapshot(&pool, run_id, event).await;
                        }
                        DbEvent::LearningEvent { .. } => learning_batch.push(event),
                    }

                    // Drain remaining buffered events without blocking
                    while let Ok(event) = receiver.try_recv() {
                        match event {
                            DbEvent::Shutdown { total_steps, final_population, max_generation } => {
                                Self::flush_births(&pool, run_id, &mut birth_batch).await;
                                Self::flush_deaths(&pool, run_id, &mut death_batch).await;
                                Self::flush_snapshots(&pool, run_id, &mut snapshot_batch).await;
                                Self::flush_reproductions(&pool, run_id, &mut repro_batch).await;
                                Self::flush_learning(&pool, run_id, &mut learning_batch).await;
                                let _ = sqlx::query(
                                    "UPDATE simulation_runs SET ended_at = $1, total_steps = $2, final_population = $3, max_generation = $4 WHERE run_id = $5"
                                )
                                .bind(Utc::now())
                                .bind(total_steps as i64)
                                .bind(final_population)
                                .bind(max_generation)
                                .bind(run_id)
                                .execute(&pool)
                                .await;
                                return;
                            }
                            DbEvent::OrganismBirth { .. } => birth_batch.push(event),
                            DbEvent::OrganismDeath { .. } => death_batch.push(event),
                            DbEvent::OrganismSnapshot { .. } => snapshot_batch.push(event),
                            DbEvent::ReproductionEvent { .. } => repro_batch.push(event),
                            DbEvent::WorldSnapshot { .. } => {
                                Self::write_world_snapshot(&pool, run_id, event).await;
                            }
                            DbEvent::LearningEvent { .. } => learning_batch.push(event),
                        }
                    }

                    // Flush batches when they reach threshold
                    if birth_batch.len() >= 200 {
                        Self::flush_births(&pool, run_id, &mut birth_batch).await;
                    }
                    if death_batch.len() >= 200 {
                        Self::flush_deaths(&pool, run_id, &mut death_batch).await;
                    }
                    if snapshot_batch.len() >= 500 {
                        Self::flush_snapshots(&pool, run_id, &mut snapshot_batch).await;
                    }
                    if repro_batch.len() >= 100 {
                        Self::flush_reproductions(&pool, run_id, &mut repro_batch).await;
                    }
                    if learning_batch.len() >= 200 {
                        Self::flush_learning(&pool, run_id, &mut learning_batch).await;
                    }
                }
                Err(_) => {
                    // Channel closed, flush and exit
                    Self::flush_births(&pool, run_id, &mut birth_batch).await;
                    Self::flush_deaths(&pool, run_id, &mut death_batch).await;
                    Self::flush_snapshots(&pool, run_id, &mut snapshot_batch).await;
                    Self::flush_reproductions(&pool, run_id, &mut repro_batch).await;
                    Self::flush_learning(&pool, run_id, &mut learning_batch).await;
                    break;
                }
            }
        }
    }

    async fn flush_births(pool: &PgPool, run_id: Uuid, batch: &mut Vec<DbEvent>) {
        if batch.is_empty() {
            return;
        }
        let mut tx = match pool.begin().await {
            Ok(tx) => tx,
            Err(e) => {
                log::error!("DB transaction begin failed: {}", e);
                batch.clear();
                return;
            }
        };
        for event in batch.drain(..) {
            if let DbEvent::OrganismBirth {
                organism_id, lineage_id, generation, parent1_id, parent2_id,
                step, x, y, brain_layers, brain_neurons, brain_connections, is_predator,
            } = event
            {
                let _ = sqlx::query(
                    "INSERT INTO organisms (run_id, organism_id, lineage_id, generation, parent1_id, parent2_id, birth_step, birth_x, birth_y, brain_layers, brain_neurons, brain_connections, is_predator) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)"
                )
                .bind(run_id)
                .bind(organism_id as i64)
                .bind(lineage_id as i32)
                .bind(generation as i16)
                .bind(parent1_id.map(|v| v as i64))
                .bind(parent2_id.map(|v| v as i64))
                .bind(step as i64)
                .bind(x as i16)
                .bind(y as i16)
                .bind(brain_layers)
                .bind(brain_neurons)
                .bind(brain_connections)
                .bind(is_predator)
                .execute(&mut *tx)
                .await;
            }
        }
        let _ = tx.commit().await;
    }

    async fn flush_deaths(pool: &PgPool, run_id: Uuid, batch: &mut Vec<DbEvent>) {
        if batch.is_empty() {
            return;
        }
        let mut tx = match pool.begin().await {
            Ok(tx) => tx,
            Err(e) => {
                log::error!("DB transaction begin failed: {}", e);
                batch.clear();
                return;
            }
        };
        for event in batch.drain(..) {
            if let DbEvent::OrganismDeath {
                organism_id, step, x, y, death_cause,
                food_eaten, kills, offspring, max_energy,
            } = event
            {
                let _ = sqlx::query(
                    "UPDATE organisms SET death_step=$1, death_x=$2, death_y=$3, death_cause=$4, lifetime_food_eaten=$5, lifetime_kills=$6, lifetime_offspring=$7, max_energy=$8 WHERE run_id=$9 AND organism_id=$10"
                )
                .bind(step as i64)
                .bind(x as i16)
                .bind(y as i16)
                .bind(&death_cause)
                .bind(food_eaten as i32)
                .bind(kills as i16)
                .bind(offspring as i16)
                .bind(max_energy)
                .bind(run_id)
                .bind(organism_id as i64)
                .execute(&mut *tx)
                .await;
            }
        }
        let _ = tx.commit().await;
    }

    async fn flush_snapshots(pool: &PgPool, run_id: Uuid, batch: &mut Vec<DbEvent>) {
        if batch.is_empty() {
            return;
        }
        let mut tx = match pool.begin().await {
            Ok(tx) => tx,
            Err(e) => {
                log::error!("DB transaction begin failed: {}", e);
                batch.clear();
                return;
            }
        };
        for event in batch.drain(..) {
            if let DbEvent::OrganismSnapshot {
                step, organism_id, x, y, energy, health, age, size,
                kills, offspring, food_eaten, brain_layers, brain_neurons, brain_connections,
                is_predator, last_action,
            } = event
            {
                let _ = sqlx::query(
                    "INSERT INTO organism_snapshots (run_id, step, organism_id, x, y, energy, health, age, size, kills, offspring, food_eaten, brain_layers, brain_neurons, brain_connections, is_predator, last_action) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)"
                )
                .bind(run_id)
                .bind(step as i64)
                .bind(organism_id as i64)
                .bind(x as i16)
                .bind(y as i16)
                .bind(energy)
                .bind(health)
                .bind(age as i32)
                .bind(size)
                .bind(kills as i16)
                .bind(offspring as i16)
                .bind(food_eaten as i32)
                .bind(brain_layers)
                .bind(brain_neurons)
                .bind(brain_connections)
                .bind(is_predator)
                .bind(last_action.as_deref())
                .execute(&mut *tx)
                .await;
            }
        }
        let _ = tx.commit().await;
    }

    async fn flush_reproductions(pool: &PgPool, run_id: Uuid, batch: &mut Vec<DbEvent>) {
        if batch.is_empty() {
            return;
        }
        let mut tx = match pool.begin().await {
            Ok(tx) => tx,
            Err(e) => {
                log::error!("DB transaction begin failed: {}", e);
                batch.clear();
                return;
            }
        };
        for event in batch.drain(..) {
            if let DbEvent::ReproductionEvent {
                step, parent1_id, parent2_id, child_id,
                child_lineage_id, child_generation, mutation_count,
                parent1_energy, child_energy,
            } = event
            {
                let _ = sqlx::query(
                    "INSERT INTO reproduction_events (run_id, step, parent1_id, parent2_id, child_id, child_lineage_id, child_generation, mutation_count, parent1_energy, child_energy) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)"
                )
                .bind(run_id)
                .bind(step as i64)
                .bind(parent1_id as i64)
                .bind(parent2_id.map(|v| v as i64))
                .bind(child_id as i64)
                .bind(child_lineage_id as i32)
                .bind(child_generation as i16)
                .bind(mutation_count)
                .bind(parent1_energy)
                .bind(child_energy)
                .execute(&mut *tx)
                .await;
            }
        }
        let _ = tx.commit().await;
    }

    async fn flush_learning(pool: &PgPool, run_id: Uuid, batch: &mut Vec<DbEvent>) {
        if batch.is_empty() {
            return;
        }
        let mut tx = match pool.begin().await {
            Ok(tx) => tx,
            Err(e) => {
                log::error!("DB transaction begin failed: {}", e);
                batch.clear();
                return;
            }
        };
        for event in batch.drain(..) {
            if let DbEvent::LearningEvent {
                step, organism_id, reward, total_lifetime_reward,
                successful_forages, failed_forages, weight_updates_count,
            } = event
            {
                let _ = sqlx::query(
                    "INSERT INTO learning_events (run_id, step, organism_id, reward, total_lifetime_reward, successful_forages, failed_forages, weight_updates_count) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)"
                )
                .bind(run_id)
                .bind(step as i64)
                .bind(organism_id as i64)
                .bind(reward)
                .bind(total_lifetime_reward)
                .bind(successful_forages as i32)
                .bind(failed_forages as i32)
                .bind(weight_updates_count)
                .execute(&mut *tx)
                .await;
            }
        }
        let _ = tx.commit().await;
    }

    async fn write_world_snapshot(pool: &PgPool, run_id: Uuid, event: DbEvent) {
        if let DbEvent::WorldSnapshot {
            step, population, births, deaths, kills, max_generation,
            avg_energy, avg_age, avg_brain_layers, avg_brain_neurons, avg_brain_connections,
            predator_count, lineage_count, total_food,
        } = event
        {
            let _ = sqlx::query(
                "INSERT INTO world_snapshots (run_id, step, population, births, deaths, kills, max_generation, avg_energy, avg_age, avg_brain_layers, avg_brain_neurons, avg_brain_connections, predator_count, lineage_count, total_food) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)"
            )
            .bind(run_id)
            .bind(step as i64)
            .bind(population)
            .bind(births)
            .bind(deaths)
            .bind(kills)
            .bind(max_generation)
            .bind(avg_energy)
            .bind(avg_age)
            .bind(avg_brain_layers)
            .bind(avg_brain_neurons)
            .bind(avg_brain_connections)
            .bind(predator_count)
            .bind(lineage_count)
            .bind(total_food)
            .execute(pool)
            .await;
        }
    }

    // -- Public API (called from main simulation thread) --

    /// Send an event to the DB writer. Non-blocking.
    pub fn send(&self, event: DbEvent) {
        let _ = self.sender.send(event);
    }

    /// Clone the sender for use in other components (e.g., World)
    pub fn sender_clone(&self) -> mpsc::Sender<DbEvent> {
        self.sender.clone()
    }

    /// Gracefully shut down the DB writer thread.
    pub fn shutdown(mut self, total_steps: u64, final_population: i32, max_generation: i32) {
        let _ = self.sender.send(DbEvent::Shutdown {
            total_steps,
            final_population,
            max_generation,
        });
        if let Some(handle) = self.writer_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Send shutdown event to cleanly terminate the writer thread
        let _ = self.sender.send(DbEvent::Shutdown {
            total_steps: 0,
            final_population: 0,
            max_generation: 0,
        });
        // Wait for writer thread to finish
        if let Some(handle) = self.writer_handle.take() {
            let _ = handle.join();
        }
    }
}
