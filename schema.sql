-- PRIMORDIAL V2 Database Schema
-- PostgreSQL 14+

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Simulation runs
CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    config JSONB NOT NULL,
    seed BIGINT,
    total_steps BIGINT DEFAULT 0,
    final_population INT DEFAULT 0,
    max_generation INT DEFAULT 0
);

-- Individual organisms (birth + death records)
CREATE TABLE IF NOT EXISTS organisms (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES simulation_runs(run_id),
    organism_id BIGINT NOT NULL,
    lineage_id INT NOT NULL,
    generation SMALLINT NOT NULL DEFAULT 0,
    parent1_id BIGINT,
    parent2_id BIGINT,
    birth_step BIGINT NOT NULL,
    death_step BIGINT,
    death_cause TEXT,  -- 'starvation', 'predation', 'old_age'
    birth_x SMALLINT NOT NULL,
    birth_y SMALLINT NOT NULL,
    death_x SMALLINT,
    death_y SMALLINT,
    lifetime_food_eaten INT DEFAULT 0,
    lifetime_kills SMALLINT DEFAULT 0,
    lifetime_offspring SMALLINT DEFAULT 0,
    max_energy REAL DEFAULT 0,
    brain_layers INT DEFAULT 0,
    brain_neurons INT DEFAULT 0,
    brain_connections INT DEFAULT 0,
    is_predator BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_organisms_run_id ON organisms(run_id);
CREATE INDEX IF NOT EXISTS idx_organisms_run_organism ON organisms(run_id, organism_id);
CREATE INDEX IF NOT EXISTS idx_organisms_lineage ON organisms(run_id, lineage_id);
CREATE INDEX IF NOT EXISTS idx_organisms_birth_step ON organisms(run_id, birth_step);
CREATE INDEX IF NOT EXISTS idx_organisms_death_step ON organisms(run_id, death_step);

-- Periodic organism snapshots (position, energy, brain state)
CREATE TABLE IF NOT EXISTS organism_snapshots (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES simulation_runs(run_id),
    step BIGINT NOT NULL,
    organism_id BIGINT NOT NULL,
    x SMALLINT NOT NULL,
    y SMALLINT NOT NULL,
    energy REAL NOT NULL,
    health REAL NOT NULL,
    age INT NOT NULL,
    size REAL NOT NULL,
    kills SMALLINT DEFAULT 0,
    offspring SMALLINT DEFAULT 0,
    food_eaten INT DEFAULT 0,
    brain_layers INT DEFAULT 0,
    brain_neurons INT DEFAULT 0,
    brain_connections INT DEFAULT 0,
    is_predator BOOLEAN DEFAULT FALSE,
    last_action TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_run_step ON organism_snapshots(run_id, step);
CREATE INDEX IF NOT EXISTS idx_snapshots_organism ON organism_snapshots(run_id, organism_id);

-- Learning events (Hebbian weight updates)
CREATE TABLE IF NOT EXISTS learning_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES simulation_runs(run_id),
    step BIGINT NOT NULL,
    organism_id BIGINT NOT NULL,
    reward REAL NOT NULL,
    total_lifetime_reward REAL NOT NULL,
    successful_forages INT DEFAULT 0,
    failed_forages INT DEFAULT 0,
    weight_updates_count INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_learning_run_step ON learning_events(run_id, step);

-- Reproduction events
CREATE TABLE IF NOT EXISTS reproduction_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES simulation_runs(run_id),
    step BIGINT NOT NULL,
    parent1_id BIGINT NOT NULL,
    parent2_id BIGINT,  -- NULL for asexual
    child_id BIGINT NOT NULL,
    child_lineage_id INT NOT NULL,
    child_generation SMALLINT NOT NULL,
    mutation_count INT DEFAULT 0,
    parent1_energy REAL,
    child_energy REAL
);

CREATE INDEX IF NOT EXISTS idx_repro_run_step ON reproduction_events(run_id, step);

-- Aggregate world snapshots per step
CREATE TABLE IF NOT EXISTS world_snapshots (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES simulation_runs(run_id),
    step BIGINT NOT NULL,
    population INT NOT NULL,
    births INT DEFAULT 0,
    deaths INT DEFAULT 0,
    kills INT DEFAULT 0,
    max_generation SMALLINT DEFAULT 0,
    avg_energy REAL,
    avg_age REAL,
    avg_brain_layers REAL,
    avg_brain_neurons REAL,
    avg_brain_connections REAL,
    predator_count INT DEFAULT 0,
    lineage_count INT DEFAULT 0,
    total_food REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_world_run_step ON world_snapshots(run_id, step);
