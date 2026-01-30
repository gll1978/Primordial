-- Queries per test rapporto popolazione/territorio
-- Usa organism_snapshots per posizioni durante la simulazione
-- Usa organisms per dati di nascita/morte

-- Parametro: sostituire RUN_ID con l'UUID della run specifica
-- Es: 'b65a43ba-15df-4e76-a3a3-2abbe89467ba'

-- ============================================================================
-- 1. VERIFICA DISPERSIONE SPAZIALE
-- Misura quanto gli organismi sono distribuiti sulla griglia
-- Se spread < 10% della grid → PROBLEMA (clustering eccessivo)
-- ============================================================================
SELECT
    run_id,
    step,
    AVG(x) as center_x,
    AVG(y) as center_y,
    STDDEV(x) as spread_x,
    STDDEV(y) as spread_y,
    COUNT(*) as population
FROM organism_snapshots
WHERE run_id = :run_id
  AND step = (SELECT MAX(step) FROM organism_snapshots WHERE run_id = :run_id)
GROUP BY run_id, step;

-- Alternativa: dispersione nel tempo
SELECT
    step,
    AVG(x) as center_x,
    AVG(y) as center_y,
    STDDEV(x) as spread_x,
    STDDEV(y) as spread_y,
    COUNT(*) as population
FROM organism_snapshots
WHERE run_id = :run_id
GROUP BY step
ORDER BY step;

-- ============================================================================
-- 2. VERIFICA MOVIMENTO
-- Organismi che si sono mossi poco → potrebbero essere bloccati
-- ============================================================================
SELECT
    organism_id,
    COUNT(DISTINCT x) as unique_x_positions,
    COUNT(DISTINCT y) as unique_y_positions,
    MAX(x) - MIN(x) as x_range,
    MAX(y) - MIN(y) as y_range,
    COUNT(*) as snapshots_count
FROM organism_snapshots
WHERE run_id = :run_id
GROUP BY organism_id
HAVING COUNT(DISTINCT x) < 3 OR COUNT(DISTINCT y) < 3
ORDER BY unique_x_positions + unique_y_positions ASC
LIMIT 20;

-- Statistiche generali sul movimento
SELECT
    AVG(unique_positions) as avg_unique_positions,
    MIN(unique_positions) as min_unique_positions,
    MAX(unique_positions) as max_unique_positions,
    STDDEV(unique_positions) as stddev_positions
FROM (
    SELECT
        organism_id,
        COUNT(DISTINCT x || ',' || y) as unique_positions
    FROM organism_snapshots
    WHERE run_id = :run_id
    GROUP BY organism_id
) movement_stats;

-- ============================================================================
-- 3. VERIFICA OFFSPRING DISTANCE
-- Distanza tra genitori e figli alla nascita
-- Se avg < 2 → Offspring nascono troppo vicini ai genitori
-- ============================================================================
SELECT
    AVG(ABS(o1.birth_x - o2.birth_x)) as avg_distance_x,
    AVG(ABS(o1.birth_y - o2.birth_y)) as avg_distance_y,
    AVG(SQRT(POWER(o1.birth_x - o2.birth_x, 2) + POWER(o1.birth_y - o2.birth_y, 2))) as avg_euclidean_distance,
    COUNT(*) as parent_child_pairs
FROM organisms o1
JOIN organisms o2 ON o2.parent1_id = o1.organism_id AND o2.run_id = o1.run_id
WHERE o1.run_id = :run_id;

-- Distribuzione delle distanze
SELECT
    FLOOR(SQRT(POWER(o1.birth_x - o2.birth_x, 2) + POWER(o1.birth_y - o2.birth_y, 2))) as distance_bucket,
    COUNT(*) as count
FROM organisms o1
JOIN organisms o2 ON o2.parent1_id = o1.organism_id AND o2.run_id = o1.run_id
WHERE o1.run_id = :run_id
GROUP BY distance_bucket
ORDER BY distance_bucket;

-- ============================================================================
-- 4. STATISTICHE GENERALI SIMULAZIONE
-- ============================================================================
SELECT
    sr.run_id,
    sr.started_at,
    sr.ended_at,
    sr.total_steps,
    sr.final_population,
    sr.max_generation,
    sr.config->>'grid_size' as grid_size,
    (SELECT COUNT(*) FROM organisms WHERE run_id = sr.run_id) as total_organisms_born,
    (SELECT COUNT(*) FROM organisms WHERE run_id = sr.run_id AND death_step IS NOT NULL) as total_deaths,
    (SELECT AVG(death_step - birth_step) FROM organisms WHERE run_id = sr.run_id AND death_step IS NOT NULL) as avg_lifespan
FROM simulation_runs sr
WHERE sr.run_id = :run_id;

-- ============================================================================
-- 5. CONFRONTO TRA SIMULAZIONI (grid 80 vs grid 255)
-- ============================================================================
SELECT
    sr.run_id,
    sr.config->>'grid_size' as grid_size,
    sr.total_steps,
    sr.final_population,
    sr.max_generation,
    (SELECT STDDEV(x) FROM organism_snapshots os WHERE os.run_id = sr.run_id AND os.step = sr.total_steps) as final_spread_x,
    (SELECT STDDEV(y) FROM organism_snapshots os WHERE os.run_id = sr.run_id AND os.step = sr.total_steps) as final_spread_y,
    (SELECT AVG(SQRT(POWER(o1.birth_x - o2.birth_x, 2) + POWER(o1.birth_y - o2.birth_y, 2)))
     FROM organisms o1 JOIN organisms o2 ON o2.parent1_id = o1.organism_id AND o2.run_id = o1.run_id
     WHERE o1.run_id = sr.run_id) as avg_offspring_distance
FROM simulation_runs sr
WHERE sr.total_steps > 0
ORDER BY sr.started_at DESC
LIMIT 5;
