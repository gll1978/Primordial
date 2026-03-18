-- ============================================================================
-- PRIMORDIAL V2 - Script Analisi Simulazione
-- ============================================================================
-- Esegui con: PGPASSWORD=primordial psql -U primordial -h localhost -d primordial_v2 -f scripts/analyze_simulation.sql
-- ============================================================================

\echo '============================================================================'
\echo 'PRIMORDIAL V2 - ANALISI SIMULAZIONE'
\echo '============================================================================'
\echo ''

-- ============================================================================
-- 1. INFO GENERALI SIMULAZIONE
-- ============================================================================
\echo '>>> 1. INFO GENERALI SIMULAZIONE'
\echo ''

SELECT
    COUNT(*) as total_organisms,
    COUNT(CASE WHEN death_step IS NULL THEN 1 END) as alive,
    COUNT(CASE WHEN death_step IS NOT NULL THEN 1 END) as dead,
    MIN(birth_step) as first_step,
    MAX(COALESCE(death_step, birth_step)) as last_step,
    COUNT(DISTINCT lineage_id) as total_lineages,
    COUNT(DISTINCT run_id) as total_runs
FROM organisms;

-- ============================================================================
-- 2. TOP 15 LINEAGES PER GENERAZIONE MASSIMA
-- ============================================================================
\echo ''
\echo '>>> 2. TOP 15 LINEAGES PER GENERAZIONE'
\echo ''

SELECT
    lineage_id,
    COUNT(*) as total_organisms,
    MAX(generation) as max_gen,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain_layers,
    ROUND(AVG(brain_neurons)::numeric, 1) as avg_neurons,
    ROUND(AVG(brain_connections)::numeric, 1) as avg_connections,
    SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) as predator_count,
    ROUND(100.0 * SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) / COUNT(*), 1) as predator_pct
FROM organisms
GROUP BY lineage_id
ORDER BY max_gen DESC
LIMIT 15;

-- ============================================================================
-- 3. CONFRONTO LINEAGE DOMINANTE VS ALTRI
-- ============================================================================
\echo ''
\echo '>>> 3. CONFRONTO LINEAGE DOMINANTE (#269) VS ALTRI'
\echo ''

SELECT
    CASE WHEN lineage_id = 269 THEN 'Lineage #269' ELSE 'Altri' END as gruppo,
    COUNT(*) as totale,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_layers,
    ROUND(AVG(brain_neurons)::numeric, 1) as avg_neurons,
    ROUND(AVG(brain_connections)::numeric, 1) as avg_connections,
    ROUND(AVG(lifetime_offspring)::numeric, 3) as avg_offspring,
    ROUND(AVG(lifetime_kills)::numeric, 3) as avg_kills,
    ROUND(AVG(lifetime_food_eaten)::numeric, 1) as avg_food_eaten,
    ROUND(AVG(COALESCE(death_step, (SELECT MAX(birth_step) FROM organisms)) - birth_step)::numeric, 1) as avg_lifespan
FROM organisms
GROUP BY CASE WHEN lineage_id = 269 THEN 'Lineage #269' ELSE 'Altri' END;

-- ============================================================================
-- 4. PREDATORI VS NON-PREDATORI
-- ============================================================================
\echo ''
\echo '>>> 4. PREDATORI VS NON-PREDATORI'
\echo ''

SELECT
    CASE WHEN is_predator THEN 'Predatore' ELSE 'Non-Predatore' END as tipo,
    COUNT(*) as totale,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM organisms), 2) as percentuale,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain,
    ROUND(AVG(brain_neurons)::numeric, 1) as avg_neurons,
    ROUND(AVG(lifetime_offspring)::numeric, 3) as avg_offspring,
    ROUND(AVG(COALESCE(death_step, (SELECT MAX(birth_step) FROM organisms)) - birth_step)::numeric, 1) as avg_lifespan,
    ROUND(AVG(lifetime_kills)::numeric, 2) as avg_kills,
    ROUND(AVG(max_energy)::numeric, 1) as avg_peak_energy
FROM organisms
GROUP BY is_predator;

-- ============================================================================
-- 5. EVOLUZIONE NEL TEMPO (per 10k step)
-- ============================================================================
\echo ''
\echo '>>> 5. EVOLUZIONE NEL TEMPO (ogni 10k step)'
\echo ''

SELECT
    (step / 10000) * 10 as time_k,
    COUNT(*) as popolazione,
    SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) as predatori,
    ROUND(100.0 * SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_predatori,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain,
    ROUND(AVG(energy)::numeric, 1) as avg_energy
FROM organism_snapshots
GROUP BY step / 10000
ORDER BY time_k;

-- ============================================================================
-- 6. COMPLESSITÀ CEREBRALE VS SUCCESSO
-- ============================================================================
\echo ''
\echo '>>> 6. COMPLESSITÀ CEREBRALE VS SUCCESSO'
\echo ''

SELECT
    CASE
        WHEN brain_layers < 10 THEN '01-09 layers'
        WHEN brain_layers < 15 THEN '10-14 layers'
        WHEN brain_layers < 20 THEN '15-19 layers'
        WHEN brain_layers < 25 THEN '20-24 layers'
        WHEN brain_layers < 30 THEN '25-29 layers'
        ELSE '30+ layers'
    END as complessita,
    COUNT(*) as totale,
    ROUND(AVG(lifetime_offspring)::numeric, 3) as avg_offspring,
    ROUND(AVG(COALESCE(death_step, (SELECT MAX(birth_step) FROM organisms)) - birth_step)::numeric, 1) as avg_lifespan,
    ROUND(AVG(lifetime_kills)::numeric, 3) as avg_kills,
    ROUND(100.0 * SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_predator
FROM organisms
GROUP BY CASE
    WHEN brain_layers < 10 THEN '01-09 layers'
    WHEN brain_layers < 15 THEN '10-14 layers'
    WHEN brain_layers < 20 THEN '15-19 layers'
    WHEN brain_layers < 25 THEN '20-24 layers'
    WHEN brain_layers < 30 THEN '25-29 layers'
    ELSE '30+ layers'
END
ORDER BY complessita;

-- ============================================================================
-- 7. LINEAGES ANCORA ATTIVI (VIVI)
-- ============================================================================
\echo ''
\echo '>>> 7. LINEAGES ANCORA ATTIVI (TOP 15)'
\echo ''

SELECT
    lineage_id,
    COUNT(*) as vivi,
    MAX(generation) as max_gen,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain,
    ROUND(AVG(brain_neurons)::numeric, 1) as avg_neurons,
    MIN(birth_step) as oldest_birth,
    MAX(birth_step) as newest_birth
FROM organisms
WHERE death_step IS NULL
GROUP BY lineage_id
ORDER BY vivi DESC
LIMIT 15;

-- ============================================================================
-- 8. CAUSE DI MORTE
-- ============================================================================
\echo ''
\echo '>>> 8. CAUSE DI MORTE'
\echo ''

SELECT
    COALESCE(death_cause, 'Unknown') as causa,
    COUNT(*) as totale,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM organisms WHERE death_step IS NOT NULL), 2) as percentuale,
    ROUND(AVG(death_step - birth_step)::numeric, 1) as avg_lifespan
FROM organisms
WHERE death_step IS NOT NULL
GROUP BY death_cause
ORDER BY totale DESC;

-- ============================================================================
-- 9. NASCITE E MORTI PER PERIODO
-- ============================================================================
\echo ''
\echo '>>> 9. NASCITE E MORTI PER PERIODO (ogni 10k step)'
\echo ''

WITH births AS (
    SELECT (birth_step / 10000) * 10 as period, COUNT(*) as births
    FROM organisms
    GROUP BY birth_step / 10000
),
deaths AS (
    SELECT (death_step / 10000) * 10 as period, COUNT(*) as deaths
    FROM organisms
    WHERE death_step IS NOT NULL
    GROUP BY death_step / 10000
)
SELECT
    COALESCE(b.period, d.period) as time_k,
    COALESCE(b.births, 0) as nascite,
    COALESCE(d.deaths, 0) as morti,
    COALESCE(b.births, 0) - COALESCE(d.deaths, 0) as delta
FROM births b
FULL OUTER JOIN deaths d ON b.period = d.period
ORDER BY time_k;

-- ============================================================================
-- 10. OUTCOME RUN CORRENTE
-- ============================================================================
\echo ''
\echo '>>> 10. OUTCOME RUN CORRENTE (organismi vivi)'
\echo ''

SELECT
    COUNT(*) as popolazione_finale,
    ROUND(100.0 * SUM(CASE WHEN NOT is_predator THEN 1 ELSE 0 END) / COUNT(*), 2) as herbivore_pct,
    ROUND(100.0 * SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) / COUNT(*), 2) as predator_pct,
    CASE
        WHEN SUM(CASE WHEN NOT is_predator THEN 1 ELSE 0 END) > SUM(CASE WHEN is_predator THEN 1 ELSE 0 END)
        THEN 'herbivore'
        ELSE 'predator'
    END as dominant_species,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain,
    MAX(generation) as max_generation
FROM organisms
WHERE death_step IS NULL;

-- ============================================================================
-- 11. DISTRIBUZIONE RUN OUTCOMES (se tabella esiste)
-- ============================================================================
\echo ''
\echo '>>> 11. DISTRIBUZIONE RUN OUTCOMES'
\echo ''

SELECT
    dominant_species,
    COUNT(*) as frequency,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM run_outcomes), 1) as pct,
    ROUND(AVG(steps)::numeric, 0) as avg_steps,
    ROUND(AVG(herbivore_pct)::numeric, 1) as avg_herb_pct,
    ROUND(AVG(predator_pct)::numeric, 1) as avg_pred_pct
FROM run_outcomes
GROUP BY dominant_species
ORDER BY frequency DESC;

-- ============================================================================
-- 12. TOP ORGANISMI PER OFFSPRING
-- ============================================================================
\echo ''
\echo '>>> 12. TOP 10 ORGANISMI PER OFFSPRING'
\echo ''

SELECT
    organism_id,
    lineage_id,
    generation,
    lifetime_offspring,
    lifetime_kills,
    brain_layers,
    brain_neurons,
    is_predator,
    death_step - birth_step as lifespan
FROM organisms
WHERE death_step IS NOT NULL
ORDER BY lifetime_offspring DESC
LIMIT 10;

-- ============================================================================
-- 13. TOP ORGANISMI PER KILLS
-- ============================================================================
\echo ''
\echo '>>> 13. TOP 10 ORGANISMI PER KILLS'
\echo ''

SELECT
    organism_id,
    lineage_id,
    generation,
    lifetime_kills,
    lifetime_offspring,
    brain_layers,
    brain_neurons,
    is_predator,
    death_step - birth_step as lifespan
FROM organisms
WHERE death_step IS NOT NULL AND lifetime_kills > 0
ORDER BY lifetime_kills DESC
LIMIT 10;

-- ============================================================================
-- 14. STATISTICHE STAGIONALI (se disponibili in snapshots)
-- ============================================================================
\echo ''
\echo '>>> 14. ATTIVITÀ GIORNO VS NOTTE'
\echo ''

SELECT
    CASE WHEN (step % 2000) < 1000 THEN 'Day' ELSE 'Night' END as period,
    COUNT(*) as snapshots,
    ROUND(AVG(energy)::numeric, 2) as avg_energy,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain
FROM organism_snapshots
GROUP BY CASE WHEN (step % 2000) < 1000 THEN 'Day' ELSE 'Night' END;

-- ============================================================================
-- 15. DIVERSITÀ GENETICA PER PERIODO
-- ============================================================================
\echo ''
\echo '>>> 15. DIVERSITÀ GENETICA NEL TEMPO'
\echo ''

SELECT
    (birth_step / 10000) * 10 as time_k,
    COUNT(DISTINCT lineage_id) as lineages_attivi,
    COUNT(*) as nascite,
    ROUND(STDDEV(brain_layers)::numeric, 2) as brain_stddev,
    ROUND(STDDEV(brain_neurons)::numeric, 2) as neurons_stddev
FROM organisms
GROUP BY birth_step / 10000
ORDER BY time_k;

-- ============================================================================
\echo ''
\echo '============================================================================'
\echo 'ANALISI COMPLETATA'
\echo '============================================================================'
