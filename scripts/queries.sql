-- ============================================================================
-- PRIMORDIAL V2 - Query Singole per Analisi Rapide
-- ============================================================================
-- Copia-incolla le query che ti servono
-- ============================================================================

-- ===================
-- INFO RAPIDE
-- ===================

-- Conteggio totale organismi
SELECT COUNT(*) as total, COUNT(CASE WHEN death_step IS NULL THEN 1 END) as alive FROM organisms;

-- Range step simulazione
SELECT MIN(birth_step) as min_step, MAX(birth_step) as max_step FROM organisms;

-- Popolazione attuale per tipo
SELECT
    CASE WHEN is_predator THEN 'Predator' ELSE 'Herbivore' END as tipo,
    COUNT(*) as count
FROM organisms WHERE death_step IS NULL GROUP BY is_predator;

-- ===================
-- LINEAGES
-- ===================

-- Top 10 lineages per generazione
SELECT lineage_id, COUNT(*) as tot, MAX(generation) as max_gen,
       ROUND(AVG(brain_layers)::numeric, 2) as avg_brain
FROM organisms GROUP BY lineage_id ORDER BY max_gen DESC LIMIT 10;

-- Lineages attivi (con organismi vivi)
SELECT lineage_id, COUNT(*) as vivi, MAX(generation) as max_gen
FROM organisms WHERE death_step IS NULL
GROUP BY lineage_id ORDER BY vivi DESC LIMIT 10;

-- Confronto lineage specifico vs altri (cambia ID)
SELECT
    CASE WHEN lineage_id = 269 THEN 'Target' ELSE 'Others' END as grp,
    COUNT(*) as tot,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain,
    ROUND(AVG(lifetime_offspring)::numeric, 3) as avg_offspring
FROM organisms GROUP BY CASE WHEN lineage_id = 269 THEN 'Target' ELSE 'Others' END;

-- ===================
-- PREDATORI
-- ===================

-- Predatori vs Non-Predatori
SELECT
    is_predator,
    COUNT(*) as total,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain,
    ROUND(AVG(lifetime_kills)::numeric, 2) as avg_kills
FROM organisms GROUP BY is_predator;

-- Top killer
SELECT organism_id, lineage_id, lifetime_kills, brain_layers
FROM organisms WHERE lifetime_kills > 0 ORDER BY lifetime_kills DESC LIMIT 10;

-- % predatori nel tempo
SELECT (step / 10000) * 10 as time_k,
       ROUND(100.0 * SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) / COUNT(*), 2) as pred_pct
FROM organism_snapshots GROUP BY step / 10000 ORDER BY time_k;

-- ===================
-- CERVELLI
-- ===================

-- Distribuzione complessità cerebrale
SELECT
    CASE WHEN brain_layers < 10 THEN '1-9'
         WHEN brain_layers < 20 THEN '10-19'
         WHEN brain_layers < 30 THEN '20-29'
         ELSE '30+' END as layers,
    COUNT(*) as count
FROM organisms GROUP BY 1 ORDER BY 1;

-- Evoluzione brain_layers nel tempo
SELECT (step / 10000) * 10 as time_k, ROUND(AVG(brain_layers)::numeric, 2) as avg_brain
FROM organism_snapshots GROUP BY step / 10000 ORDER BY time_k;

-- Correlazione cervello-successo
SELECT
    CASE WHEN brain_layers < 15 THEN 'Simple' ELSE 'Complex' END as brain_type,
    ROUND(AVG(lifetime_offspring)::numeric, 3) as avg_offspring,
    ROUND(AVG(lifetime_kills)::numeric, 3) as avg_kills
FROM organisms GROUP BY 1;

-- ===================
-- SOPRAVVIVENZA
-- ===================

-- Cause di morte
SELECT death_cause, COUNT(*) as count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
FROM organisms WHERE death_step IS NOT NULL GROUP BY death_cause ORDER BY count DESC;

-- Lifespan medio per generazione
SELECT generation, COUNT(*) as count,
       ROUND(AVG(death_step - birth_step)::numeric, 1) as avg_lifespan
FROM organisms WHERE death_step IS NOT NULL
GROUP BY generation ORDER BY generation LIMIT 20;

-- ===================
-- RUN OUTCOMES
-- ===================

-- Aggiungi nuova run
-- INSERT INTO run_outcomes VALUES ('run_X', <seed>, <steps>, <herb%>, <pred%>, 0, '<dominant>');

-- Distribuzione outcomes
SELECT dominant_species, COUNT(*) as freq,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM run_outcomes), 1) as pct
FROM run_outcomes GROUP BY dominant_species ORDER BY freq DESC;

-- Lista tutte le run
SELECT * FROM run_outcomes ORDER BY steps DESC;

-- ===================
-- SNAPSHOT ANALYSIS
-- ===================

-- Popolazione per step (ultimi 10 snapshot)
SELECT step, COUNT(*) as pop, ROUND(AVG(energy)::numeric, 1) as avg_energy
FROM organism_snapshots
WHERE step > (SELECT MAX(step) - 1000 FROM organism_snapshots)
GROUP BY step ORDER BY step DESC LIMIT 10;

-- Giorno vs Notte (cycle 2000 step)
SELECT CASE WHEN (step % 2000) < 1000 THEN 'Day' ELSE 'Night' END as period,
       ROUND(AVG(energy)::numeric, 2) as avg_energy
FROM organism_snapshots GROUP BY 1;

-- ===================
-- EXPORT
-- ===================

-- Export lineages to CSV
-- \copy (SELECT * FROM organisms WHERE lineage_id = 269) TO '/tmp/lineage_269.csv' CSV HEADER;

-- Export summary to CSV
-- \copy (SELECT lineage_id, COUNT(*) as tot, MAX(generation) as max_gen FROM organisms GROUP BY lineage_id ORDER BY max_gen DESC) TO '/tmp/lineages_summary.csv' CSV HEADER;

-- ===================
-- EARLY GAME ANALYSIS (Primi 1000 step)
-- ===================

-- Cosa succede nei primi 1000 step che determina l'outcome?
SELECT
    CASE WHEN is_predator THEN 'Predator' ELSE 'Herbivore' END as species,
    MIN(birth_step) as first_birth,
    COUNT(*) as births_0_1k,
    COUNT(DISTINCT lineage_id) as lineages_attivi
FROM organisms
WHERE birth_step < 1000
GROUP BY is_predator
ORDER BY first_birth;

-- Chi si riproduce per primo vince?
-- Ipotesi: specie che si riproduce PRIMA nei primi 1000 step → domina alla fine!

-- Territori occupati per primi (primi 1000 step)
WITH early_territory AS (
    SELECT
        CASE WHEN is_predator THEN 'Predator' ELSE 'Herbivore' END as species,
        x, y,
        MIN(step) as first_occupation
    FROM organism_snapshots
    WHERE step < 1000
    GROUP BY is_predator, x, y
)
SELECT
    species,
    COUNT(*) as territories_claimed,
    ROUND(AVG(first_occupation)::numeric, 1) as avg_claim_time
FROM early_territory
GROUP BY species
ORDER BY territories_claimed DESC;

-- Species con più territori early → probabile vincitore!

-- ===================
-- MECCANISMI DI DOMINANZA
-- ===================

-- Analisi vantaggio predatore (brain, kills, efficiency)
SELECT
    CASE WHEN is_predator THEN 'Predator' ELSE 'Herbivore' END as species,
    ROUND(AVG(brain_layers)::numeric, 2) as avg_brain,
    ROUND(AVG(brain_neurons)::numeric, 1) as avg_neurons,
    ROUND(AVG(lifetime_kills)::numeric, 2) as avg_kills,
    ROUND(AVG(lifetime_food_eaten)::numeric, 1) as avg_food,
    ROUND(AVG(max_energy)::numeric, 1) as avg_max_energy,
    ROUND(AVG(lifetime_offspring)::numeric, 3) as avg_offspring
FROM organisms
GROUP BY is_predator;

-- Efficienza energetica per tipo
SELECT
    CASE WHEN is_predator THEN 'Predator' ELSE 'Herbivore' END as species,
    ROUND(AVG(max_energy / NULLIF(COALESCE(death_step, 210000) - birth_step, 0))::numeric, 4) as energy_per_step,
    ROUND(AVG(lifetime_offspring::float / NULLIF(COALESCE(death_step, 210000) - birth_step, 0))::numeric, 6) as offspring_per_step
FROM organisms
WHERE death_step IS NOT NULL OR death_step IS NULL
GROUP BY is_predator;

-- ===================
-- ANALISI EQUILIBRIO ECOSISTEMA
-- ===================

-- Ratio predatori/prede nel tempo
SELECT
    (step / 5000) * 5 as time_k,
    SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) as predators,
    SUM(CASE WHEN NOT is_predator THEN 1 ELSE 0 END) as herbivores,
    ROUND(
        SUM(CASE WHEN is_predator THEN 1 ELSE 0 END)::numeric /
        NULLIF(SUM(CASE WHEN NOT is_predator THEN 1 ELSE 0 END), 0),
        4
    ) as pred_herb_ratio
FROM organism_snapshots
GROUP BY step / 5000
ORDER BY time_k;

-- Stabilità popolazione (deviazione standard)
SELECT
    (step / 10000) * 10 as time_k,
    COUNT(*) as snapshots,
    ROUND(STDDEV(energy)::numeric, 2) as energy_stddev,
    ROUND(STDDEV(brain_layers)::numeric, 2) as brain_stddev
FROM organism_snapshots
GROUP BY step / 10000
ORDER BY time_k;

-- ===================
-- ANALISI COMPETIZIONE TERRITORIALE
-- ===================

-- Densità popolazione per area (griglia 10x10)
SELECT
    (x / 10) as grid_x,
    (y / 10) as grid_y,
    COUNT(*) as organisms,
    SUM(CASE WHEN is_predator THEN 1 ELSE 0 END) as predators,
    ROUND(AVG(energy)::numeric, 1) as avg_energy
FROM organism_snapshots
WHERE step = (SELECT MAX(step) FROM organism_snapshots)
GROUP BY x / 10, y / 10
ORDER BY organisms DESC
LIMIT 20;

-- Posizione nascita vs morte (dispersione)
SELECT
    CASE WHEN is_predator THEN 'Predator' ELSE 'Herbivore' END as species,
    ROUND(AVG(ABS(death_x - birth_x) + ABS(death_y - birth_y))::numeric, 2) as avg_displacement,
    ROUND(STDDEV(ABS(death_x - birth_x) + ABS(death_y - birth_y))::numeric, 2) as displacement_stddev
FROM organisms
WHERE death_step IS NOT NULL AND death_x IS NOT NULL
GROUP BY is_predator;

-- ===================
-- PREDIZIONE VINCITORE
-- ===================

-- Chi dominerà? Guarda i primi 5000 step
WITH early_stats AS (
    SELECT
        CASE WHEN is_predator THEN 'Predator' ELSE 'Herbivore' END as species,
        COUNT(*) as pop_5k,
        AVG(brain_layers) as brain_5k,
        COUNT(DISTINCT lineage_id) as lineages_5k
    FROM organism_snapshots
    WHERE step BETWEEN 4000 AND 5000
    GROUP BY is_predator
),
late_stats AS (
    SELECT
        CASE WHEN is_predator THEN 'Predator' ELSE 'Herbivore' END as species,
        COUNT(*) as pop_final
    FROM organisms
    WHERE death_step IS NULL
    GROUP BY is_predator
)
SELECT
    e.species,
    e.pop_5k as popolazione_5k,
    ROUND(e.brain_5k::numeric, 2) as brain_5k,
    e.lineages_5k,
    l.pop_final as popolazione_finale,
    CASE WHEN l.pop_final > (SELECT SUM(pop_final)/2 FROM late_stats) THEN 'WINNER' ELSE '' END as status
FROM early_stats e
JOIN late_stats l ON e.species = l.species
ORDER BY l.pop_final DESC;
