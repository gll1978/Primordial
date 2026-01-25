//! Diversity metrics for population genetics analysis.

use crate::genetics::phylogeny::PhylogeneticTree;
use crate::organism::Organism;
use std::collections::{HashMap, HashSet};

/// Collection of diversity metrics for a population
#[derive(Clone, Debug, Default)]
pub struct DiversityMetrics {
    /// Simpson's diversity index (0 = no diversity, 1 = max diversity)
    pub simpsons_index: f32,
    /// Shannon entropy (higher = more diverse)
    pub shannon_entropy: f32,
    /// Mean genetic distance between organisms
    pub mean_genetic_distance: f32,
    /// Number of unique lineages
    pub lineage_count: usize,
    /// Number of unique species (by brain complexity)
    pub species_count: usize,
    /// Effective population size (genetic)
    pub effective_population: f32,
    /// Heterozygosity estimate
    pub heterozygosity: f32,
}

impl DiversityMetrics {
    /// Create empty metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Format as a human-readable string
    pub fn to_string(&self) -> String {
        format!(
            "Diversity: Simpson={:.3}, Shannon={:.3}, MeanDist={:.1}, Lineages={}, Species={}, Ne={:.1}",
            self.simpsons_index,
            self.shannon_entropy,
            self.mean_genetic_distance,
            self.lineage_count,
            self.species_count,
            self.effective_population
        )
    }
}

/// Calculate Simpson's diversity index
/// D = 1 - Σ(p_i²) where p_i is the proportion of lineage i
pub fn calculate_simpsons_index(organisms: &[Organism]) -> f32 {
    let alive: Vec<_> = organisms.iter().filter(|o| o.is_alive()).collect();

    if alive.is_empty() {
        return 0.0;
    }

    let mut lineage_counts: HashMap<u32, usize> = HashMap::new();

    for org in &alive {
        *lineage_counts.entry(org.lineage_id).or_insert(0) += 1;
    }

    let total = alive.len() as f32;
    let sum_squares: f32 = lineage_counts
        .values()
        .map(|&count| (count as f32 / total).powi(2))
        .sum();

    1.0 - sum_squares
}

/// Calculate Shannon entropy (diversity index)
/// H = -Σ(p_i * ln(p_i)) where p_i is the proportion of lineage i
pub fn calculate_shannon_entropy(organisms: &[Organism]) -> f32 {
    let alive: Vec<_> = organisms.iter().filter(|o| o.is_alive()).collect();

    if alive.is_empty() {
        return 0.0;
    }

    let mut lineage_counts: HashMap<u32, usize> = HashMap::new();

    for org in &alive {
        *lineage_counts.entry(org.lineage_id).or_insert(0) += 1;
    }

    let total = alive.len() as f32;
    let entropy: f32 = lineage_counts
        .values()
        .map(|&count| {
            let p = count as f32 / total;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    entropy
}

/// Calculate mean genetic distance between organisms
/// Uses phylogenetic tree to compute distances
pub fn calculate_mean_genetic_distance(
    organisms: &[Organism],
    phylogeny: &PhylogeneticTree,
) -> f32 {
    let alive: Vec<_> = organisms.iter().filter(|o| o.is_alive()).collect();

    if alive.len() < 2 {
        return 0.0;
    }

    let mut total_distance = 0.0;
    let mut comparisons = 0;

    // Sample up to 100 pairs for performance
    let sample_size = alive.len().min(100);

    for i in 0..sample_size {
        for j in (i + 1)..sample_size {
            if let Some(distance) = phylogeny.genetic_distance(alive[i].id, alive[j].id) {
                total_distance += distance as f32;
                comparisons += 1;
            }
        }
    }

    if comparisons > 0 {
        total_distance / comparisons as f32
    } else {
        0.0
    }
}

/// Count unique lineages in the population
pub fn count_lineages(organisms: &[Organism]) -> usize {
    organisms
        .iter()
        .filter(|o| o.is_alive())
        .map(|o| o.lineage_id)
        .collect::<HashSet<_>>()
        .len()
}

/// Count species by brain complexity (simplified speciation)
pub fn count_species(organisms: &[Organism], threshold: usize) -> usize {
    let alive: Vec<_> = organisms.iter().filter(|o| o.is_alive()).collect();

    if alive.is_empty() {
        return 0;
    }

    // Group by brain complexity buckets
    let mut complexity_buckets: HashSet<usize> = HashSet::new();

    for org in &alive {
        let bucket = org.brain.complexity() / threshold.max(1);
        complexity_buckets.insert(bucket);
    }

    complexity_buckets.len()
}

/// Calculate effective population size (Ne)
/// Uses variance in reproductive success
pub fn calculate_effective_population(organisms: &[Organism]) -> f32 {
    let alive: Vec<_> = organisms.iter().filter(|o| o.is_alive()).collect();

    if alive.is_empty() {
        return 0.0;
    }

    let n = alive.len() as f32;

    // Calculate variance in offspring count
    let mean_offspring: f32 = alive.iter()
        .map(|o| o.offspring_count as f32)
        .sum::<f32>() / n;

    let variance: f32 = alive.iter()
        .map(|o| (o.offspring_count as f32 - mean_offspring).powi(2))
        .sum::<f32>() / n;

    if variance > 0.0 {
        // Ne = N * mean_k / (variance + mean_k - 1)
        // where k is offspring count
        let mean_k = mean_offspring.max(1.0);
        n * mean_k / (variance + mean_k - 1.0).max(1.0)
    } else {
        n
    }
}

/// Calculate heterozygosity estimate based on lineage diversity
pub fn calculate_heterozygosity(organisms: &[Organism]) -> f32 {
    let alive: Vec<_> = organisms.iter().filter(|o| o.is_alive()).collect();

    if alive.len() < 2 {
        return 0.0;
    }

    let mut lineage_counts: HashMap<u32, usize> = HashMap::new();

    for org in &alive {
        *lineage_counts.entry(org.lineage_id).or_insert(0) += 1;
    }

    let n = alive.len() as f32;

    // Expected heterozygosity: H = 1 - Σ(p_i²) * n/(n-1)
    let sum_squares: f32 = lineage_counts
        .values()
        .map(|&count| (count as f32 / n).powi(2))
        .sum();

    (1.0 - sum_squares) * n / (n - 1.0)
}

/// Calculate all diversity metrics at once
pub fn calculate_all_metrics(
    organisms: &[Organism],
    phylogeny: &PhylogeneticTree,
) -> DiversityMetrics {
    DiversityMetrics {
        simpsons_index: calculate_simpsons_index(organisms),
        shannon_entropy: calculate_shannon_entropy(organisms),
        mean_genetic_distance: calculate_mean_genetic_distance(organisms, phylogeny),
        lineage_count: count_lineages(organisms),
        species_count: count_species(organisms, 3), // Group by complexity/3
        effective_population: calculate_effective_population(organisms),
        heterozygosity: calculate_heterozygosity(organisms),
    }
}

/// Track diversity over time
#[derive(Clone, Debug, Default)]
pub struct DiversityHistory {
    pub records: Vec<DiversityRecord>,
    pub record_interval: u64,
}

#[derive(Clone, Debug)]
pub struct DiversityRecord {
    pub time: u64,
    pub population: usize,
    pub metrics: DiversityMetrics,
}

impl DiversityHistory {
    pub fn new(record_interval: u64) -> Self {
        Self {
            records: Vec::new(),
            record_interval,
        }
    }

    /// Record current diversity metrics
    pub fn record(
        &mut self,
        time: u64,
        organisms: &[Organism],
        phylogeny: &PhylogeneticTree,
    ) {
        let population = organisms.iter().filter(|o| o.is_alive()).count();
        let metrics = calculate_all_metrics(organisms, phylogeny);

        self.records.push(DiversityRecord {
            time,
            population,
            metrics,
        });
    }

    /// Get latest record
    pub fn latest(&self) -> Option<&DiversityRecord> {
        self.records.last()
    }

    /// Get diversity trend (positive = increasing, negative = decreasing)
    pub fn diversity_trend(&self, window: usize) -> f32 {
        if self.records.len() < 2 {
            return 0.0;
        }

        let recent: Vec<_> = self.records.iter().rev().take(window).collect();

        if recent.len() < 2 {
            return 0.0;
        }

        let first = recent.last().unwrap().metrics.simpsons_index;
        let last = recent.first().unwrap().metrics.simpsons_index;

        last - first
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::from(
            "time,population,simpsons,shannon,mean_distance,lineages,species,effective_pop,heterozygosity\n"
        );

        for record in &self.records {
            csv.push_str(&format!(
                "{},{},{:.4},{:.4},{:.2},{},{},{:.1},{:.4}\n",
                record.time,
                record.population,
                record.metrics.simpsons_index,
                record.metrics.shannon_entropy,
                record.metrics.mean_genetic_distance,
                record.metrics.lineage_count,
                record.metrics.species_count,
                record.metrics.effective_population,
                record.metrics.heterozygosity,
            ));
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn create_test_population(count: usize, lineages: usize) -> Vec<Organism> {
        let config = Config::default();
        let mut organisms = Vec::new();

        for i in 0..count {
            let lineage_id = (i % lineages) as u32;
            let org = Organism::new(i as u64, lineage_id, 10, 10, &config);
            organisms.push(org);
        }

        organisms
    }

    #[test]
    fn test_simpsons_index_uniform() {
        // 10 organisms in 10 different lineages = max diversity
        let organisms = create_test_population(10, 10);
        let index = calculate_simpsons_index(&organisms);

        // Should be close to 0.9 (1 - 10 * 0.1²)
        assert!(index > 0.8 && index <= 1.0);
    }

    #[test]
    fn test_simpsons_index_single_lineage() {
        // All organisms in same lineage = no diversity
        let organisms = create_test_population(10, 1);
        let index = calculate_simpsons_index(&organisms);

        // Should be 0 (1 - 1.0²)
        assert!(index < 0.01);
    }

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution
        let organisms = create_test_population(100, 10);
        let entropy = calculate_shannon_entropy(&organisms);

        // Should be close to ln(10) ≈ 2.3
        assert!(entropy > 2.0);
    }

    #[test]
    fn test_count_lineages() {
        let organisms = create_test_population(100, 5);
        let count = count_lineages(&organisms);

        assert_eq!(count, 5);
    }

    #[test]
    fn test_effective_population() {
        let organisms = create_test_population(50, 10);
        let ne = calculate_effective_population(&organisms);

        // Should be close to actual population when variance is low
        assert!(ne > 0.0);
        assert!(ne <= 50.0);
    }

    #[test]
    fn test_heterozygosity() {
        // Diverse population
        let organisms = create_test_population(100, 20);
        let h = calculate_heterozygosity(&organisms);

        // Should be high with many lineages
        assert!(h > 0.5);
    }

    #[test]
    fn test_diversity_metrics() {
        let organisms = create_test_population(100, 10);
        let phylogeny = PhylogeneticTree::new();
        let metrics = calculate_all_metrics(&organisms, &phylogeny);

        assert!(metrics.simpsons_index > 0.0);
        assert!(metrics.shannon_entropy > 0.0);
        assert_eq!(metrics.lineage_count, 10);
    }

    #[test]
    fn test_diversity_history() {
        let mut history = DiversityHistory::new(100);
        let organisms = create_test_population(50, 5);
        let phylogeny = PhylogeneticTree::new();

        history.record(0, &organisms, &phylogeny);
        history.record(100, &organisms, &phylogeny);

        assert_eq!(history.records.len(), 2);
        assert!(history.latest().is_some());
    }

    #[test]
    fn test_csv_export() {
        let mut history = DiversityHistory::new(100);
        let organisms = create_test_population(50, 5);
        let phylogeny = PhylogeneticTree::new();

        history.record(0, &organisms, &phylogeny);

        let csv = history.to_csv();
        assert!(csv.contains("time,population"));
        assert!(csv.contains("0,50"));
    }
}
