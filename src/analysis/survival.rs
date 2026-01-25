//! Survival analysis and death tracking.

use crate::organism::{DeathCause, Organism};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Record of a single death event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeathRecord {
    pub organism_id: u64,
    pub birth_time: u64,
    pub death_time: u64,
    pub lifespan: u64,
    pub cause: DeathCause,
    pub lineage_id: u32,
    pub generation: u16,
    pub final_energy: f32,
    pub final_size: f32,
    pub brain_complexity: usize,
    pub offspring_count: u16,
    pub kills: u16,
    pub food_eaten: u32,
}

/// Survival statistics summary
#[derive(Clone, Debug, Default)]
pub struct SurvivalStats {
    pub total_deaths: usize,
    pub deaths_by_starvation: usize,
    pub deaths_by_predation: usize,
    pub deaths_by_old_age: usize,
    pub mean_lifespan: f32,
    pub max_lifespan: u64,
    pub mean_offspring: f32,
    pub mean_kills: f32,
    pub survival_rate_by_generation: HashMap<u16, f32>,
}

impl SurvivalStats {
    /// Format as human-readable string
    pub fn to_string(&self) -> String {
        format!(
            "Deaths: {} (Starve: {}, Pred: {}, Age: {}), Mean lifespan: {:.1}, Max: {}",
            self.total_deaths,
            self.deaths_by_starvation,
            self.deaths_by_predation,
            self.deaths_by_old_age,
            self.mean_lifespan,
            self.max_lifespan
        )
    }
}

/// Survival analyzer for tracking and analyzing deaths
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SurvivalAnalyzer {
    /// All death records
    pub records: Vec<DeathRecord>,
    /// Deaths by cause
    pub by_cause: HashMap<String, usize>,
    /// Total organisms tracked
    pub total_tracked: usize,
}

impl SurvivalAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a death from an organism
    pub fn record_death(&mut self, organism: &Organism, birth_time: u64, death_time: u64) {
        let cause = organism.cause_of_death.unwrap_or(DeathCause::Starvation);
        let lifespan = death_time.saturating_sub(birth_time);

        let record = DeathRecord {
            organism_id: organism.id,
            birth_time,
            death_time,
            lifespan,
            cause,
            lineage_id: organism.lineage_id,
            generation: organism.generation,
            final_energy: organism.energy,
            final_size: organism.size,
            brain_complexity: organism.brain.complexity(),
            offspring_count: organism.offspring_count,
            kills: organism.kills,
            food_eaten: organism.food_eaten,
        };

        self.records.push(record);
        self.total_tracked += 1;

        // Update by_cause counter
        let cause_str = match cause {
            DeathCause::Starvation => "starvation",
            DeathCause::Predation => "predation",
            DeathCause::OldAge => "old_age",
        };
        *self.by_cause.entry(cause_str.to_string()).or_insert(0) += 1;
    }

    /// Calculate survival statistics
    pub fn calculate_stats(&self) -> SurvivalStats {
        if self.records.is_empty() {
            return SurvivalStats::default();
        }

        let total = self.records.len();

        let deaths_by_starvation = self.records.iter()
            .filter(|r| matches!(r.cause, DeathCause::Starvation))
            .count();

        let deaths_by_predation = self.records.iter()
            .filter(|r| matches!(r.cause, DeathCause::Predation))
            .count();

        let deaths_by_old_age = self.records.iter()
            .filter(|r| matches!(r.cause, DeathCause::OldAge))
            .count();

        let total_lifespan: u64 = self.records.iter().map(|r| r.lifespan).sum();
        let mean_lifespan = total_lifespan as f32 / total as f32;

        let max_lifespan = self.records.iter().map(|r| r.lifespan).max().unwrap_or(0);

        let total_offspring: u32 = self.records.iter().map(|r| r.offspring_count as u32).sum();
        let mean_offspring = total_offspring as f32 / total as f32;

        let total_kills: u32 = self.records.iter().map(|r| r.kills as u32).sum();
        let mean_kills = total_kills as f32 / total as f32;

        // Calculate survival rate by generation
        let mut gen_births: HashMap<u16, usize> = HashMap::new();
        let mut gen_deaths: HashMap<u16, usize> = HashMap::new();

        for record in &self.records {
            *gen_births.entry(record.generation).or_insert(0) += 1;
            *gen_deaths.entry(record.generation).or_insert(0) += 1;
        }

        let survival_rate_by_generation: HashMap<u16, f32> = gen_births
            .iter()
            .map(|(&gen, &births)| {
                let deaths = gen_deaths.get(&gen).copied().unwrap_or(0);
                let rate = if births > 0 {
                    1.0 - (deaths as f32 / births as f32)
                } else {
                    0.0
                };
                (gen, rate.max(0.0))
            })
            .collect();

        SurvivalStats {
            total_deaths: total,
            deaths_by_starvation,
            deaths_by_predation,
            deaths_by_old_age,
            mean_lifespan,
            max_lifespan,
            mean_offspring,
            mean_kills,
            survival_rate_by_generation,
        }
    }

    /// Get death rate by cause
    pub fn death_rate_by_cause(&self) -> HashMap<String, f32> {
        if self.records.is_empty() {
            return HashMap::new();
        }

        let total = self.records.len() as f32;

        self.by_cause
            .iter()
            .map(|(cause, &count)| (cause.clone(), count as f32 / total))
            .collect()
    }

    /// Get lifespan distribution (bucket counts)
    pub fn lifespan_distribution(&self, bucket_size: u64) -> HashMap<u64, usize> {
        let mut distribution = HashMap::new();

        for record in &self.records {
            let bucket = (record.lifespan / bucket_size) * bucket_size;
            *distribution.entry(bucket).or_insert(0) += 1;
        }

        distribution
    }

    /// Filter records by generation range
    pub fn filter_by_generation(&self, min_gen: u16, max_gen: u16) -> Vec<&DeathRecord> {
        self.records
            .iter()
            .filter(|r| r.generation >= min_gen && r.generation <= max_gen)
            .collect()
    }

    /// Filter records by time range
    pub fn filter_by_time(&self, min_time: u64, max_time: u64) -> Vec<&DeathRecord> {
        self.records
            .iter()
            .filter(|r| r.death_time >= min_time && r.death_time <= max_time)
            .collect()
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::from(
            "organism_id,birth_time,death_time,lifespan,cause,lineage_id,generation,final_energy,final_size,brain_complexity,offspring_count,kills,food_eaten\n"
        );

        for record in &self.records {
            let cause_str = match record.cause {
                DeathCause::Starvation => "starvation",
                DeathCause::Predation => "predation",
                DeathCause::OldAge => "old_age",
            };

            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{:.2},{:.2},{},{},{},{}\n",
                record.organism_id,
                record.birth_time,
                record.death_time,
                record.lifespan,
                cause_str,
                record.lineage_id,
                record.generation,
                record.final_energy,
                record.final_size,
                record.brain_complexity,
                record.offspring_count,
                record.kills,
                record.food_eaten,
            ));
        }

        csv
    }

    /// Clear old records to save memory
    pub fn prune(&mut self, keep_last: usize) {
        if self.records.len() > keep_last {
            let start = self.records.len() - keep_last;
            self.records = self.records[start..].to_vec();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn create_test_organism(id: u64, generation: u16) -> Organism {
        let config = Config::default();
        let mut org = Organism::new(id, 1, 10, 10, &config);
        org.generation = generation;
        org.offspring_count = 2;
        org.kills = 1;
        org.food_eaten = 50;
        org.cause_of_death = Some(DeathCause::Starvation);
        org
    }

    #[test]
    fn test_record_death() {
        let mut analyzer = SurvivalAnalyzer::new();
        let org = create_test_organism(1, 5);

        analyzer.record_death(&org, 0, 100);

        assert_eq!(analyzer.records.len(), 1);
        assert_eq!(analyzer.records[0].lifespan, 100);
        assert_eq!(analyzer.records[0].generation, 5);
    }

    #[test]
    fn test_survival_stats() {
        let mut analyzer = SurvivalAnalyzer::new();

        for i in 0..10 {
            let mut org = create_test_organism(i, i as u16 % 3);
            org.cause_of_death = if i % 3 == 0 {
                Some(DeathCause::Starvation)
            } else if i % 3 == 1 {
                Some(DeathCause::Predation)
            } else {
                Some(DeathCause::OldAge)
            };

            analyzer.record_death(&org, 0, (i + 1) * 100);
        }

        let stats = analyzer.calculate_stats();

        assert_eq!(stats.total_deaths, 10);
        assert!(stats.mean_lifespan > 0.0);
        assert!(stats.max_lifespan > 0);
    }

    #[test]
    fn test_death_rate_by_cause() {
        let mut analyzer = SurvivalAnalyzer::new();

        // 3 starvation, 1 predation
        for i in 0..4 {
            let mut org = create_test_organism(i, 0);
            org.cause_of_death = if i < 3 {
                Some(DeathCause::Starvation)
            } else {
                Some(DeathCause::Predation)
            };
            analyzer.record_death(&org, 0, 100);
        }

        let rates = analyzer.death_rate_by_cause();

        assert!((rates["starvation"] - 0.75).abs() < 0.01);
        assert!((rates["predation"] - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_lifespan_distribution() {
        let mut analyzer = SurvivalAnalyzer::new();

        // Create organisms with lifespans 50, 150, 250, 350
        for i in 0..4 {
            let org = create_test_organism(i, 0);
            analyzer.record_death(&org, 0, (i + 1) * 100 - 50);
        }

        let dist = analyzer.lifespan_distribution(100);

        // Should have entries for buckets 0, 100, 200, 300
        assert!(dist.len() >= 2);
    }

    #[test]
    fn test_csv_export() {
        let mut analyzer = SurvivalAnalyzer::new();
        let org = create_test_organism(1, 0);
        analyzer.record_death(&org, 0, 100);

        let csv = analyzer.to_csv();

        assert!(csv.contains("organism_id"));
        assert!(csv.contains("1,0,100,100"));
    }

    #[test]
    fn test_prune() {
        let mut analyzer = SurvivalAnalyzer::new();

        for i in 0..100 {
            let org = create_test_organism(i, 0);
            analyzer.record_death(&org, 0, i * 10);
        }

        assert_eq!(analyzer.records.len(), 100);

        analyzer.prune(50);

        assert_eq!(analyzer.records.len(), 50);
    }
}
