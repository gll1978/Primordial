//! Statistics tracking for the simulation.

use crate::organism::Organism;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistics snapshot for a simulation step
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stats {
    /// Current simulation time
    pub time: u64,
    /// Total population count
    pub population: usize,
    /// Maximum generation reached
    pub generation_max: u16,
    /// Mean brain complexity (hidden neurons)
    pub brain_mean: f32,
    /// Maximum brain complexity
    pub brain_max: usize,
    /// Mean energy across organisms
    pub energy_mean: f32,
    /// Mean health across organisms
    pub health_mean: f32,
    /// Mean age across organisms
    pub age_mean: f32,
    /// Number of distinct lineages
    pub lineage_count: usize,
    /// Total food in the world
    pub total_food: f32,
    /// Births this step
    pub births: usize,
    /// Deaths this step
    pub deaths: usize,
    /// Steps per second (performance)
    pub steps_per_second: f32,
    /// Learning: mean efficiency across organisms with learning enabled
    pub learning_efficiency: f32,
    /// Learning: total updates across all organisms
    pub learning_updates: u64,
    /// Learning: mean lifetime reward
    pub learning_reward_mean: f32,
}

impl Stats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Update stats from current simulation state
    pub fn update(&mut self, organisms: &[Organism], total_food: f32) {
        self.population = organisms.iter().filter(|o| o.is_alive()).count();

        if self.population == 0 {
            self.generation_max = 0;
            self.brain_mean = 0.0;
            self.brain_max = 0;
            self.energy_mean = 0.0;
            self.health_mean = 0.0;
            self.age_mean = 0.0;
            self.lineage_count = 0;
        } else {
            let alive: Vec<&Organism> = organisms.iter().filter(|o| o.is_alive()).collect();

            // Generation
            self.generation_max = alive.iter().map(|o| o.generation).max().unwrap_or(0);

            // Brain stats
            let brain_complexities: Vec<usize> = alive.iter().map(|o| o.brain.complexity()).collect();
            self.brain_mean = brain_complexities.iter().sum::<usize>() as f32 / alive.len() as f32;
            self.brain_max = brain_complexities.into_iter().max().unwrap_or(0);

            // Energy
            self.energy_mean = alive.iter().map(|o| o.energy).sum::<f32>() / alive.len() as f32;

            // Health
            self.health_mean = alive.iter().map(|o| o.health).sum::<f32>() / alive.len() as f32;

            // Age
            self.age_mean = alive.iter().map(|o| o.age as f32).sum::<f32>() / alive.len() as f32;

            // Lineages
            let lineages: std::collections::HashSet<_> =
                alive.iter().map(|o| o.lineage_id).collect();
            self.lineage_count = lineages.len();

            // Learning stats
            let mut eff_sum = 0.0f32;
            let mut eff_count = 0usize;
            let mut total_updates = 0u64;
            let mut reward_sum = 0.0f32;

            for org in &alive {
                if let Some(stats) = org.brain.learning_stats() {
                    eff_sum += stats.efficiency;
                    eff_count += 1;
                    total_updates += stats.update_count;
                }
                reward_sum += org.total_lifetime_reward;
            }

            self.learning_efficiency = if eff_count > 0 { eff_sum / eff_count as f32 } else { 0.0 };
            self.learning_updates = total_updates;
            self.learning_reward_mean = reward_sum / alive.len() as f32;
        }

        self.total_food = total_food;
    }

    /// Save stats to JSON file
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load stats from JSON file
    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Format stats as a one-line summary
    pub fn summary(&self) -> String {
        if self.learning_updates > 0 {
            format!(
                "T:{:6} | Pop:{:5} | Gen:{:3} | Brain:{:.1} | Energy:{:.0} | Food:{:.0} | Lrn:{:.2} Upd:{} Rwd:{:.1}",
                self.time,
                self.population,
                self.generation_max,
                self.brain_mean,
                self.energy_mean,
                self.total_food,
                self.learning_efficiency,
                self.learning_updates,
                self.learning_reward_mean,
            )
        } else {
            format!(
                "T:{:6} | Pop:{:5} | Gen:{:3} | Brain:{:.1} | Energy:{:.0} | Food:{:.0}",
                self.time,
                self.population,
                self.generation_max,
                self.brain_mean,
                self.energy_mean,
                self.total_food
            )
        }
    }
}

/// Historical statistics tracker
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StatsHistory {
    /// All recorded stats snapshots
    pub snapshots: Vec<Stats>,
    /// Recording interval
    pub interval: u64,
}

impl StatsHistory {
    /// Create new history with recording interval
    pub fn new(interval: u64) -> Self {
        Self {
            snapshots: Vec::new(),
            interval,
        }
    }

    /// Record a stats snapshot
    pub fn record(&mut self, stats: Stats) {
        self.snapshots.push(stats);
    }

    /// Get stats at a specific time (approximate)
    pub fn get_at(&self, time: u64) -> Option<&Stats> {
        let index = (time / self.interval) as usize;
        self.snapshots.get(index)
    }

    /// Get population over time
    pub fn population_series(&self) -> Vec<(u64, usize)> {
        self.snapshots
            .iter()
            .map(|s| (s.time, s.population))
            .collect()
    }

    /// Get brain complexity over time
    pub fn brain_series(&self) -> Vec<(u64, f32)> {
        self.snapshots
            .iter()
            .map(|s| (s.time, s.brain_mean))
            .collect()
    }

    /// Get generation max over time
    pub fn generation_series(&self) -> Vec<(u64, u16)> {
        self.snapshots
            .iter()
            .map(|s| (s.time, s.generation_max))
            .collect()
    }

    /// Save history to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)
    }

    /// Load history from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Lineage tracker for evolutionary analysis
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LineageTracker {
    /// Lineage ID -> (founder generation, current population, max generation reached)
    pub lineages: HashMap<u32, LineageStats>,
    /// Next lineage ID
    pub next_lineage_id: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LineageStats {
    pub founder_time: u64,
    pub current_population: usize,
    pub max_generation: u16,
    pub total_offspring: u64,
    pub extinct: bool,
}

impl LineageTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new lineage
    pub fn register_lineage(&mut self, time: u64) -> u32 {
        let id = self.next_lineage_id;
        self.next_lineage_id += 1;

        self.lineages.insert(
            id,
            LineageStats {
                founder_time: time,
                current_population: 1,
                max_generation: 0,
                total_offspring: 0,
                extinct: false,
            },
        );

        id
    }

    /// Update lineage statistics
    pub fn update(&mut self, organisms: &[Organism]) {
        // Reset current populations
        for stats in self.lineages.values_mut() {
            stats.current_population = 0;
        }

        // Count current populations
        for org in organisms.iter().filter(|o| o.is_alive()) {
            if let Some(stats) = self.lineages.get_mut(&org.lineage_id) {
                stats.current_population += 1;
                stats.max_generation = stats.max_generation.max(org.generation);
            }
        }

        // Mark extinct lineages
        for stats in self.lineages.values_mut() {
            if stats.current_population == 0 && !stats.extinct {
                stats.extinct = true;
            }
        }
    }

    /// Get surviving lineages count
    pub fn surviving_count(&self) -> usize {
        self.lineages.values().filter(|s| !s.extinct).count()
    }

    /// Get dominant lineage (highest population)
    pub fn dominant_lineage(&self) -> Option<(u32, &LineageStats)> {
        self.lineages
            .iter()
            .filter(|(_, s)| !s.extinct)
            .max_by_key(|(_, s)| s.current_population)
            .map(|(&id, stats)| (id, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_stats_update() {
        let config = Config::default();
        let organisms = vec![
            Organism::new(1, 1, 10, 10, &config),
            Organism::new(2, 1, 20, 20, &config),
            Organism::new(3, 2, 30, 30, &config),
        ];

        let mut stats = Stats::new();
        stats.update(&organisms, 1000.0);

        assert_eq!(stats.population, 3);
        assert_eq!(stats.lineage_count, 2);
    }

    #[test]
    fn test_stats_history() {
        let mut history = StatsHistory::new(10);

        for i in 0..5 {
            let mut stats = Stats::new();
            stats.time = i * 10;
            stats.population = (i + 1) as usize * 100;
            history.record(stats);
        }

        let series = history.population_series();
        assert_eq!(series.len(), 5);
        assert_eq!(series[0], (0, 100));
        assert_eq!(series[4], (40, 500));
    }

    #[test]
    fn test_lineage_tracker() {
        let mut tracker = LineageTracker::new();

        let id1 = tracker.register_lineage(0);
        let id2 = tracker.register_lineage(0);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(tracker.surviving_count(), 2);
    }
}
