//! Memory consolidation: short-term weight changes → long-term stable changes.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Maximum number of long-term changes to keep per organism
const MAX_LONG_TERM_CHANGES: usize = 1000;

/// A single weight change record
#[derive(Clone, Debug)]
pub struct WeightChange {
    pub layer: usize,
    pub i: usize,
    pub j: usize,
    pub delta: f32,
    pub timestamp: u64,
    pub importance: f32,
}

/// Accumulated long-term change for a specific weight
#[derive(Clone, Debug, Default)]
struct LongTermChange {
    accumulated_delta: f32,
    count: u32,
    total_importance: f32,
}

/// Statistics about consolidation
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConsolidationStats {
    pub short_term_count: usize,
    pub long_term_count: usize,
    pub consolidations_performed: u64,
}

/// Consolidates short-term weight changes into long-term stable changes
#[derive(Clone, Debug)]
pub struct MemoryConsolidator {
    short_term_changes: Vec<WeightChange>,
    long_term_changes: HashMap<(usize, usize, usize), LongTermChange>,
    pub consolidation_threshold: f32,
    pub working_memory_size: usize,
    consolidations_performed: u64,
}

impl MemoryConsolidator {
    pub fn new(consolidation_threshold: f32, working_memory_size: usize) -> Self {
        Self {
            short_term_changes: Vec::new(),
            long_term_changes: HashMap::new(),
            consolidation_threshold,
            working_memory_size,
            consolidations_performed: 0,
        }
    }

    /// Record a weight change in short-term memory
    pub fn record_change(&mut self, change: WeightChange) {
        self.short_term_changes.push(change);

        // If short-term is full, auto-consolidate
        if self.short_term_changes.len() >= self.working_memory_size {
            self.consolidate();
        }
    }

    /// Consolidate short-term changes into long-term memory
    /// Groups by weight position, computes weighted average delta
    pub fn consolidate(&mut self) {
        if self.short_term_changes.is_empty() {
            return;
        }

        for change in self.short_term_changes.drain(..) {
            let key = (change.layer, change.i, change.j);
            let entry = self.long_term_changes.entry(key).or_default();

            // Weighted average: accumulate importance-weighted delta
            entry.accumulated_delta += change.delta * change.importance;
            entry.total_importance += change.importance;
            entry.count += 1;
        }

        self.consolidations_performed += 1;

        // Prune long-term changes if over limit (keep most important)
        if self.long_term_changes.len() > MAX_LONG_TERM_CHANGES {
            self.prune_long_term_changes();
        }
    }

    /// Remove least important long-term changes to stay under limit
    fn prune_long_term_changes(&mut self) {
        if self.long_term_changes.len() <= MAX_LONG_TERM_CHANGES {
            return;
        }

        // Collect entries with their importance scores
        let mut entries: Vec<_> = self
            .long_term_changes
            .iter()
            .map(|(k, v)| (*k, v.total_importance))
            .collect();

        // Sort by importance (ascending, so least important first)
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove least important entries to get back under limit
        let to_remove = self.long_term_changes.len() - MAX_LONG_TERM_CHANGES;
        for (key, _) in entries.into_iter().take(to_remove) {
            self.long_term_changes.remove(&key);
        }
    }

    /// Get consolidated changes that exceed the threshold
    /// Returns (layer, i, j, delta) tuples
    pub fn get_consolidated_changes(&mut self) -> Vec<(usize, usize, usize, f32)> {
        // Consolidate any remaining short-term changes first
        self.consolidate();

        let threshold = self.consolidation_threshold;
        let results: Vec<_> = self
            .long_term_changes
            .iter()
            .filter_map(|(&(layer, i, j), change)| {
                if change.total_importance > 0.0 {
                    let avg_delta = change.accumulated_delta / change.total_importance;
                    if avg_delta.abs() >= threshold {
                        return Some((layer, i, j, avg_delta));
                    }
                }
                None
            })
            .collect();

        results
    }

    /// Get consolidation statistics
    pub fn stats(&self) -> ConsolidationStats {
        ConsolidationStats {
            short_term_count: self.short_term_changes.len(),
            long_term_count: self.long_term_changes.len(),
            consolidations_performed: self.consolidations_performed,
        }
    }

    /// Clear all memory
    pub fn clear(&mut self) {
        self.short_term_changes.clear();
        self.long_term_changes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidation_basic() {
        let mut consolidator = MemoryConsolidator::new(0.05, 100);

        // Record several changes to the same weight
        for i in 0..5 {
            consolidator.record_change(WeightChange {
                layer: 0,
                i: 0,
                j: 0,
                delta: 0.1,
                timestamp: i,
                importance: 1.0,
            });
        }

        let changes = consolidator.get_consolidated_changes();
        assert!(!changes.is_empty());

        // Average delta should be 0.1 (all same)
        let (_, _, _, delta) = changes[0];
        assert!((delta - 0.1).abs() < 1e-4);
    }

    #[test]
    fn test_threshold_filtering() {
        let mut consolidator = MemoryConsolidator::new(0.5, 100);

        // Small change below threshold
        consolidator.record_change(WeightChange {
            layer: 0,
            i: 0,
            j: 0,
            delta: 0.01,
            timestamp: 0,
            importance: 1.0,
        });

        let changes = consolidator.get_consolidated_changes();
        assert!(changes.is_empty(), "Small changes should be filtered out");
    }

    #[test]
    fn test_auto_consolidation() {
        let mut consolidator = MemoryConsolidator::new(0.01, 5);

        // Fill past working memory size to trigger auto-consolidation
        for i in 0..6 {
            consolidator.record_change(WeightChange {
                layer: 0,
                i: 0,
                j: 0,
                delta: 0.2,
                timestamp: i,
                importance: 1.0,
            });
        }

        let stats = consolidator.stats();
        assert!(stats.consolidations_performed > 0);
    }
}
