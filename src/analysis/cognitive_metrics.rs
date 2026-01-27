//! Cognitive metrics: correlate brain size with memory efficiency.

use crate::analysis::behavior_tracker::BehaviorTrackerManager;

/// Aggregate cognitive metrics
#[derive(Debug, Clone, Default)]
pub struct CognitiveMetrics {
    pub avg_brain_size: f32,
    pub avg_memory_efficiency: f32,
    pub avg_exploration_efficiency: f32,
    pub brain_memory_correlation: f32,
    pub sample_count: usize,
}

impl CognitiveMetrics {
    /// Compute metrics from behavior tracker and organism brain sizes.
    /// `brain_sizes` maps organism_id -> brain layer count (f32).
    pub fn compute(
        tracker: &BehaviorTrackerManager,
        brain_sizes: &std::collections::HashMap<u64, f32>,
    ) -> Self {
        let mut brain_vals = Vec::new();
        let mut mem_vals = Vec::new();

        for (id, t) in &tracker.trackers {
            if !t.alive {
                continue;
            }
            if let Some(&bs) = brain_sizes.get(id) {
                brain_vals.push(bs);
                mem_vals.push(t.memory_efficiency());
            }
        }

        let n = brain_vals.len();
        if n < 2 {
            return Self {
                sample_count: n,
                avg_memory_efficiency: tracker.avg_memory_efficiency(),
                avg_exploration_efficiency: tracker.avg_exploration_efficiency(),
                ..Default::default()
            };
        }

        let avg_b: f32 = brain_vals.iter().sum::<f32>() / n as f32;
        let avg_m: f32 = mem_vals.iter().sum::<f32>() / n as f32;

        // Pearson correlation
        let mut cov = 0.0f32;
        let mut var_b = 0.0f32;
        let mut var_m = 0.0f32;
        for i in 0..n {
            let db = brain_vals[i] - avg_b;
            let dm = mem_vals[i] - avg_m;
            cov += db * dm;
            var_b += db * db;
            var_m += dm * dm;
        }
        let denom = (var_b * var_m).sqrt();
        let corr = if denom > 1e-8 { cov / denom } else { 0.0 };

        Self {
            avg_brain_size: avg_b,
            avg_memory_efficiency: avg_m,
            avg_exploration_efficiency: tracker.avg_exploration_efficiency(),
            brain_memory_correlation: corr,
            sample_count: n,
        }
    }
}

impl std::fmt::Display for CognitiveMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cognitive: brain={:.2}, mem_eff={:.3}, expl_eff={:.3}, corr={:.3} (n={})",
            self.avg_brain_size,
            self.avg_memory_efficiency,
            self.avg_exploration_efficiency,
            self.brain_memory_correlation,
            self.sample_count,
        )
    }
}
