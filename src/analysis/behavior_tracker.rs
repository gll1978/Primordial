//! Behavior tracking for foraging memory analysis.
//!
//! Tracks organism movement patterns and patch visits to measure
//! spatial memory efficiency.

use std::collections::{HashMap, VecDeque};

/// Maximum number of patch visits to track per organism (ring buffer)
const MAX_PATCH_VISITS: usize = 500;

/// Maximum number of positions to track per organism (ring buffer)
const MAX_POSITIONS: usize = 64;

/// Tracks behavior of a single organism
#[derive(Debug, Clone)]
pub struct BehaviorTracker {
    pub organism_id: u64,
    pub positions: VecDeque<(u8, u8)>,
    pub patch_visits: VecDeque<(u64, usize)>, // (time, patch_index) - ring buffer
    pub unique_cells_visited: u16,
    pub total_moves: u32,
    pub total_eats: u32,
    pub alive: bool,
}

impl BehaviorTracker {
    pub fn new(organism_id: u64) -> Self {
        Self {
            organism_id,
            positions: VecDeque::with_capacity(MAX_POSITIONS),
            patch_visits: VecDeque::with_capacity(MAX_PATCH_VISITS),
            unique_cells_visited: 0,
            total_moves: 0,
            total_eats: 0,
            alive: true,
        }
    }

    pub fn track_movement(&mut self, x: u8, y: u8) {
        self.total_moves += 1;
        // Ring buffer: remove oldest if at capacity (O(1) with VecDeque)
        if self.positions.len() >= MAX_POSITIONS {
            self.positions.pop_front();
        }
        self.positions.push_back((x, y));
    }

    pub fn track_patch_visit(&mut self, time: u64, patch_index: usize) {
        self.total_eats += 1;
        // Ring buffer: remove oldest if at capacity (O(1) with VecDeque)
        if self.patch_visits.len() >= MAX_PATCH_VISITS {
            self.patch_visits.pop_front();
        }
        self.patch_visits.push_back((time, patch_index));
    }

    pub fn mark_dead(&mut self) {
        self.alive = false;
    }

    /// Memory efficiency: ratio of patch revisits to total patch visits.
    /// Higher means the organism returns to known patches (has memory).
    pub fn memory_efficiency(&self) -> f32 {
        if self.patch_visits.len() < 2 {
            return 0.0;
        }
        let mut visited_patches: HashMap<usize, u32> = HashMap::new();
        for &(_, patch_idx) in &self.patch_visits {
            *visited_patches.entry(patch_idx).or_insert(0) += 1;
        }
        let revisits: u32 = visited_patches.values().filter(|&&v| v > 1).map(|v| v - 1).sum();
        revisits as f32 / self.patch_visits.len() as f32
    }

    /// Exploration efficiency: unique cells / total moves
    pub fn exploration_efficiency(&self) -> f32 {
        if self.total_moves == 0 {
            return 0.0;
        }
        // Count unique positions in recent history
        let mut unique = std::collections::HashSet::new();
        for pos in &self.positions {
            unique.insert(*pos);
        }
        unique.len() as f32 / self.positions.len() as f32
    }
}

/// Manages tracking for multiple organisms
pub struct BehaviorTrackerManager {
    pub trackers: HashMap<u64, BehaviorTracker>,
    pub max_tracked: usize,
    pub sample_rate: u64,
}

impl BehaviorTrackerManager {
    pub fn new(max_tracked: usize, sample_rate: u64) -> Self {
        Self {
            trackers: HashMap::with_capacity(max_tracked),
            max_tracked,
            sample_rate,
        }
    }

    /// Start tracking an organism (if under limit)
    pub fn start_tracking(&mut self, organism_id: u64) {
        if self.trackers.len() < self.max_tracked {
            self.trackers.entry(organism_id).or_insert_with(|| BehaviorTracker::new(organism_id));
        }
    }

    pub fn track_movement(&mut self, organism_id: u64, x: u8, y: u8) {
        if let Some(tracker) = self.trackers.get_mut(&organism_id) {
            tracker.track_movement(x, y);
        }
    }

    pub fn track_patch_visit(&mut self, organism_id: u64, time: u64, patch_index: usize) {
        if let Some(tracker) = self.trackers.get_mut(&organism_id) {
            tracker.track_patch_visit(time, patch_index);
        }
    }

    pub fn mark_dead(&mut self, organism_id: u64) {
        if let Some(tracker) = self.trackers.get_mut(&organism_id) {
            tracker.mark_dead();
        }
    }

    /// Cleanup dead trackers periodically
    pub fn cleanup_dead(&mut self) {
        self.trackers.retain(|_, t| t.alive);
    }

    /// Average memory efficiency across all tracked organisms
    pub fn avg_memory_efficiency(&self) -> f32 {
        let active: Vec<_> = self.trackers.values().filter(|t| t.alive).collect();
        if active.is_empty() {
            return 0.0;
        }
        let sum: f32 = active.iter().map(|t| t.memory_efficiency()).sum();
        sum / active.len() as f32
    }

    /// Average exploration efficiency
    pub fn avg_exploration_efficiency(&self) -> f32 {
        let active: Vec<_> = self.trackers.values().filter(|t| t.alive).collect();
        if active.is_empty() {
            return 0.0;
        }
        let sum: f32 = active.iter().map(|t| t.exploration_efficiency()).sum();
        sum / active.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_tracker_movement() {
        let mut tracker = BehaviorTracker::new(1);
        tracker.track_movement(10, 20);
        tracker.track_movement(11, 20);
        assert_eq!(tracker.total_moves, 2);
        assert_eq!(tracker.positions.len(), 2);
    }

    #[test]
    fn test_memory_efficiency() {
        let mut tracker = BehaviorTracker::new(1);
        // Visit patch 0 three times, patch 1 once
        tracker.track_patch_visit(0, 0);
        tracker.track_patch_visit(10, 0);
        tracker.track_patch_visit(20, 0);
        tracker.track_patch_visit(30, 1);
        let eff = tracker.memory_efficiency();
        assert!(eff > 0.0); // Should have revisits
    }

    #[test]
    fn test_manager() {
        let mut mgr = BehaviorTrackerManager::new(10, 10);
        mgr.start_tracking(1);
        mgr.track_movement(1, 5, 5);
        mgr.track_patch_visit(1, 0, 0);
        mgr.mark_dead(1);
        mgr.cleanup_dead();
        assert!(mgr.trackers.is_empty());
    }
}
