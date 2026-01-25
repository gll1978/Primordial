//! Phylogenetic tree tracking for complete lineage history.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A node in the phylogenetic tree representing one organism
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeNode {
    /// Organism ID
    pub organism_id: u64,
    /// First parent ID (for sexual reproduction)
    pub parent1_id: Option<u64>,
    /// Second parent ID (for sexual reproduction)
    pub parent2_id: Option<u64>,
    /// Birth time (simulation step)
    pub birth_time: u64,
    /// Death time (None if still alive)
    pub death_time: Option<u64>,

    // Phenotype snapshot at birth
    /// Brain complexity at birth
    pub brain_complexity: usize,
    /// Energy at birth
    pub birth_energy: f32,
    /// Size at birth
    pub birth_size: f32,

    // Lifetime achievements
    /// Peak energy achieved
    pub peak_energy: f32,
    /// Number of offspring
    pub offspring_count: u16,
    /// Number of kills (if predator)
    pub kills: u16,

    // Lineage info
    /// Lineage ID
    pub lineage_id: u32,
    /// Generation number
    pub generation: u16,

    // Genomic identity
    /// Hash of neural network weights
    pub genome_hash: u64,
}

impl TreeNode {
    /// Get lifespan (None if still alive)
    pub fn lifespan(&self) -> Option<u64> {
        self.death_time.map(|d| d - self.birth_time)
    }

    /// Check if organism is still alive
    pub fn is_alive(&self) -> bool {
        self.death_time.is_none()
    }
}

/// Complete phylogenetic tree for the simulation
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhylogeneticTree {
    /// All nodes indexed by organism ID
    pub nodes: HashMap<u64, TreeNode>,
    /// Root organism IDs (no parents)
    pub root_ids: Vec<u64>,
    /// Maximum generation reached
    pub max_generation: u16,
    /// Total organisms tracked
    pub total_organisms: u64,
    /// Total deaths recorded
    pub total_deaths: u64,
}

impl PhylogeneticTree {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record birth of a new organism
    pub fn record_birth(
        &mut self,
        organism_id: u64,
        parent1_id: Option<u64>,
        parent2_id: Option<u64>,
        time: u64,
        brain_complexity: usize,
        birth_energy: f32,
        birth_size: f32,
        lineage_id: u32,
        generation: u16,
        genome_hash: u64,
    ) {
        let node = TreeNode {
            organism_id,
            parent1_id,
            parent2_id,
            birth_time: time,
            death_time: None,
            brain_complexity,
            birth_energy,
            birth_size,
            peak_energy: birth_energy,
            offspring_count: 0,
            kills: 0,
            lineage_id,
            generation,
            genome_hash,
        };

        self.nodes.insert(organism_id, node);
        self.total_organisms += 1;

        // Track max generation
        if generation > self.max_generation {
            self.max_generation = generation;
        }

        // If no parents, this is a root
        if parent1_id.is_none() && parent2_id.is_none() {
            self.root_ids.push(organism_id);
        }
    }

    /// Record death of an organism
    pub fn record_death(&mut self, organism_id: u64, time: u64) {
        if let Some(node) = self.nodes.get_mut(&organism_id) {
            node.death_time = Some(time);
            self.total_deaths += 1;
        }
    }

    /// Update peak energy for an organism
    pub fn update_peak_energy(&mut self, organism_id: u64, energy: f32) {
        if let Some(node) = self.nodes.get_mut(&organism_id) {
            if energy > node.peak_energy {
                node.peak_energy = energy;
            }
        }
    }

    /// Increment offspring count for parents
    pub fn record_offspring(&mut self, parent1_id: Option<u64>, parent2_id: Option<u64>) {
        if let Some(id) = parent1_id {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.offspring_count += 1;
            }
        }
        if let Some(id) = parent2_id {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.offspring_count += 1;
            }
        }
    }

    /// Record a kill for an organism
    pub fn record_kill(&mut self, organism_id: u64) {
        if let Some(node) = self.nodes.get_mut(&organism_id) {
            node.kills += 1;
        }
    }

    /// Get all ancestors of an organism (following parent1 chain)
    pub fn get_ancestors(&self, organism_id: u64) -> Vec<u64> {
        let mut ancestors = Vec::new();
        let mut current = organism_id;

        while let Some(node) = self.nodes.get(&current) {
            if let Some(parent1_id) = node.parent1_id {
                ancestors.push(parent1_id);
                current = parent1_id;
            } else {
                break;
            }
        }

        ancestors
    }

    /// Get all ancestors including both parents (full tree traversal)
    pub fn get_all_ancestors(&self, organism_id: u64) -> HashSet<u64> {
        let mut ancestors = HashSet::new();
        let mut to_visit = vec![organism_id];

        while let Some(current) = to_visit.pop() {
            if let Some(node) = self.nodes.get(&current) {
                if let Some(p1) = node.parent1_id {
                    if ancestors.insert(p1) {
                        to_visit.push(p1);
                    }
                }
                if let Some(p2) = node.parent2_id {
                    if ancestors.insert(p2) {
                        to_visit.push(p2);
                    }
                }
            }
        }

        ancestors
    }

    /// Find most recent common ancestor (MRCA) of two organisms
    pub fn common_ancestor(&self, id1: u64, id2: u64) -> Option<u64> {
        let ancestors1 = self.get_all_ancestors(id1);

        // Also include id1 itself
        let mut ancestors1_with_self = ancestors1.clone();
        ancestors1_with_self.insert(id1);

        // Walk up from id2 until we find a common ancestor
        let mut to_visit = vec![id2];
        let mut visited = HashSet::new();

        while let Some(current) = to_visit.pop() {
            if ancestors1_with_self.contains(&current) {
                return Some(current);
            }

            if visited.insert(current) {
                if let Some(node) = self.nodes.get(&current) {
                    if let Some(p1) = node.parent1_id {
                        to_visit.push(p1);
                    }
                    if let Some(p2) = node.parent2_id {
                        to_visit.push(p2);
                    }
                }
            }
        }

        // Check if id2 itself is an ancestor of id1
        if ancestors1.contains(&id2) {
            return Some(id2);
        }

        None
    }

    /// Calculate genetic distance (generations to MRCA)
    pub fn genetic_distance(&self, id1: u64, id2: u64) -> Option<usize> {
        if id1 == id2 {
            return Some(0);
        }

        if let Some(mrca) = self.common_ancestor(id1, id2) {
            let dist1 = self.distance_to_ancestor(id1, mrca);
            let dist2 = self.distance_to_ancestor(id2, mrca);

            match (dist1, dist2) {
                (Some(d1), Some(d2)) => Some(d1 + d2),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Calculate distance from organism to a specific ancestor
    fn distance_to_ancestor(&self, organism_id: u64, ancestor_id: u64) -> Option<usize> {
        if organism_id == ancestor_id {
            return Some(0);
        }

        let mut to_visit = vec![(organism_id, 0usize)];
        let mut visited = HashSet::new();

        while let Some((current, distance)) = to_visit.pop() {
            if current == ancestor_id {
                return Some(distance);
            }

            if visited.insert(current) {
                if let Some(node) = self.nodes.get(&current) {
                    if let Some(p1) = node.parent1_id {
                        to_visit.push((p1, distance + 1));
                    }
                    if let Some(p2) = node.parent2_id {
                        to_visit.push((p2, distance + 1));
                    }
                }
            }
        }

        None
    }

    /// Get all children of an organism
    pub fn get_children(&self, organism_id: u64) -> Vec<u64> {
        self.nodes
            .iter()
            .filter(|(_, node)| {
                node.parent1_id == Some(organism_id) || node.parent2_id == Some(organism_id)
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all descendants of an organism
    pub fn get_descendants(&self, organism_id: u64) -> HashSet<u64> {
        let mut descendants = HashSet::new();
        let mut to_visit = self.get_children(organism_id);

        while let Some(current) = to_visit.pop() {
            if descendants.insert(current) {
                to_visit.extend(self.get_children(current));
            }
        }

        descendants
    }

    /// Export tree to Newick format for visualization
    pub fn export_newick(&self, root_id: u64) -> String {
        self.build_newick_recursive(root_id)
    }

    fn build_newick_recursive(&self, node_id: u64) -> String {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return format!("{}", node_id),
        };

        // Find children
        let children = self.get_children(node_id);

        if children.is_empty() {
            // Leaf node
            format!("{}:{}", node_id, node.generation)
        } else {
            // Internal node
            let child_strings: Vec<String> = children
                .iter()
                .map(|&child_id| self.build_newick_recursive(child_id))
                .collect();

            format!("({}){}:{}", child_strings.join(","), node_id, node.generation)
        }
    }

    /// Export all trees to Newick format
    pub fn export_all_newick(&self) -> Vec<String> {
        self.root_ids
            .iter()
            .map(|&root_id| format!("{};", self.export_newick(root_id)))
            .collect()
    }

    /// Get statistics about the tree
    pub fn statistics(&self) -> PhylogenyStatistics {
        let alive_count = self.nodes.values().filter(|n| n.is_alive()).count();
        let dead_count = self.nodes.len() - alive_count;

        let avg_lifespan = if dead_count > 0 {
            let total_lifespan: u64 = self
                .nodes
                .values()
                .filter_map(|n| n.lifespan())
                .sum();
            total_lifespan as f32 / dead_count as f32
        } else {
            0.0
        };

        let avg_offspring = if !self.nodes.is_empty() {
            let total_offspring: u32 = self
                .nodes
                .values()
                .map(|n| n.offspring_count as u32)
                .sum();
            total_offspring as f32 / self.nodes.len() as f32
        } else {
            0.0
        };

        let unique_lineages: HashSet<u32> = self.nodes.values().map(|n| n.lineage_id).collect();

        PhylogenyStatistics {
            total_organisms: self.nodes.len(),
            alive_organisms: alive_count,
            dead_organisms: dead_count,
            root_count: self.root_ids.len(),
            max_generation: self.max_generation,
            unique_lineages: unique_lineages.len(),
            average_lifespan: avg_lifespan,
            average_offspring: avg_offspring,
        }
    }

    /// Prune dead branches to save memory (keep only ancestors of living organisms)
    pub fn prune_dead_branches(&mut self) {
        // Get all ancestors of living organisms
        let living: Vec<u64> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.is_alive())
            .map(|(id, _)| *id)
            .collect();

        let mut to_keep: HashSet<u64> = living.iter().copied().collect();

        // Add all ancestors of living organisms
        for &id in &living {
            to_keep.extend(self.get_all_ancestors(id));
        }

        // Remove nodes not in to_keep
        self.nodes.retain(|id, _| to_keep.contains(id));

        // Update root_ids
        self.root_ids.retain(|id| to_keep.contains(id));
    }
}

/// Statistics about the phylogenetic tree
#[derive(Clone, Debug)]
pub struct PhylogenyStatistics {
    pub total_organisms: usize,
    pub alive_organisms: usize,
    pub dead_organisms: usize,
    pub root_count: usize,
    pub max_generation: u16,
    pub unique_lineages: usize,
    pub average_lifespan: f32,
    pub average_offspring: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tree() -> PhylogeneticTree {
        let mut tree = PhylogeneticTree::new();

        // Create lineage: 0 -> 1 -> 2
        //                   -> 3
        tree.record_birth(0, None, None, 0, 5, 100.0, 1.0, 1, 0, 12345);
        tree.record_birth(1, Some(0), None, 100, 6, 80.0, 1.1, 1, 1, 12346);
        tree.record_birth(2, Some(1), None, 200, 7, 90.0, 1.2, 1, 2, 12347);
        tree.record_birth(3, Some(0), None, 150, 5, 85.0, 1.0, 1, 1, 12348);

        tree
    }

    #[test]
    fn test_tree_creation() {
        let tree = PhylogeneticTree::new();
        assert_eq!(tree.nodes.len(), 0);
        assert_eq!(tree.root_ids.len(), 0);
    }

    #[test]
    fn test_record_birth() {
        let mut tree = PhylogeneticTree::new();

        tree.record_birth(1, None, None, 0, 5, 100.0, 1.0, 1, 0, 12345);

        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.root_ids.len(), 1);
        assert_eq!(tree.root_ids[0], 1);

        let node = tree.nodes.get(&1).unwrap();
        assert_eq!(node.organism_id, 1);
        assert!(node.parent1_id.is_none());
        assert_eq!(node.brain_complexity, 5);
    }

    #[test]
    fn test_record_death() {
        let mut tree = PhylogeneticTree::new();
        tree.record_birth(1, None, None, 0, 5, 100.0, 1.0, 1, 0, 12345);

        assert!(tree.nodes.get(&1).unwrap().is_alive());

        tree.record_death(1, 500);

        let node = tree.nodes.get(&1).unwrap();
        assert!(!node.is_alive());
        assert_eq!(node.death_time, Some(500));
        assert_eq!(node.lifespan(), Some(500));
    }

    #[test]
    fn test_get_ancestors() {
        let tree = create_test_tree();

        let ancestors = tree.get_ancestors(2);
        assert_eq!(ancestors, vec![1, 0]);

        let ancestors = tree.get_ancestors(1);
        assert_eq!(ancestors, vec![0]);

        let ancestors = tree.get_ancestors(0);
        assert!(ancestors.is_empty());
    }

    #[test]
    fn test_common_ancestor() {
        let tree = create_test_tree();

        // Siblings (1 and 3) have common ancestor 0
        let mrca = tree.common_ancestor(1, 3);
        assert_eq!(mrca, Some(0));

        // 2's ancestor is 1, and 3's ancestor is 0
        let mrca = tree.common_ancestor(2, 3);
        assert_eq!(mrca, Some(0));

        // Parent-child
        let mrca = tree.common_ancestor(0, 1);
        assert_eq!(mrca, Some(0));
    }

    #[test]
    fn test_genetic_distance() {
        let tree = create_test_tree();

        // Same organism
        assert_eq!(tree.genetic_distance(1, 1), Some(0));

        // Parent-child
        assert_eq!(tree.genetic_distance(0, 1), Some(1));

        // Grandparent-grandchild
        assert_eq!(tree.genetic_distance(0, 2), Some(2));

        // Siblings (through parent)
        assert_eq!(tree.genetic_distance(1, 3), Some(2)); // 1 -> 0 -> 3
    }

    #[test]
    fn test_get_children() {
        let tree = create_test_tree();

        let children = tree.get_children(0);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&1));
        assert!(children.contains(&3));

        let children = tree.get_children(1);
        assert_eq!(children.len(), 1);
        assert!(children.contains(&2));
    }

    #[test]
    fn test_get_descendants() {
        let tree = create_test_tree();

        let descendants = tree.get_descendants(0);
        assert_eq!(descendants.len(), 3); // 1, 2, 3
        assert!(descendants.contains(&1));
        assert!(descendants.contains(&2));
        assert!(descendants.contains(&3));
    }

    #[test]
    fn test_offspring_tracking() {
        let mut tree = create_test_tree();

        // Initially from creation
        assert_eq!(tree.nodes.get(&0).unwrap().offspring_count, 0);

        // Record offspring
        tree.record_offspring(Some(0), None);
        tree.record_offspring(Some(0), None);

        assert_eq!(tree.nodes.get(&0).unwrap().offspring_count, 2);
    }

    #[test]
    fn test_statistics() {
        let mut tree = create_test_tree();

        // Mark one as dead
        tree.record_death(0, 300);

        let stats = tree.statistics();
        assert_eq!(stats.total_organisms, 4);
        assert_eq!(stats.alive_organisms, 3);
        assert_eq!(stats.dead_organisms, 1);
        assert_eq!(stats.root_count, 1);
        assert_eq!(stats.max_generation, 2);
    }

    #[test]
    fn test_newick_export() {
        let tree = create_test_tree();

        let newick = tree.export_newick(0);
        // Should contain node IDs and generations
        assert!(newick.contains("0:0"));
        assert!(newick.contains("1:1"));
        assert!(newick.contains("2:2"));
        assert!(newick.contains("3:1"));
    }

    #[test]
    fn test_prune_dead_branches() {
        let mut tree = create_test_tree();

        // Mark all except 2 as dead
        tree.record_death(0, 100);
        tree.record_death(1, 200);
        tree.record_death(3, 150);

        // Prune - should keep 0, 1, 2 (ancestors of living 2)
        tree.prune_dead_branches();

        assert!(tree.nodes.contains_key(&0)); // ancestor of 2
        assert!(tree.nodes.contains_key(&1)); // ancestor of 2
        assert!(tree.nodes.contains_key(&2)); // living
        assert!(!tree.nodes.contains_key(&3)); // dead, not ancestor of living
    }
}
