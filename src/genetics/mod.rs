//! Genetics module - sexual reproduction, crossover, phylogeny tracking, and diversity metrics.

pub mod crossover;
pub mod diversity;
pub mod phylogeny;
pub mod sex;

pub use crossover::CrossoverSystem;
pub use diversity::{DiversityHistory, DiversityMetrics, DiversityRecord};
pub use phylogeny::{PhylogeneticTree, TreeNode};
pub use sex::{Sex, SexualReproductionSystem};
