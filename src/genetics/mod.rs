//! Genetics module - sexual reproduction, crossover, and phylogeny tracking.

pub mod crossover;
pub mod phylogeny;
pub mod sex;

pub use crossover::CrossoverSystem;
pub use phylogeny::{PhylogeneticTree, TreeNode};
pub use sex::{Sex, SexualReproductionSystem};
