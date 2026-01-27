//! Ecological systems for PRIMORDIAL Fase 2.
//!
//! This module contains:
//! - Predation system (attack/kill mechanics)
//! - Seasonal variation (food multipliers)
//! - Multi-type food system (plant, meat, fruit, insects)
//! - Terrain system (movement costs, food modifiers)
//! - Resource depletion (over-exploitation tracking)
//! - Large prey and cooperation (B3)

pub mod depletion;
pub mod food_patches;
pub mod food_types;
pub mod large_prey;
pub mod predation;
pub mod seasons;
pub mod terrain;

pub use depletion::{DepletionConfig, DepletionState, DepletionSystem};
pub use food_patches::{FoodPatch, PatchConfig, PatchWorld};
pub use food_types::{DietSpecialization, FoodCell, FoodConfig};
pub use large_prey::{CooperationManager, CooperationSignal, LargePrey, LargePreyConfig, TrustRelationship};
pub use predation::PredationConfig;
pub use seasons::{Season, SeasonalSystem, SeasonsConfig};
pub use terrain::{Terrain, TerrainConfig, TerrainGrid};
