//! Ecological systems for PRIMORDIAL Fase 2.
//!
//! This module contains:
//! - Predation system (attack/kill mechanics)
//! - Seasonal variation (food multipliers)
//! - Multi-type food system (plant, meat, fruit, insects)
//! - Terrain system (movement costs, food modifiers)
//! - Resource depletion (over-exploitation tracking)

pub mod depletion;
pub mod food_types;
pub mod predation;
pub mod seasons;
pub mod terrain;

pub use depletion::{DepletionConfig, DepletionState, DepletionSystem};
pub use food_types::{DietSpecialization, FoodCell, FoodConfig};
pub use predation::PredationConfig;
pub use seasons::{Season, SeasonalSystem, SeasonsConfig};
pub use terrain::{Terrain, TerrainConfig, TerrainGrid};
