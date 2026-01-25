//! Ecological systems for PRIMORDIAL Fase 2.
//!
//! This module contains:
//! - Predation system (attack/kill mechanics)
//! - Seasonal variation (food multipliers)
//! - Multi-type food system (plant, meat, fruit, insects)

pub mod predation;
pub mod seasons;
pub mod food_types;

pub use predation::{AttackResult, PredationConfig};
pub use seasons::{Season, SeasonalSystem, SeasonsConfig};
pub use food_types::{FoodCell, FoodConfig};
