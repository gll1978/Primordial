//! Multi-type food system.

use serde::{Deserialize, Serialize};

/// A cell containing multiple food types
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FoodCell {
    /// Plant matter (grass, leaves)
    pub plant: f32,
    /// Meat from kills
    pub meat: f32,
    /// Seasonal fruit
    pub fruit: f32,
    /// Insects
    pub insects: f32,
}

impl FoodCell {
    /// Create a new food cell with default values
    pub fn new() -> Self {
        Self {
            plant: 10.0,
            meat: 0.0,
            fruit: 5.0,
            insects: 3.0,
        }
    }

    /// Create an empty food cell
    pub fn empty() -> Self {
        Self::default()
    }

    /// Total food in cell
    pub fn total(&self) -> f32 {
        self.plant + self.meat + self.fruit + self.insects
    }

    /// Check if cell has any food
    pub fn is_empty(&self) -> bool {
        self.total() < 0.1
    }

    /// Regenerate food based on config and season multipliers
    pub fn regenerate(
        &mut self,
        config: &FoodConfig,
        plant_multiplier: f32,
        fruit_multiplier: f32,
        insect_multiplier: f32,
    ) {
        // Plant regenerates steadily
        self.plant += config.plant_regen_rate * plant_multiplier;
        self.plant = self.plant.min(config.max_plant);

        // Meat decays (doesn't regenerate)
        self.meat *= config.meat_decay_rate;

        // Fruit regenerates seasonally
        self.fruit += config.fruit_regen_rate * fruit_multiplier;
        self.fruit = self.fruit.min(config.max_fruit);

        // Insects regenerate seasonally
        self.insects += config.insect_regen_rate * insect_multiplier;
        self.insects = self.insects.min(config.max_insects);
    }

    /// Consume food based on organism's diet specialization
    /// Returns (energy_gained, meat_produced) - meat from leftover organic matter
    pub fn consume(&mut self, diet: &DietSpecialization, consumption_rate: f32) -> f32 {
        let mut energy = 0.0;

        // Consume each food type based on efficiency and availability
        let plant_consumed = (self.plant * consumption_rate).min(self.plant);
        self.plant -= plant_consumed;
        energy += plant_consumed * diet.plant_efficiency * PLANT_ENERGY_DENSITY;

        let meat_consumed = (self.meat * consumption_rate).min(self.meat);
        self.meat -= meat_consumed;
        energy += meat_consumed * diet.meat_efficiency * MEAT_ENERGY_DENSITY;

        let fruit_consumed = (self.fruit * consumption_rate).min(self.fruit);
        self.fruit -= fruit_consumed;
        energy += fruit_consumed * diet.fruit_efficiency * FRUIT_ENERGY_DENSITY;

        let insect_consumed = (self.insects * consumption_rate).min(self.insects);
        self.insects -= insect_consumed;
        energy += insect_consumed * diet.insect_efficiency * INSECT_ENERGY_DENSITY;

        energy
    }

    /// Add meat to this cell (from a kill)
    pub fn add_meat(&mut self, amount: f32, config: &FoodConfig) {
        self.meat += amount;
        self.meat = self.meat.min(config.max_meat);
    }
}

/// Energy density constants
pub const PLANT_ENERGY_DENSITY: f32 = 1.0;
pub const MEAT_ENERGY_DENSITY: f32 = 3.0;
pub const FRUIT_ENERGY_DENSITY: f32 = 2.0;
pub const INSECT_ENERGY_DENSITY: f32 = 0.5;

/// Diet specialization - how efficiently an organism extracts energy from each food type
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DietSpecialization {
    pub plant_efficiency: f32,
    pub meat_efficiency: f32,
    pub fruit_efficiency: f32,
    pub insect_efficiency: f32,
}

impl Default for DietSpecialization {
    fn default() -> Self {
        // Omnivore - moderate efficiency for all
        Self {
            plant_efficiency: 0.5,
            meat_efficiency: 0.5,
            fruit_efficiency: 0.5,
            insect_efficiency: 0.5,
        }
    }
}

impl DietSpecialization {
    /// Create a herbivore diet
    pub fn herbivore() -> Self {
        Self {
            plant_efficiency: 1.0,
            meat_efficiency: 0.1,
            fruit_efficiency: 0.8,
            insect_efficiency: 0.3,
        }
    }

    /// Create a carnivore diet
    pub fn carnivore() -> Self {
        Self {
            plant_efficiency: 0.1,
            meat_efficiency: 1.0,
            fruit_efficiency: 0.2,
            insect_efficiency: 0.5,
        }
    }

    /// Create an omnivore diet
    pub fn omnivore() -> Self {
        Self::default()
    }

    /// Create a random diet (for initial population)
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        Self {
            plant_efficiency: rng.gen_range(0.2..1.0),
            meat_efficiency: rng.gen_range(0.2..1.0),
            fruit_efficiency: rng.gen_range(0.2..1.0),
            insect_efficiency: rng.gen_range(0.2..1.0),
        }
    }

    /// Mutate diet slightly
    pub fn mutate(&mut self, strength: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        self.plant_efficiency += rng.gen_range(-strength..strength);
        self.meat_efficiency += rng.gen_range(-strength..strength);
        self.fruit_efficiency += rng.gen_range(-strength..strength);
        self.insect_efficiency += rng.gen_range(-strength..strength);

        // Clamp to valid range
        self.plant_efficiency = self.plant_efficiency.clamp(0.0, 1.0);
        self.meat_efficiency = self.meat_efficiency.clamp(0.0, 1.0);
        self.fruit_efficiency = self.fruit_efficiency.clamp(0.0, 1.0);
        self.insect_efficiency = self.insect_efficiency.clamp(0.0, 1.0);
    }

    /// Get the dominant diet type
    pub fn dominant_type(&self) -> &'static str {
        let max = self
            .plant_efficiency
            .max(self.meat_efficiency)
            .max(self.fruit_efficiency)
            .max(self.insect_efficiency);

        if (self.meat_efficiency - max).abs() < 0.01 {
            "Carnivore"
        } else if (self.plant_efficiency - max).abs() < 0.01 {
            "Herbivore"
        } else if (self.fruit_efficiency - max).abs() < 0.01 {
            "Frugivore"
        } else if (self.insect_efficiency - max).abs() < 0.01 {
            "Insectivore"
        } else {
            "Omnivore"
        }
    }
}

/// Food configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoodConfig {
    /// Plant regeneration rate per step
    pub plant_regen_rate: f32,
    /// Fruit regeneration rate per step
    pub fruit_regen_rate: f32,
    /// Insect regeneration rate per step
    pub insect_regen_rate: f32,
    /// Meat decay rate (multiplier, <1.0)
    pub meat_decay_rate: f32,

    /// Maximum plant per cell
    pub max_plant: f32,
    /// Maximum meat per cell
    pub max_meat: f32,
    /// Maximum fruit per cell
    pub max_fruit: f32,
    /// Maximum insects per cell
    pub max_insects: f32,
}

impl Default for FoodConfig {
    fn default() -> Self {
        Self {
            plant_regen_rate: 0.5,
            fruit_regen_rate: 0.2,
            insect_regen_rate: 0.1,
            meat_decay_rate: 0.95, // 5% decay per step

            max_plant: 50.0,
            max_meat: 100.0,
            max_fruit: 30.0,
            max_insects: 20.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_food_cell_creation() {
        let cell = FoodCell::new();
        assert!((cell.plant - 10.0).abs() < 0.01);
        assert!((cell.meat - 0.0).abs() < 0.01);
        assert!(cell.total() > 0.0);
    }

    #[test]
    fn test_food_consumption() {
        let mut cell = FoodCell::new();
        cell.plant = 20.0;
        cell.meat = 10.0;

        let diet = DietSpecialization::herbivore();
        let energy = cell.consume(&diet, 0.5);

        // Herbivore gets full energy from plants, low from meat
        assert!(energy > 0.0);
        assert!(cell.plant < 20.0);
        assert!(cell.meat < 10.0);
    }

    #[test]
    fn test_food_regeneration() {
        let mut cell = FoodCell::empty();
        let config = FoodConfig::default();

        cell.regenerate(&config, 1.0, 1.0, 1.0);

        assert!(cell.plant > 0.0);
        assert!(cell.fruit > 0.0);
        assert!(cell.insects > 0.0);
    }

    #[test]
    fn test_meat_decay() {
        let mut cell = FoodCell::empty();
        cell.meat = 100.0;
        let config = FoodConfig::default();

        cell.regenerate(&config, 1.0, 1.0, 1.0);

        // Meat should decay
        assert!(cell.meat < 100.0);
        assert!((cell.meat - 95.0).abs() < 0.01); // 5% decay
    }

    #[test]
    fn test_diet_specialization() {
        let herbivore = DietSpecialization::herbivore();
        assert!(herbivore.plant_efficiency > herbivore.meat_efficiency);

        let carnivore = DietSpecialization::carnivore();
        assert!(carnivore.meat_efficiency > carnivore.plant_efficiency);
    }

    #[test]
    fn test_diet_mutation() {
        let mut diet = DietSpecialization::default();
        let original_plant = diet.plant_efficiency;

        // Mutate many times - should change
        for _ in 0..100 {
            diet.mutate(0.1);
        }

        // Values should still be valid
        assert!(diet.plant_efficiency >= 0.0 && diet.plant_efficiency <= 1.0);
        assert!(diet.meat_efficiency >= 0.0 && diet.meat_efficiency <= 1.0);

        // Should have changed
        assert!((diet.plant_efficiency - original_plant).abs() > 0.01);
    }
}
