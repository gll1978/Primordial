//! Evolution mechanics and selection.

use crate::config::Config;
use crate::ecology::food_types::DietSpecialization;
use crate::genetics::Sex;
use crate::neural::{CrossoverStrategy, MutationConfig};
use crate::organism::Organism;
use rand::seq::SliceRandom;

/// Evolution engine for managing population genetics
pub struct EvolutionEngine {
    pub mutation_config: MutationConfig,
    pub crossover_strategy: CrossoverStrategy,
    pub crossover_rate: f32,
    pub elitism_rate: f32,
}

impl EvolutionEngine {
    /// Create evolution engine from config
    pub fn from_config(config: &Config) -> Self {
        Self {
            mutation_config: MutationConfig {
                weight_mutation_rate: config.evolution.mutation_rate,
                weight_mutation_strength: config.evolution.mutation_strength,
                add_neuron_rate: config.evolution.add_neuron_rate,
                add_connection_rate: config.evolution.add_connection_rate,
                max_neurons: config.safety.max_neurons,
            },
            crossover_strategy: CrossoverStrategy::FitterParent,
            crossover_rate: 0.1,
            elitism_rate: 0.1,
        }
    }

    /// Mutate an organism's brain
    pub fn mutate(&self, organism: &mut Organism) {
        organism.brain.mutate(&self.mutation_config);
    }

    /// Perform crossover between two organisms
    pub fn crossover(&self, parent1: &Organism, parent2: &Organism) -> Organism {
        let fitness1 = parent1.fitness();
        let fitness2 = parent2.fitness();

        let mut child_brain = parent1
            .brain
            .crossover_with_strategy(&parent2.brain, fitness1, fitness2, &self.crossover_strategy);

        // Apply mutation to child
        child_brain.mutate(&self.mutation_config);

        // Create child diet by averaging parents
        let mut child_diet = DietSpecialization {
            plant_efficiency: (parent1.diet.plant_efficiency + parent2.diet.plant_efficiency) / 2.0,
            meat_efficiency: (parent1.diet.meat_efficiency + parent2.diet.meat_efficiency) / 2.0,
            fruit_efficiency: (parent1.diet.fruit_efficiency + parent2.diet.fruit_efficiency) / 2.0,
            insect_efficiency: (parent1.diet.insect_efficiency + parent2.diet.insect_efficiency) / 2.0,
        };
        child_diet.mutate(self.mutation_config.weight_mutation_strength);

        // Create child organism
        Organism {
            id: 0, // Will be assigned later
            lineage_id: parent1.lineage_id, // Inherit from primary parent
            generation: parent1.generation.max(parent2.generation) + 1,
            x: parent1.x,
            y: parent1.y,
            size: (parent1.size + parent2.size) / 2.0,
            energy: 50.0,
            health: 100.0,
            age: 0,
            brain: child_brain,
            memory: [0.0; 5],
            kills: 0,
            offspring_count: 0,
            food_eaten: 0,
            is_predator: parent1.is_predator || parent2.is_predator,
            signal: 0.0,
            last_action: None,
            diet: child_diet,
            attack_cooldown: 0,
            cause_of_death: None,
            is_aquatic: parent1.is_aquatic || parent2.is_aquatic,
            sex: Sex::random(),
            parent1_id: Some(parent1.id),
            parent2_id: Some(parent2.id),
            mate_cooldown: 0,
            social_signal: crate::organism::SocialSignal::None,
            signal_cooldown: 0,
            path_history: std::collections::VecDeque::with_capacity(5),
            observed_predators: std::collections::HashMap::with_capacity(10),
            cooperation_signal: crate::ecology::large_prey::CooperationSignal::None,
            trust_relationships: std::collections::HashMap::with_capacity(10),
            current_hunt_target: None,
            coop_successes: 0,
            coop_failures: 0,
            last_coop_time: 0,
            last_reward: 0.0,
            total_lifetime_reward: 0.0,
            successful_forages: 0,
            failed_forages: 0,
        }
    }

    /// Select organisms for reproduction based on fitness
    pub fn select_parents<'a>(&self, organisms: &'a [Organism], count: usize) -> Vec<&'a Organism> {
        let mut rng = rand::thread_rng();

        // Filter alive organisms with sufficient energy
        let candidates: Vec<&Organism> = organisms
            .iter()
            .filter(|o| o.is_alive() && o.energy > 30.0)
            .collect();

        if candidates.is_empty() {
            return Vec::new();
        }

        // Tournament selection
        let mut selected = Vec::with_capacity(count);
        let tournament_size = 3;

        for _ in 0..count {
            let tournament: Vec<_> = candidates
                .choose_multiple(&mut rng, tournament_size.min(candidates.len()))
                .collect();

            if let Some(winner) = tournament.into_iter().max_by(|a, b| {
                a.fitness()
                    .partial_cmp(&b.fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                selected.push(*winner);
            }
        }

        selected
    }

    /// Get elite organisms (top performers)
    pub fn get_elites<'a>(&self, organisms: &'a [Organism], count: usize) -> Vec<&'a Organism> {
        let mut alive: Vec<_> = organisms.iter().filter(|o| o.is_alive()).collect();

        alive.sort_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        alive.into_iter().take(count).collect()
    }
}

/// Species for NEAT-style speciation (simplified)
#[derive(Clone, Debug)]
pub struct Species {
    pub id: u32,
    pub representative_brain_complexity: usize,
    pub members: Vec<usize>, // organism indices
    pub best_fitness: f32,
    pub stagnation: u32,
}

impl Species {
    pub fn new(id: u32, representative_complexity: usize) -> Self {
        Self {
            id,
            representative_brain_complexity: representative_complexity,
            members: Vec::new(),
            best_fitness: 0.0,
            stagnation: 0,
        }
    }

    /// Check if an organism belongs to this species (simplified check)
    pub fn belongs(&self, organism: &Organism, threshold: usize) -> bool {
        let complexity_diff = (organism.brain.complexity() as i32
            - self.representative_brain_complexity as i32)
            .unsigned_abs() as usize;
        complexity_diff <= threshold
    }

    /// Update species after a generation
    pub fn update(&mut self, organisms: &[Organism]) {
        let current_best = self
            .members
            .iter()
            .filter_map(|&idx| organisms.get(idx))
            .filter(|o| o.is_alive())
            .map(|o| o.fitness())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        if current_best > self.best_fitness {
            self.best_fitness = current_best;
            self.stagnation = 0;
        } else {
            self.stagnation += 1;
        }
    }
}

/// Speciation manager
#[derive(Default)]
pub struct SpeciationManager {
    pub species: Vec<Species>,
    pub next_species_id: u32,
    pub compatibility_threshold: usize,
}

impl SpeciationManager {
    pub fn new(threshold: usize) -> Self {
        Self {
            species: Vec::new(),
            next_species_id: 0,
            compatibility_threshold: threshold,
        }
    }

    /// Assign organisms to species
    pub fn speciate(&mut self, organisms: &[Organism]) {
        // Clear current members
        for species in &mut self.species {
            species.members.clear();
        }

        // Assign each organism to a species
        for (idx, organism) in organisms.iter().enumerate() {
            if !organism.is_alive() {
                continue;
            }

            let mut assigned = false;

            for species in &mut self.species {
                if species.belongs(organism, self.compatibility_threshold) {
                    species.members.push(idx);
                    assigned = true;
                    break;
                }
            }

            // Create new species if not assigned
            if !assigned {
                let mut new_species = Species::new(self.next_species_id, organism.brain.complexity());
                self.next_species_id += 1;
                new_species.members.push(idx);
                self.species.push(new_species);
            }
        }

        // Remove empty species
        self.species.retain(|s| !s.members.is_empty());

        // Update species stats
        for species in &mut self.species {
            species.update(organisms);
        }
    }

    /// Get species count
    pub fn species_count(&self) -> usize {
        self.species.len()
    }

    /// Get largest species
    pub fn largest_species(&self) -> Option<&Species> {
        self.species.iter().max_by_key(|s| s.members.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config::default()
    }

    #[test]
    fn test_evolution_engine() {
        let config = test_config();
        let engine = EvolutionEngine::from_config(&config);

        let mut org = Organism::new(1, 1, 10, 10, &config);
        let _original_weights = org.brain.layers[0].weights.clone();

        engine.mutate(&mut org);

        // Weights should potentially change (high probability given mutation rate)
        assert!(org.brain.is_valid());
    }

    #[test]
    fn test_crossover() {
        let config = test_config();
        let engine = EvolutionEngine::from_config(&config);

        let parent1 = Organism::new(1, 1, 10, 10, &config);
        let parent2 = Organism::new(2, 2, 20, 20, &config);

        let child = engine.crossover(&parent1, &parent2);

        assert_eq!(child.generation, 1);
        assert!(child.brain.is_valid());
    }

    #[test]
    fn test_selection() {
        let config = test_config();
        let engine = EvolutionEngine::from_config(&config);

        let mut organisms = Vec::new();
        for i in 0..10 {
            let mut org = Organism::new(i as u64, 1, 10, 10, &config);
            org.energy = 50.0 + i as f32 * 10.0;
            organisms.push(org);
        }

        let selected = engine.select_parents(&organisms, 5);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_speciation() {
        let config = test_config();
        let mut manager = SpeciationManager::new(5);

        let mut organisms = Vec::new();
        for i in 0..20 {
            let mut org = Organism::new(i as u64, 1, 10, 10, &config);
            // Add varying complexity
            if i % 3 == 0 {
                org.brain.add_neuron();
            }
            if i % 5 == 0 {
                org.brain.add_neuron();
                org.brain.add_neuron();
            }
            organisms.push(org);
        }

        manager.speciate(&organisms);

        assert!(manager.species_count() > 0);
    }
}
