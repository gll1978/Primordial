use primordial::Config;

fn main() {
    let default_config = Config::default();
    let yaml_config = Config::from_file("config.yaml").unwrap();

    println!("=== COMPARISON ===");
    println!("\nOrganisms:");
    println!("  initial_energy: default={}, yaml={}", 
        default_config.organisms.initial_energy, yaml_config.organisms.initial_energy);
    println!("  metabolism_base: default={}, yaml={}", 
        default_config.organisms.metabolism_base, yaml_config.organisms.metabolism_base);
    println!("  food_energy: default={}, yaml={}", 
        default_config.organisms.food_energy, yaml_config.organisms.food_energy);
    println!("  move_cost: default={}, yaml={}", 
        default_config.organisms.move_cost, yaml_config.organisms.move_cost);
    
    println!("\nWorld:");
    println!("  grid_size: default={}, yaml={}", 
        default_config.world.grid_size, yaml_config.world.grid_size);
    println!("  food_regen_rate: default={}, yaml={}", 
        default_config.world.food_regen_rate, yaml_config.world.food_regen_rate);
        
    println!("\nSafety:");
    println!("  max_population: default={}, yaml={}", 
        default_config.safety.max_population, yaml_config.safety.max_population);
    println!("  max_age: default={}, yaml={}", 
        default_config.safety.max_age, yaml_config.safety.max_age);
        
    println!("\nEvolution:");
    println!("  mutation_rate: default={}, yaml={}", 
        default_config.evolution.mutation_rate, yaml_config.evolution.mutation_rate);
    println!("  add_neuron_rate: default={}, yaml={}", 
        default_config.evolution.add_neuron_rate, yaml_config.evolution.add_neuron_rate);
        
    println!("\nNeural:");
    println!("  use_instincts: default={}, yaml={}", 
        default_config.neural.use_instincts, yaml_config.neural.use_instincts);
}
