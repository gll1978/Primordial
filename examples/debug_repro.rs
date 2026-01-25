use primordial::{World, Config};

fn main() {
    // Use ONLY default config - no modifications
    let config = Config::default();
    
    println!("=== DEFAULT CONFIG ===");
    println!("initial_energy: {}", config.organisms.initial_energy);
    println!("reproduction_threshold: {}", config.organisms.reproduction_threshold);
    println!("reproduction_cost: {}", config.organisms.reproduction_cost);
    println!("metabolism_base: {}", config.organisms.metabolism_base);
    println!("move_cost: {}", config.organisms.move_cost);
    println!("food_energy: {}", config.organisms.food_energy);
    println!("food_regen_rate: {}", config.world.food_regen_rate);
    println!("reproduction.enabled (sexual): {}", config.reproduction.enabled);
    
    let mut world = World::new(config.clone());
    
    println!("\n=== SIMULATION ===");
    
    let mut total_births = 0u64;
    
    for step in 0..=5000 {
        world.step();
        total_births += world.stats.births as u64;
        
        if step % 1000 == 0 {
            println!("Step {:5}: pop={:4}, births={:6}, gen_max={:3}, avg_energy={:.1}",
                step, 
                world.population(),
                total_births,
                world.generation_max,
                world.stats.energy_mean);
        }
        
        if world.is_extinct() {
            println!("\n*** EXTINCTION at step {} ***", step);
            break;
        }
    }
    
    println!("\n=== FINAL ===");
    println!("Population: {}", world.population());
    println!("Total births: {}", total_births);
    println!("Max generation: {}", world.generation_max);
    println!("Status: {}", if world.is_extinct() { "EXTINCT" } else { "ALIVE" });
}
