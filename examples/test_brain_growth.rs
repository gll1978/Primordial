//! Test per verificare la crescita del brain_mean con le nuove modifiche

use primordial::{Config, World};

fn main() {
    println!("=== Test Brain Growth con Predation ===\n");

    // Configurazione ottimizzata per testare la crescita del brain
    let mut config = Config::default();
    config.world.grid_size = 100;
    config.organisms.initial_population = 300;
    config.predation.enabled = true;
    config.predation.damage_multiplier = 1.5;
    config.evolution.add_neuron_rate = 0.03;
    config.evolution.add_connection_rate = 0.05;
    config.neural.use_instincts = false;

    let mut world = World::new(config);

    println!("Step  | Pop | Gen | brain_mean | brain_max | kills | predators | avg_fitness");
    println!("------|-----|-----|------------|-----------|-------|-----------|------------");

    let total_steps = 20000;
    let report_interval = 1000;

    for step in 1..=total_steps {
        world.step();

        if step % report_interval == 0 || step == 1 {
            let alive: Vec<_> = world.organisms.iter().filter(|o| o.is_alive()).collect();
            let pop = alive.len();

            if pop == 0 {
                println!("ESTINZIONE a step {}", step);
                break;
            }

            let gen_max = alive.iter().map(|o| o.generation).max().unwrap_or(0);

            // Calcola brain complexity (numero di hidden layers)
            let brain_complexities: Vec<f32> = alive.iter()
                .map(|o| o.brain.complexity() as f32)
                .collect();
            let brain_mean = brain_complexities.iter().sum::<f32>() / pop as f32;
            let brain_max = brain_complexities.iter().cloned().fold(0.0f32, f32::max);

            // Conta kills totali e predatori
            let total_kills: u32 = alive.iter().map(|o| o.kills as u32).sum();
            let predator_count = alive.iter().filter(|o| o.is_predator).count();

            // Fitness media
            let avg_fitness: f32 = alive.iter().map(|o| o.fitness()).sum::<f32>() / pop as f32;

            println!("{:5} | {:3} | {:3} | {:10.3} | {:9.0} | {:5} | {:9} | {:11.1}",
                step, pop, gen_max, brain_mean, brain_max, total_kills, predator_count, avg_fitness);
        }
    }

    // Statistiche finali
    println!("\n=== Statistiche Finali ===");
    let alive: Vec<_> = world.organisms.iter().filter(|o| o.is_alive()).collect();
    if !alive.is_empty() {
        let brain_complexities: Vec<usize> = alive.iter().map(|o| o.brain.complexity()).collect();
        let total_neurons: Vec<usize> = alive.iter().map(|o| o.brain.total_hidden_neurons()).collect();

        println!("Popolazione finale: {}", alive.len());
        println!("Generazione max: {}", alive.iter().map(|o| o.generation).max().unwrap_or(0));
        println!("Brain complexity (layers):");
        println!("  - mean: {:.3}", brain_complexities.iter().sum::<usize>() as f32 / alive.len() as f32);
        println!("  - max: {}", brain_complexities.iter().max().unwrap_or(&0));
        println!("Total hidden neurons:");
        println!("  - mean: {:.2}", total_neurons.iter().sum::<usize>() as f32 / alive.len() as f32);
        println!("  - max: {}", total_neurons.iter().max().unwrap_or(&0));

        // Distribuzione complessità
        let mut dist = [0usize; 11]; // 0-10 layers
        for &c in &brain_complexities {
            if c < 11 { dist[c] += 1; }
        }
        println!("\nDistribuzione brain complexity:");
        for (i, &count) in dist.iter().enumerate() {
            if count > 0 {
                let pct = count as f32 / alive.len() as f32 * 100.0;
                println!("  {} layers: {} ({:.1}%)", i, count, pct);
            }
        }

        // Top predatori
        let mut predators: Vec<_> = alive.iter().filter(|o| o.kills > 0).collect();
        predators.sort_by(|a, b| b.kills.cmp(&a.kills));
        if !predators.is_empty() {
            println!("\nTop 5 predatori:");
            for (i, p) in predators.iter().take(5).enumerate() {
                println!("  {}. kills={}, brain_layers={}, neurons={}, fitness={:.0}",
                    i+1, p.kills, p.brain.complexity(), p.brain.total_hidden_neurons(), p.fitness());
            }
        }
    }
}
