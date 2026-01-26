//! Test lungo per verificare la stabilità della crescita del brain_mean

use primordial::{Config, World};

fn main() {
    println!("=== Test Brain Growth LUNGO (50000 steps) ===\n");

    let mut config = Config::default();
    config.world.grid_size = 100;
    config.organisms.initial_population = 300;
    config.predation.enabled = true;
    config.evolution.add_neuron_rate = 0.03;
    config.evolution.add_connection_rate = 0.05;
    config.neural.use_instincts = false;

    let mut world = World::new(config);

    println!("Step  | Pop  | Gen | brain_mean | brain_max | kills | predators");
    println!("------|------|-----|------------|-----------|-------|----------");

    let total_steps = 50000;
    let report_interval = 2500;

    let mut max_brain_mean = 0.0f32;
    let mut max_brain_mean_step = 0u64;

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

            let brain_complexities: Vec<f32> = alive.iter()
                .map(|o| o.brain.complexity() as f32)
                .collect();
            let brain_mean = brain_complexities.iter().sum::<f32>() / pop as f32;
            let brain_max = brain_complexities.iter().cloned().fold(0.0f32, f32::max);

            let total_kills: u32 = alive.iter().map(|o| o.kills as u32).sum();
            let predator_count = alive.iter().filter(|o| o.is_predator).count();

            if brain_mean > max_brain_mean {
                max_brain_mean = brain_mean;
                max_brain_mean_step = step;
            }

            println!("{:5} | {:4} | {:3} | {:10.3} | {:9.0} | {:5} | {:9}",
                step, pop, gen_max, brain_mean, brain_max, total_kills, predator_count);
        }
    }

    println!("\n=== RISULTATO CHIAVE ===");
    println!("Max brain_mean raggiunto: {:.3} (step {})", max_brain_mean, max_brain_mean_step);

    // Statistiche finali
    let alive: Vec<_> = world.organisms.iter().filter(|o| o.is_alive()).collect();
    if !alive.is_empty() {
        let brain_complexities: Vec<usize> = alive.iter().map(|o| o.brain.complexity()).collect();
        let total_neurons: Vec<usize> = alive.iter().map(|o| o.brain.total_hidden_neurons()).collect();

        println!("\n=== Statistiche Finali ===");
        println!("Popolazione: {}", alive.len());
        println!("Generazione max: {}", alive.iter().map(|o| o.generation).max().unwrap_or(0));
        println!("Brain layers: mean={:.3}, max={}",
            brain_complexities.iter().sum::<usize>() as f32 / alive.len() as f32,
            brain_complexities.iter().max().unwrap_or(&0));
        println!("Hidden neurons: mean={:.2}, max={}",
            total_neurons.iter().sum::<usize>() as f32 / alive.len() as f32,
            total_neurons.iter().max().unwrap_or(&0));

        // Distribuzione
        let mut dist = [0usize; 11];
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
    }
}
