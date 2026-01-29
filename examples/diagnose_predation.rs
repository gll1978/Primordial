//! Diagnosi: perché la predazione non sta avvenendo?

use primordial::{Config, World};
use primordial::organism::{Action, CognitiveInputs};

fn main() {
    println!("=== Diagnosi Predazione ===\n");

    let mut config = Config::default();
    config.world.grid_size = 50; // Griglia più piccola = più incontri
    config.organisms.initial_population = 200;
    config.predation.enabled = true;
    config.predation.damage_multiplier = 2.0; // Danno aumentato
    config.neural.use_instincts = false;

    let mut world = World::new(config.clone());

    let mut total_attack_actions = 0u64;
    let mut total_kills = 0u64;
    let mut attack_attempts_blocked = 0u64;

    println!("Eseguendo 5000 steps con diagnostica...\n");

    for step in 1..=5000 {
        // Conta quanti organismi vogliono attaccare
        let alive: Vec<_> = world.organisms.iter().filter(|o| o.is_alive()).collect();
        let cognitive = CognitiveInputs::default();

        for org in &alive {
            let inputs = org.sense(
                &world.food_grid,
                &world.spatial_index,
                &world.organisms,
                world.time,
                &world.config,
                &cognitive,
            );
            let outputs = org.brain.forward(&inputs);
            let mut output_array = [0.0f32; 19];
            for (i, &val) in outputs.iter().take(19).enumerate() {
                output_array[i] = val;
            }
            let action = org.decide_action(&output_array);

            if matches!(action, Action::Attack) {
                total_attack_actions += 1;

                // Verifica se ha cooldown
                if org.attack_cooldown > 0 {
                    attack_attempts_blocked += 1;
                }
            }
        }

        let _kills_before = world.kills_this_step;
        world.step();
        total_kills += world.kills_this_step as u64;

        if step % 1000 == 0 {
            let pop = world.organisms.iter().filter(|o| o.is_alive()).count();
            let predators = world.organisms.iter().filter(|o| o.is_alive() && o.is_predator).count();
            println!("Step {}: pop={}, predators={}, kills_this_interval={}",
                step, pop, predators, total_kills);
        }
    }

    println!("\n=== Risultati Diagnosi ===");
    println!("Totale azioni Attack scelte: {}", total_attack_actions);
    println!("Attack bloccati da cooldown: {}", attack_attempts_blocked);
    println!("Totale kills: {}", total_kills);
    println!("Ratio kills/attack: {:.4}",
        if total_attack_actions > 0 { total_kills as f64 / total_attack_actions as f64 } else { 0.0 });

    // Analizza distribuzione output neurali
    println!("\n=== Analisi Output Neurali ===");
    let alive: Vec<_> = world.organisms.iter().filter(|o| o.is_alive()).collect();
    let sample_size = alive.len().min(100);
    let cognitive = CognitiveInputs::default();

    let mut output_sums = [0.0f64; 12];
    let mut attack_chosen = 0;

    for org in alive.iter().take(sample_size) {
        let inputs = org.sense(
            &world.food_grid,
            &world.spatial_index,
            &world.organisms,
            world.time,
            &world.config,
            &cognitive,
        );
        let outputs = org.brain.forward(&inputs);

        for (i, &val) in outputs.iter().take(12).enumerate() {
            output_sums[i] += val as f64;
        }

        // Trova azione scelta
        let max_idx = outputs.iter().take(12)
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        if max_idx == 6 { attack_chosen += 1; }
    }

    println!("Media output per {} organismi campione:", sample_size);
    let action_names = ["MoveN", "MoveE", "MoveS", "MoveW", "Eat", "Repro", "Attack", "Signal", "Wait", "SigDng", "SigFd", "Res"];
    for (i, &sum) in output_sums.iter().enumerate() {
        let avg = sum / sample_size as f64;
        println!("  {}: {:.4}", action_names[i], avg);
    }
    println!("\nOrganismi che sceglierebbero Attack: {}/{} ({:.1}%)",
        attack_chosen, sample_size, attack_chosen as f64 / sample_size as f64 * 100.0);

    // Verifica vicinanza tra organismi
    println!("\n=== Analisi Densità ===");
    let mut orgs_with_neighbors = 0;
    for org in alive.iter().take(sample_size) {
        let neighbors = world.spatial_index.query_neighbors(org.x, org.y, 1);
        let valid_neighbors: Vec<_> = neighbors.iter()
            .filter(|&&idx| idx < world.organisms.len() &&
                           world.organisms[idx].is_alive() &&
                           world.organisms[idx].id != org.id)
            .collect();
        if !valid_neighbors.is_empty() {
            orgs_with_neighbors += 1;
        }
    }
    println!("Organismi con vicini in range 1: {}/{} ({:.1}%)",
        orgs_with_neighbors, sample_size, orgs_with_neighbors as f64 / sample_size as f64 * 100.0);
}
