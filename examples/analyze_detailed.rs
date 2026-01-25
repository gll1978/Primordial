//! Detailed checkpoint analysis

use primordial::checkpoint::Checkpoint;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("validation_final/checkpoint_00003000.bin");

    println!("Loading checkpoint: {}", path);
    let checkpoint = Checkpoint::load(path)?;

    let alive: Vec<_> = checkpoint.organisms.iter().filter(|o| o.is_alive()).collect();
    let n = alive.len();

    if n == 0 {
        println!("No alive organisms!");
        return Ok(());
    }

    // === 1. Population ===
    println!("\n=== POPULATION ===");
    println!("Total alive: {}", n);
    println!("Time step: {}", checkpoint.time);

    // === 2. Energy Stats ===
    let mut energies: Vec<f32> = alive.iter().map(|o| o.energy).collect();
    energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let energy_sum: f32 = energies.iter().sum();
    let energy_mean = energy_sum / n as f32;
    let energy_max = energies.last().copied().unwrap_or(0.0);
    let energy_min = energies.first().copied().unwrap_or(0.0);
    let energy_median = energies[n / 2];

    println!("\n=== ENERGY ===");
    println!("Mean:   {:.2}", energy_mean);
    println!("Median: {:.2}", energy_median);
    println!("Max:    {:.2}", energy_max);
    println!("Min:    {:.2}", energy_min);

    // === 3. Elite Count ===
    let elite_100k = alive.iter().filter(|o| o.energy > 100000.0).count();
    let elite_10k = alive.iter().filter(|o| o.energy > 10000.0).count();
    let elite_1k = alive.iter().filter(|o| o.energy > 1000.0).count();
    let elite_100 = alive.iter().filter(|o| o.energy > 100.0).count();

    println!("\n=== ELITE COUNT ===");
    println!("Energy > 100k: {}", elite_100k);
    println!("Energy > 10k:  {}", elite_10k);
    println!("Energy > 1k:   {}", elite_1k);
    println!("Energy > 100:  {}", elite_100);

    // === 4. Brain Stats ===
    let brain_complexities: Vec<usize> = alive.iter().map(|o| o.brain.complexity()).collect();
    let brain_sum: usize = brain_complexities.iter().sum();
    let brain_mean = brain_sum as f32 / n as f32;
    let brain_max = brain_complexities.iter().max().copied().unwrap_or(0);

    let mut brain_dist: HashMap<usize, usize> = HashMap::new();
    for &c in &brain_complexities {
        *brain_dist.entry(c).or_insert(0) += 1;
    }

    println!("\n=== BRAIN COMPLEXITY ===");
    println!("Mean: {:.3}", brain_mean);
    println!("Max:  {}", brain_max);
    println!("Distribution:");
    let mut keys: Vec<_> = brain_dist.keys().collect();
    keys.sort();
    for k in keys {
        let count = brain_dist[k];
        let pct = 100.0 * count as f32 / n as f32;
        println!("  {} neurons: {} ({:.1}%)", k, count, pct);
    }

    // === 5. Age Stats ===
    let mut ages: Vec<u32> = alive.iter().map(|o| o.age).collect();
    ages.sort();

    let age_sum: u32 = ages.iter().sum();
    let age_mean = age_sum as f32 / n as f32;
    let age_max = ages.last().copied().unwrap_or(0);
    let age_median = ages[n / 2];
    let veterans = alive.iter().filter(|o| o.age > 500).count();
    let veterans_300 = alive.iter().filter(|o| o.age > 300).count();

    println!("\n=== AGE ===");
    println!("Mean:   {:.1}", age_mean);
    println!("Median: {}", age_median);
    println!("Max:    {}", age_max);
    println!("Veterans (>500): {}", veterans);
    println!("Veterans (>300): {}", veterans_300);

    // === 6. Lineages ===
    let mut lineages: HashMap<u32, usize> = HashMap::new();
    for o in &alive {
        *lineages.entry(o.lineage_id).or_insert(0) += 1;
    }

    println!("\n=== LINEAGES ===");
    println!("Unique lineages: {}", lineages.len());
    let mut lineage_vec: Vec<_> = lineages.iter().collect();
    lineage_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (id, count) in lineage_vec.iter().take(5) {
        let pct = 100.0 * **count as f32 / n as f32;
        println!("  Lineage {}: {} organisms ({:.1}%)", id, count, pct);
    }

    // === 7. Generation Stats ===
    let mut generations: Vec<u16> = alive.iter().map(|o| o.generation).collect();
    generations.sort();
    let gen_max = generations.last().copied().unwrap_or(0);
    let gen_mean = generations.iter().map(|&g| g as f32).sum::<f32>() / n as f32;

    let mut gen_dist: HashMap<u16, usize> = HashMap::new();
    for &g in &generations {
        *gen_dist.entry(g).or_insert(0) += 1;
    }

    println!("\n=== GENERATION ===");
    println!("Max:  {}", gen_max);
    println!("Mean: {:.1}", gen_mean);
    println!("Top generations:");
    let mut gen_vec: Vec<_> = gen_dist.iter().collect();
    gen_vec.sort_by(|a, b| b.0.cmp(a.0));
    for (gen, count) in gen_vec.iter().take(10) {
        let pct = 100.0 * **count as f32 / n as f32;
        println!("  Gen {}: {} ({:.1}%)", gen, count, pct);
    }

    // === 8. Top 5 by Energy ===
    let mut by_energy: Vec<_> = alive.iter().collect();
    by_energy.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());

    println!("\n=== TOP 5 BY ENERGY ===");
    println!("{:<8} {:>10} {:>6} {:>6} {:>8} {:>6}", "ID", "Energy", "Gen", "Age", "Offspring", "Brain");
    for o in by_energy.iter().take(5) {
        println!("{:<8} {:>10.1} {:>6} {:>6} {:>8} {:>6}",
            o.id, o.energy, o.generation, o.age, o.offspring_count, o.brain.complexity());
    }

    // === 9. Top 5 by Generation ===
    let mut by_gen: Vec<_> = alive.iter().collect();
    by_gen.sort_by(|a, b| b.generation.cmp(&a.generation));

    println!("\n=== TOP 5 BY GENERATION ===");
    println!("{:<8} {:>10} {:>6} {:>6} {:>8} {:>6}", "ID", "Energy", "Gen", "Age", "Offspring", "Brain");
    for o in by_gen.iter().take(5) {
        println!("{:<8} {:>10.1} {:>6} {:>6} {:>8} {:>6}",
            o.id, o.energy, o.generation, o.age, o.offspring_count, o.brain.complexity());
    }

    // === JSON Output ===
    println!("\n=== JSON SUMMARY ===");
    println!(r#"{{
  "time": {},
  "population": {{
    "total": {}
  }},
  "energy": {{
    "mean": {:.2},
    "median": {:.2},
    "max": {:.2},
    "min": {:.2}
  }},
  "elite_count": {{
    "above_100k": {},
    "above_10k": {},
    "above_1k": {},
    "above_100": {}
  }},
  "brain": {{
    "mean": {:.3},
    "max": {}
  }},
  "age": {{
    "mean": {:.1},
    "median": {},
    "max": {},
    "veterans_500": {},
    "veterans_300": {}
  }},
  "lineages": {{
    "unique": {}
  }},
  "generation": {{
    "max": {},
    "mean": {:.1}
  }}
}}"#,
        checkpoint.time,
        n,
        energy_mean, energy_median, energy_max, energy_min,
        elite_100k, elite_10k, elite_1k, elite_100,
        brain_mean, brain_max,
        age_mean, age_median, age_max, veterans, veterans_300,
        lineages.len(),
        gen_max, gen_mean
    );

    Ok(())
}
