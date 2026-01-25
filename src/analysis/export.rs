//! Data export system for analysis in external tools.

use crate::genetics::diversity::DiversityHistory;
use crate::genetics::phylogeny::PhylogeneticTree;
use crate::organism::Organism;
use crate::world::World;
use std::fs::File;
use std::io::{Result, Write};
use std::path::Path;

/// Export system for saving simulation data
pub struct ExportSystem;

impl ExportSystem {
    /// Export current organisms to CSV
    pub fn export_organisms_csv<P: AsRef<Path>>(organisms: &[Organism], path: P) -> Result<()> {
        let mut file = File::create(path)?;

        writeln!(
            file,
            "id,x,y,energy,size,brain_complexity,age,lineage_id,generation,sex,kills,offspring,food_eaten,is_predator,is_aquatic"
        )?;

        for org in organisms {
            if !org.is_alive() {
                continue;
            }

            writeln!(
                file,
                "{},{},{},{:.2},{:.2},{},{},{},{},{:?},{},{},{},{},{}",
                org.id,
                org.x,
                org.y,
                org.energy,
                org.size,
                org.brain.complexity(),
                org.age,
                org.lineage_id,
                org.generation,
                org.sex,
                org.kills,
                org.offspring_count,
                org.food_eaten,
                org.is_predator,
                org.is_aquatic,
            )?;
        }

        Ok(())
    }

    /// Export phylogeny to Newick format
    pub fn export_phylogeny_newick<P: AsRef<Path>>(
        phylogeny: &PhylogeneticTree,
        path: P,
    ) -> Result<()> {
        let mut file = File::create(path)?;

        for tree in phylogeny.export_all_newick() {
            writeln!(file, "{}", tree)?;
        }

        Ok(())
    }

    /// Export phylogeny nodes to CSV
    pub fn export_phylogeny_csv<P: AsRef<Path>>(
        phylogeny: &PhylogeneticTree,
        path: P,
    ) -> Result<()> {
        let mut file = File::create(path)?;

        writeln!(
            file,
            "organism_id,parent1_id,parent2_id,birth_time,death_time,brain_complexity,birth_energy,birth_size,peak_energy,offspring_count,kills,lineage_id,generation,genome_hash"
        )?;

        for node in phylogeny.nodes.values() {
            let parent1 = node.parent1_id.map_or("".to_string(), |id| id.to_string());
            let parent2 = node.parent2_id.map_or("".to_string(), |id| id.to_string());
            let death_time = node.death_time.map_or("".to_string(), |t| t.to_string());

            writeln!(
                file,
                "{},{},{},{},{},{},{:.2},{:.2},{:.2},{},{},{},{},{}",
                node.organism_id,
                parent1,
                parent2,
                node.birth_time,
                death_time,
                node.brain_complexity,
                node.birth_energy,
                node.birth_size,
                node.peak_energy,
                node.offspring_count,
                node.kills,
                node.lineage_id,
                node.generation,
                node.genome_hash,
            )?;
        }

        Ok(())
    }

    /// Export diversity history to CSV
    pub fn export_diversity_csv<P: AsRef<Path>>(
        history: &DiversityHistory,
        path: P,
    ) -> Result<()> {
        let mut file = File::create(path)?;
        write!(file, "{}", history.to_csv())?;
        Ok(())
    }

    /// Export world snapshot to JSON
    pub fn export_world_json<P: AsRef<Path>>(world: &World, path: P) -> Result<()> {
        let snapshot = WorldSnapshot {
            time: world.time,
            population: world.population(),
            generation_max: world.generation_max,
            organisms: world
                .organisms
                .iter()
                .filter(|o| o.is_alive())
                .map(OrganismSnapshot::from)
                .collect(),
            season: format!("{:?}", world.seasonal_system.current_season),
            total_food: world.food_grid.total_food(),
        };

        let json = serde_json::to_string_pretty(&snapshot)?;
        std::fs::write(path, json)?;

        Ok(())
    }

    /// Export full state for checkpoint-like analysis
    pub fn export_full_state<P: AsRef<Path>>(
        world: &World,
        base_path: P,
    ) -> Result<ExportManifest> {
        let base = base_path.as_ref();
        std::fs::create_dir_all(base)?;

        let organisms_path = base.join("organisms.csv");
        let phylogeny_path = base.join("phylogeny.csv");
        let newick_path = base.join("phylogeny.newick");
        let snapshot_path = base.join("snapshot.json");

        Self::export_organisms_csv(&world.organisms, &organisms_path)?;
        Self::export_phylogeny_csv(&world.phylogeny, &phylogeny_path)?;
        Self::export_phylogeny_newick(&world.phylogeny, &newick_path)?;
        Self::export_world_json(world, &snapshot_path)?;

        Ok(ExportManifest {
            time: world.time,
            organisms_file: organisms_path.to_string_lossy().to_string(),
            phylogeny_file: phylogeny_path.to_string_lossy().to_string(),
            newick_file: newick_path.to_string_lossy().to_string(),
            snapshot_file: snapshot_path.to_string_lossy().to_string(),
        })
    }

    /// Export summary statistics
    pub fn export_summary<P: AsRef<Path>>(world: &World, path: P) -> Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "=== PRIMORDIAL Simulation Summary ===")?;
        writeln!(file, "Time: {}", world.time)?;
        writeln!(file, "Population: {}", world.population())?;
        writeln!(file, "Max Generation: {}", world.generation_max)?;
        writeln!(file, "Season: {:?}", world.seasonal_system.current_season)?;
        writeln!(file, "Total Food: {:.2}", world.food_grid.total_food())?;
        writeln!(file)?;

        // Phylogeny stats
        let phylo_stats = world.phylogeny.statistics();
        writeln!(file, "=== Phylogeny Statistics ===")?;
        writeln!(file, "Total Organisms Tracked: {}", phylo_stats.total_organisms)?;
        writeln!(file, "Currently Alive: {}", phylo_stats.alive_organisms)?;
        writeln!(file, "Total Deaths: {}", phylo_stats.dead_organisms)?;
        writeln!(file, "Unique Lineages: {}", phylo_stats.unique_lineages)?;
        writeln!(file, "Average Lifespan: {:.1}", phylo_stats.average_lifespan)?;
        writeln!(file, "Average Offspring: {:.2}", phylo_stats.average_offspring)?;
        writeln!(file)?;

        // Sexual reproduction stats
        writeln!(file, "=== Sexual Reproduction Statistics ===")?;
        writeln!(file, "Total Matings: {}", world.sexual_reproduction.total_matings)?;
        writeln!(file, "Failed Matings: {}", world.sexual_reproduction.failed_matings)?;
        writeln!(file, "Inbreeding Events: {}", world.sexual_reproduction.inbreeding_events)?;
        writeln!(file, "Total Offspring: {}", world.sexual_reproduction.total_offspring)?;
        writeln!(file, "Success Rate: {:.1}%", world.sexual_reproduction.success_rate() * 100.0)?;
        writeln!(file, "Inbreeding Rate: {:.1}%", world.sexual_reproduction.inbreeding_rate() * 100.0)?;

        Ok(())
    }
}

/// Manifest of exported files
#[derive(Debug)]
pub struct ExportManifest {
    pub time: u64,
    pub organisms_file: String,
    pub phylogeny_file: String,
    pub newick_file: String,
    pub snapshot_file: String,
}

/// Simplified organism snapshot for JSON export
#[derive(serde::Serialize)]
struct OrganismSnapshot {
    id: u64,
    x: u8,
    y: u8,
    energy: f32,
    size: f32,
    brain_complexity: usize,
    age: u32,
    lineage_id: u32,
    generation: u16,
    sex: String,
    kills: u16,
    offspring: u16,
    is_predator: bool,
}

impl From<&Organism> for OrganismSnapshot {
    fn from(org: &Organism) -> Self {
        Self {
            id: org.id,
            x: org.x,
            y: org.y,
            energy: org.energy,
            size: org.size,
            brain_complexity: org.brain.complexity(),
            age: org.age,
            lineage_id: org.lineage_id,
            generation: org.generation,
            sex: format!("{:?}", org.sex),
            kills: org.kills,
            offspring: org.offspring_count,
            is_predator: org.is_predator,
        }
    }
}

/// World snapshot for JSON export
#[derive(serde::Serialize)]
struct WorldSnapshot {
    time: u64,
    population: usize,
    generation_max: u16,
    organisms: Vec<OrganismSnapshot>,
    season: String,
    total_food: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_export_organisms_csv() {
        let config = Config::default();
        let organisms: Vec<Organism> = (0..5)
            .map(|i| Organism::new(i, i as u32, 10, 10, &config))
            .collect();

        let dir = tempdir().unwrap();
        let path = dir.path().join("organisms.csv");

        ExportSystem::export_organisms_csv(&organisms, &path).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("id,x,y,energy"));
        assert!(content.lines().count() > 1);
    }

    #[test]
    fn test_export_phylogeny_csv() {
        let mut phylogeny = PhylogeneticTree::new();
        phylogeny.record_birth(1, None, None, 0, 5, 100.0, 1.0, 1, 0, 12345);
        phylogeny.record_birth(2, Some(1), None, 100, 6, 80.0, 1.1, 1, 1, 12346);

        let dir = tempdir().unwrap();
        let path = dir.path().join("phylogeny.csv");

        ExportSystem::export_phylogeny_csv(&phylogeny, &path).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("organism_id,parent1_id"));
        assert!(content.contains("1,"));
        assert!(content.contains("2,1"));
    }

    #[test]
    fn test_export_phylogeny_newick() {
        let mut phylogeny = PhylogeneticTree::new();
        phylogeny.record_birth(1, None, None, 0, 5, 100.0, 1.0, 1, 0, 12345);

        let dir = tempdir().unwrap();
        let path = dir.path().join("phylogeny.newick");

        ExportSystem::export_phylogeny_newick(&phylogeny, &path).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("1:0"));
    }

    #[test]
    fn test_export_world_json() {
        let config = Config::default();
        let world = World::new(config);

        let dir = tempdir().unwrap();
        let path = dir.path().join("snapshot.json");

        ExportSystem::export_world_json(&world, &path).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"time\":"));
        assert!(content.contains("\"population\":"));
    }

    #[test]
    fn test_export_full_state() {
        let config = Config::default();
        let world = World::new(config);

        let dir = tempdir().unwrap();
        let base_path = dir.path().join("export");

        let manifest = ExportSystem::export_full_state(&world, &base_path).unwrap();

        assert!(Path::new(&manifest.organisms_file).exists());
        assert!(Path::new(&manifest.phylogeny_file).exists());
        assert!(Path::new(&manifest.newick_file).exists());
        assert!(Path::new(&manifest.snapshot_file).exists());
    }
}
