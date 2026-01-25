//! Checkpoint system for saving and loading simulation state.

use crate::config::Config;
use crate::grid::FoodGrid;
use crate::organism::Organism;
use crate::stats::{Stats, LineageTracker};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Complete simulation state for checkpointing
#[derive(Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Version for compatibility checking
    pub version: u32,
    /// Simulation time
    pub time: u64,
    /// Configuration
    pub config: Config,
    /// All organisms
    pub organisms: Vec<Organism>,
    /// Food grid
    pub food_grid: FoodGrid,
    /// Current statistics
    pub stats: Stats,
    /// Lineage tracker
    pub lineage_tracker: LineageTracker,
    /// Next organism ID
    pub next_organism_id: u64,
    /// Random seed (for reproducibility)
    pub random_seed: u64,
}

impl Checkpoint {
    /// Current checkpoint version
    pub const VERSION: u32 = 1;

    /// Create a new checkpoint
    pub fn new(
        time: u64,
        config: Config,
        organisms: Vec<Organism>,
        food_grid: FoodGrid,
        stats: Stats,
        lineage_tracker: LineageTracker,
        next_organism_id: u64,
        random_seed: u64,
    ) -> Self {
        Self {
            version: Self::VERSION,
            time,
            config,
            organisms,
            food_grid,
            stats,
            lineage_tracker,
            next_organism_id,
            random_seed,
        }
    }

    /// Save checkpoint to binary file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), CheckpointError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes for identification
        writer.write_all(b"PRMD")?;

        // Serialize and write
        let encoded = bincode::serialize(self)?;
        writer.write_all(&encoded)?;

        Ok(())
    }

    /// Load checkpoint from binary file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CheckpointError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Check magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"PRMD" {
            return Err(CheckpointError::InvalidFormat("Invalid magic bytes".to_string()));
        }

        // Read and deserialize
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        let checkpoint: Checkpoint = bincode::deserialize(&buffer)?;

        // Version check
        if checkpoint.version != Self::VERSION {
            return Err(CheckpointError::VersionMismatch {
                expected: Self::VERSION,
                found: checkpoint.version,
            });
        }

        Ok(checkpoint)
    }

    /// Save checkpoint with compression (optional future enhancement)
    pub fn save_compressed<P: AsRef<Path>>(&self, path: P) -> Result<(), CheckpointError> {
        // For now, just use regular save
        // In future: use flate2 or zstd compression
        self.save(path)
    }

    /// Get approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap_or(0) as usize
    }
}

/// Errors that can occur during checkpoint operations
#[derive(Debug)]
pub enum CheckpointError {
    Io(std::io::Error),
    Serialization(bincode::Error),
    InvalidFormat(String),
    VersionMismatch { expected: u32, found: u32 },
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Serialization(e) => write!(f, "Serialization error: {}", e),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            Self::VersionMismatch { expected, found } => {
                write!(f, "Version mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<bincode::Error> for CheckpointError {
    fn from(e: bincode::Error) -> Self {
        Self::Serialization(e)
    }
}

/// Checkpoint manager for automatic saving
pub struct CheckpointManager {
    /// Base directory for checkpoints
    pub base_dir: String,
    /// Interval between checkpoints
    pub interval: u64,
    /// Maximum checkpoints to keep
    pub max_checkpoints: usize,
    /// Last checkpoint time
    last_checkpoint: u64,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(base_dir: String, interval: u64, max_checkpoints: usize) -> Self {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&base_dir).ok();

        Self {
            base_dir,
            interval,
            max_checkpoints,
            last_checkpoint: 0,
        }
    }

    /// Check if a checkpoint should be saved
    pub fn should_save(&self, time: u64) -> bool {
        time > 0 && time % self.interval == 0 && time != self.last_checkpoint
    }

    /// Generate checkpoint filename
    pub fn checkpoint_path(&self, time: u64) -> String {
        format!("{}/checkpoint_{:08}.bin", self.base_dir, time)
    }

    /// Save checkpoint and update state
    pub fn save(&mut self, checkpoint: &Checkpoint) -> Result<String, CheckpointError> {
        let path = self.checkpoint_path(checkpoint.time);
        checkpoint.save(&path)?;
        self.last_checkpoint = checkpoint.time;

        // Cleanup old checkpoints
        self.cleanup()?;

        Ok(path)
    }

    /// Remove old checkpoints beyond max limit
    fn cleanup(&self) -> Result<(), CheckpointError> {
        let mut checkpoints: Vec<_> = std::fs::read_dir(&self.base_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("checkpoint_")
            })
            .collect();

        if checkpoints.len() > self.max_checkpoints {
            // Sort by name (which includes time)
            checkpoints.sort_by_key(|e| e.file_name());

            // Remove oldest
            let to_remove = checkpoints.len() - self.max_checkpoints;
            for entry in checkpoints.into_iter().take(to_remove) {
                std::fs::remove_file(entry.path())?;
            }
        }

        Ok(())
    }

    /// Find latest checkpoint in directory
    pub fn find_latest(&self) -> Option<String> {
        std::fs::read_dir(&self.base_dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("checkpoint_")
            })
            .max_by_key(|e| e.file_name())
            .map(|e| e.path().to_string_lossy().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_checkpoint() -> Checkpoint {
        let config = Config::default();
        Checkpoint::new(
            1000,
            config.clone(),
            vec![Organism::new(1, 1, 10, 10, &config)],
            FoodGrid::new(80, 50.0),
            Stats::default(),
            LineageTracker::new(),
            2,
            12345,
        )
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let checkpoint = create_test_checkpoint();
        let temp_path = "/tmp/test_checkpoint.bin";

        checkpoint.save(temp_path).unwrap();
        let loaded = Checkpoint::load(temp_path).unwrap();

        assert_eq!(loaded.time, checkpoint.time);
        assert_eq!(loaded.organisms.len(), checkpoint.organisms.len());
        assert_eq!(loaded.random_seed, checkpoint.random_seed);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_checkpoint_size() {
        let checkpoint = create_test_checkpoint();
        let size = checkpoint.size_bytes();

        // Should be reasonably small
        assert!(size > 0);
        assert!(size < 1_000_000); // Less than 1MB for single organism
    }
}
