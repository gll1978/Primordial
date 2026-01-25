//! Simulation logging for analysis and parameter tuning.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use crate::gui::commands::SimSettings;
use crate::gui::snapshot::WorldSnapshot;

/// A single data point in the simulation history
#[derive(Clone, Debug)]
pub struct LogEntry {
    pub time: u64,
    pub population: usize,
    pub generation_max: u16,
    pub energy_mean: f32,
    pub health_mean: f32,
    pub lineage_count: usize,
    pub total_food: f32,
    pub brain_mean: f32,
    pub births: usize,
    pub deaths: usize,
}

impl LogEntry {
    pub fn from_snapshot(snapshot: &WorldSnapshot) -> Self {
        Self {
            time: snapshot.time,
            population: snapshot.stats.population,
            generation_max: snapshot.stats.generation_max,
            energy_mean: snapshot.stats.energy_mean,
            health_mean: snapshot.stats.health_mean,
            lineage_count: snapshot.stats.lineage_count,
            total_food: snapshot.stats.total_food,
            brain_mean: snapshot.stats.brain_mean,
            births: snapshot.stats.births,
            deaths: snapshot.stats.deaths,
        }
    }

    pub fn csv_header() -> &'static str {
        "time,population,generation_max,energy_mean,health_mean,lineage_count,total_food,brain_mean,births,deaths"
    }

    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.2},{},{:.2},{:.2},{},{}",
            self.time,
            self.population,
            self.generation_max,
            self.energy_mean,
            self.health_mean,
            self.lineage_count,
            self.total_food,
            self.brain_mean,
            self.births,
            self.deaths
        )
    }
}

/// Complete simulation log
#[derive(Clone, Debug)]
pub struct SimulationLog {
    /// Unique run ID (timestamp-based)
    pub run_id: String,
    /// Settings used for this run
    pub settings: SimSettings,
    /// Time series data
    pub entries: Vec<LogEntry>,
    /// Log interval (record every N steps)
    pub log_interval: u64,
    /// Last recorded time
    last_time: u64,
}

impl SimulationLog {
    /// Create a new simulation log
    pub fn new(settings: SimSettings) -> Self {
        let run_id = chrono_lite_timestamp();
        Self {
            run_id,
            settings,
            entries: Vec::new(),
            log_interval: 50, // Record every 50 steps
            last_time: 0,
        }
    }

    /// Record a snapshot if enough time has passed
    pub fn record(&mut self, snapshot: &WorldSnapshot) {
        if snapshot.time >= self.last_time + self.log_interval || self.entries.is_empty() {
            self.entries.push(LogEntry::from_snapshot(snapshot));
            self.last_time = snapshot.time;
        }
    }

    /// Force record current state
    pub fn force_record(&mut self, snapshot: &WorldSnapshot) {
        self.entries.push(LogEntry::from_snapshot(snapshot));
        self.last_time = snapshot.time;
    }

    /// Clear the log for a new run
    pub fn reset(&mut self, settings: SimSettings) {
        self.run_id = chrono_lite_timestamp();
        self.settings = settings;
        self.entries.clear();
        self.last_time = 0;
    }

    /// Get summary statistics
    pub fn summary(&self) -> LogSummary {
        if self.entries.is_empty() {
            return LogSummary::default();
        }

        let last = self.entries.last().unwrap();
        let max_pop = self.entries.iter().map(|e| e.population).max().unwrap_or(0);
        let max_gen = self.entries.iter().map(|e| e.generation_max).max().unwrap_or(0);
        let avg_pop = self.entries.iter().map(|e| e.population).sum::<usize>() as f32
            / self.entries.len() as f32;

        LogSummary {
            total_steps: last.time,
            final_population: last.population,
            max_population: max_pop,
            avg_population: avg_pop,
            max_generation: max_gen,
            extinct: last.population == 0,
            data_points: self.entries.len(),
        }
    }

    /// Export to CSV file
    pub fn export_csv(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Write header comment with settings
        writeln!(file, "# PRIMORDIAL Simulation Log")?;
        writeln!(file, "# Run ID: {}", self.run_id)?;
        writeln!(file, "# Settings:")?;
        writeln!(file, "#   max_population: {}", self.settings.max_population)?;
        writeln!(file, "#   initial_population: {}", self.settings.initial_population)?;
        writeln!(file, "#   max_steps: {}", self.settings.max_steps)?;
        writeln!(file, "#   grid_size: {}", self.settings.grid_size)?;
        writeln!(file, "#   mutation_rate: {}", self.settings.mutation_rate)?;
        writeln!(file, "#   mutation_strength: {}", self.settings.mutation_strength)?;
        writeln!(file, "#   food_regen_rate: {}", self.settings.food_regen_rate)?;
        writeln!(file, "#   reproduction_threshold: {}", self.settings.reproduction_threshold)?;
        writeln!(file, "#   predation_enabled: {}", self.settings.predation_enabled)?;
        writeln!(file, "#   seasons_enabled: {}", self.settings.seasons_enabled)?;
        writeln!(file, "#   terrain_enabled: {}", self.settings.terrain_enabled)?;
        writeln!(file, "#")?;

        // Write CSV header
        writeln!(file, "{}", LogEntry::csv_header())?;

        // Write data rows
        for entry in &self.entries {
            writeln!(file, "{}", entry.to_csv_row())?;
        }

        Ok(())
    }

    /// Export to JSON file
    pub fn export_json(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"run_id\": \"{}\",", self.run_id)?;
        writeln!(file, "  \"settings\": {{")?;
        writeln!(file, "    \"max_population\": {},", self.settings.max_population)?;
        writeln!(file, "    \"initial_population\": {},", self.settings.initial_population)?;
        writeln!(file, "    \"max_steps\": {},", self.settings.max_steps)?;
        writeln!(file, "    \"grid_size\": {},", self.settings.grid_size)?;
        writeln!(file, "    \"mutation_rate\": {},", self.settings.mutation_rate)?;
        writeln!(file, "    \"mutation_strength\": {},", self.settings.mutation_strength)?;
        writeln!(file, "    \"food_regen_rate\": {},", self.settings.food_regen_rate)?;
        writeln!(file, "    \"reproduction_threshold\": {},", self.settings.reproduction_threshold)?;
        writeln!(file, "    \"predation_enabled\": {},", self.settings.predation_enabled)?;
        writeln!(file, "    \"seasons_enabled\": {},", self.settings.seasons_enabled)?;
        writeln!(file, "    \"terrain_enabled\": {}", self.settings.terrain_enabled)?;
        writeln!(file, "  }},")?;

        let summary = self.summary();
        writeln!(file, "  \"summary\": {{")?;
        writeln!(file, "    \"total_steps\": {},", summary.total_steps)?;
        writeln!(file, "    \"final_population\": {},", summary.final_population)?;
        writeln!(file, "    \"max_population\": {},", summary.max_population)?;
        writeln!(file, "    \"avg_population\": {:.2},", summary.avg_population)?;
        writeln!(file, "    \"max_generation\": {},", summary.max_generation)?;
        writeln!(file, "    \"extinct\": {},", summary.extinct)?;
        writeln!(file, "    \"data_points\": {}", summary.data_points)?;
        writeln!(file, "  }},")?;

        writeln!(file, "  \"data\": [")?;
        for (i, entry) in self.entries.iter().enumerate() {
            let comma = if i < self.entries.len() - 1 { "," } else { "" };
            writeln!(file, "    {{")?;
            writeln!(file, "      \"time\": {},", entry.time)?;
            writeln!(file, "      \"population\": {},", entry.population)?;
            writeln!(file, "      \"generation_max\": {},", entry.generation_max)?;
            writeln!(file, "      \"energy_mean\": {:.2},", entry.energy_mean)?;
            writeln!(file, "      \"health_mean\": {:.2},", entry.health_mean)?;
            writeln!(file, "      \"lineage_count\": {},", entry.lineage_count)?;
            writeln!(file, "      \"total_food\": {:.2},", entry.total_food)?;
            writeln!(file, "      \"brain_mean\": {:.2},", entry.brain_mean)?;
            writeln!(file, "      \"births\": {},", entry.births)?;
            writeln!(file, "      \"deaths\": {}", entry.deaths)?;
            writeln!(file, "    }}{}", comma)?;
        }
        writeln!(file, "  ]")?;
        writeln!(file, "}}")?;

        Ok(())
    }

    /// Get default log directory
    pub fn log_directory() -> PathBuf {
        let dir = PathBuf::from("logs");
        if !dir.exists() {
            let _ = fs::create_dir_all(&dir);
        }
        dir
    }

    /// Auto-generate filename based on run_id
    pub fn auto_filename(&self, extension: &str) -> PathBuf {
        Self::log_directory().join(format!("sim_{}.{}", self.run_id, extension))
    }
}

/// Summary statistics for a simulation run
#[derive(Clone, Debug, Default)]
pub struct LogSummary {
    pub total_steps: u64,
    pub final_population: usize,
    pub max_population: usize,
    pub avg_population: f32,
    pub max_generation: u16,
    pub extinct: bool,
    pub data_points: usize,
}

/// Generate a timestamp string for run IDs (without chrono dependency)
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    format!("{}", duration.as_secs())
}

/// Log panel UI component
pub struct LogPanel {
    /// The simulation log
    pub log: SimulationLog,
    /// Auto-save on reset/extinction
    pub auto_save: bool,
    /// Export format
    pub export_format: ExportFormat,
    /// Last export message
    pub last_message: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ExportFormat {
    Csv,
    Json,
}

impl Default for LogPanel {
    fn default() -> Self {
        Self {
            log: SimulationLog::new(SimSettings::default()),
            auto_save: true,
            export_format: ExportFormat::Csv,
            last_message: None,
        }
    }
}

impl LogPanel {
    pub fn new(settings: SimSettings) -> Self {
        Self {
            log: SimulationLog::new(settings),
            auto_save: true,
            export_format: ExportFormat::Csv,
            last_message: None,
        }
    }

    /// Record snapshot data
    pub fn record(&mut self, snapshot: &WorldSnapshot) {
        self.log.record(snapshot);
    }

    /// Reset log for new simulation
    pub fn reset(&mut self, settings: SimSettings) {
        // Auto-save before reset if enabled and has data
        if self.auto_save && !self.log.entries.is_empty() {
            self.export();
        }
        self.log.reset(settings);
        self.last_message = None;
    }

    /// Export log to file
    pub fn export(&mut self) {
        if self.log.entries.is_empty() {
            self.last_message = Some("No data to export".to_string());
            return;
        }

        let result = match self.export_format {
            ExportFormat::Csv => {
                let path = self.log.auto_filename("csv");
                self.log.export_csv(&path).map(|_| path)
            }
            ExportFormat::Json => {
                let path = self.log.auto_filename("json");
                self.log.export_json(&path).map(|_| path)
            }
        };

        match result {
            Ok(path) => {
                self.last_message = Some(format!("Saved: {}", path.display()));
            }
            Err(e) => {
                self.last_message = Some(format!("Error: {}", e));
            }
        }
    }

    /// Show the log panel UI
    pub fn show(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Logging & Export")
            .default_open(false)
            .show(ui, |ui| {
                let summary = self.log.summary();

                // Current run info
                ui.horizontal(|ui| {
                    ui.label("Run ID:");
                    ui.label(&self.log.run_id);
                });

                ui.horizontal(|ui| {
                    ui.label("Data points:");
                    ui.label(format!("{}", summary.data_points));
                });

                ui.separator();

                // Summary stats
                if summary.data_points > 0 {
                    egui::Grid::new("log_summary")
                        .num_columns(2)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Total steps:");
                            ui.label(format!("{}", summary.total_steps));
                            ui.end_row();

                            ui.label("Max population:");
                            ui.label(format!("{}", summary.max_population));
                            ui.end_row();

                            ui.label("Avg population:");
                            ui.label(format!("{:.1}", summary.avg_population));
                            ui.end_row();

                            ui.label("Max generation:");
                            ui.label(format!("{}", summary.max_generation));
                            ui.end_row();

                            ui.label("Status:");
                            if summary.extinct {
                                ui.colored_label(egui::Color32::RED, "EXTINCT");
                            } else {
                                ui.colored_label(egui::Color32::GREEN, "ALIVE");
                            }
                            ui.end_row();
                        });

                    ui.separator();
                }

                // Export options
                ui.horizontal(|ui| {
                    ui.label("Format:");
                    ui.radio_value(&mut self.export_format, ExportFormat::Csv, "CSV");
                    ui.radio_value(&mut self.export_format, ExportFormat::Json, "JSON");
                });

                ui.checkbox(&mut self.auto_save, "Auto-save on reset");

                if ui.button("Export Now").clicked() {
                    self.export();
                }

                ui.label(format!("Logs folder: {}", SimulationLog::log_directory().display()));

                // Status message
                if let Some(ref msg) = self.last_message {
                    ui.separator();
                    ui.label(msg);
                }
            });
    }
}
