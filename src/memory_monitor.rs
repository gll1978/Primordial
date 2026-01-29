//! Memory monitoring for safety and crash prevention.
//!
//! Monitors process memory usage and provides warnings/actions when
//! memory usage exceeds configured thresholds.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Memory usage statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// Resident set size in bytes (actual physical memory used)
    pub rss_bytes: u64,
    /// Virtual memory size in bytes
    pub vms_bytes: u64,
    /// Percentage of total system memory used by this process
    pub percent_used: f32,
}

/// Memory thresholds for warnings and actions
#[derive(Debug, Clone, Copy)]
pub struct MemoryThresholds {
    /// Warning threshold (percentage of system memory)
    pub warning_percent: f32,
    /// Critical threshold (percentage) - triggers auto-pause
    pub critical_percent: f32,
    /// Absolute limit in MB (0 = no limit)
    pub max_memory_mb: u64,
}

impl Default for MemoryThresholds {
    fn default() -> Self {
        Self {
            warning_percent: 70.0,
            critical_percent: 85.0,
            max_memory_mb: 0, // No limit by default
        }
    }
}

/// Memory monitor state
pub struct MemoryMonitor {
    thresholds: MemoryThresholds,
    total_system_memory: u64,
    last_warning_time: AtomicU64,
    warning_issued: AtomicBool,
    critical_issued: AtomicBool,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new(thresholds: MemoryThresholds) -> Self {
        let total_system_memory = get_total_system_memory().unwrap_or(16 * 1024 * 1024 * 1024); // Default 16GB
        Self {
            thresholds,
            total_system_memory,
            last_warning_time: AtomicU64::new(0),
            warning_issued: AtomicBool::new(false),
            critical_issued: AtomicBool::new(false),
        }
    }

    /// Get current memory usage
    pub fn get_stats(&self) -> MemoryStats {
        let (rss, vms) = get_process_memory().unwrap_or((0, 0));
        let percent = if self.total_system_memory > 0 {
            (rss as f64 / self.total_system_memory as f64 * 100.0) as f32
        } else {
            0.0
        };

        MemoryStats {
            rss_bytes: rss,
            vms_bytes: vms,
            percent_used: percent,
        }
    }

    /// Check memory and return action needed
    pub fn check(&self, current_time: u64) -> MemoryAction {
        let stats = self.get_stats();

        // Check absolute limit first
        if self.thresholds.max_memory_mb > 0 {
            let current_mb = stats.rss_bytes / (1024 * 1024);
            if current_mb >= self.thresholds.max_memory_mb {
                self.critical_issued.store(true, Ordering::Relaxed);
                return MemoryAction::Critical(stats);
            }
        }

        // Check percentage thresholds
        if stats.percent_used >= self.thresholds.critical_percent {
            self.critical_issued.store(true, Ordering::Relaxed);
            return MemoryAction::Critical(stats);
        }

        if stats.percent_used >= self.thresholds.warning_percent {
            // Only warn once per 1000 time units to avoid spam
            let last_warning = self.last_warning_time.load(Ordering::Relaxed);
            if current_time - last_warning >= 1000 {
                self.last_warning_time.store(current_time, Ordering::Relaxed);
                self.warning_issued.store(true, Ordering::Relaxed);
                return MemoryAction::Warning(stats);
            }
        }

        // Reset critical flag if we're back below warning
        if stats.percent_used < self.thresholds.warning_percent {
            self.critical_issued.store(false, Ordering::Relaxed);
            self.warning_issued.store(false, Ordering::Relaxed);
        }

        MemoryAction::Ok(stats)
    }

    /// Check if we're in critical state
    pub fn is_critical(&self) -> bool {
        self.critical_issued.load(Ordering::Relaxed)
    }

    /// Get thresholds
    pub fn thresholds(&self) -> &MemoryThresholds {
        &self.thresholds
    }

    /// Update thresholds
    pub fn set_thresholds(&mut self, thresholds: MemoryThresholds) {
        self.thresholds = thresholds;
    }

    /// Get total system memory
    pub fn total_system_memory(&self) -> u64 {
        self.total_system_memory
    }
}

/// Action to take based on memory check
#[derive(Debug, Clone, Copy)]
pub enum MemoryAction {
    /// Memory usage is OK
    Ok(MemoryStats),
    /// Memory usage is high - warning
    Warning(MemoryStats),
    /// Memory usage is critical - should pause/checkpoint
    Critical(MemoryStats),
}

impl MemoryAction {
    pub fn stats(&self) -> MemoryStats {
        match self {
            MemoryAction::Ok(s) | MemoryAction::Warning(s) | MemoryAction::Critical(s) => *s,
        }
    }

    pub fn is_critical(&self) -> bool {
        matches!(self, MemoryAction::Critical(_))
    }

    pub fn is_warning(&self) -> bool {
        matches!(self, MemoryAction::Warning(_))
    }
}

/// Get process memory usage (RSS, VMS) in bytes
#[cfg(target_os = "linux")]
fn get_process_memory() -> Option<(u64, u64)> {
    // Read /proc/self/statm
    // Format: size resident shared text lib data dt
    // Values are in pages (typically 4KB)
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let parts: Vec<&str> = statm.split_whitespace().collect();

    if parts.len() < 2 {
        return None;
    }

    let page_size = 4096u64; // Standard page size on Linux
    let vms_pages: u64 = parts[0].parse().ok()?;
    let rss_pages: u64 = parts[1].parse().ok()?;

    Some((rss_pages * page_size, vms_pages * page_size))
}

#[cfg(not(target_os = "linux"))]
fn get_process_memory() -> Option<(u64, u64)> {
    // Fallback for non-Linux systems
    None
}

/// Get total system memory in bytes
#[cfg(target_os = "linux")]
fn get_total_system_memory() -> Option<u64> {
    // Read /proc/meminfo
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: u64 = parts[1].parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn get_total_system_memory() -> Option<u64> {
    // Fallback: assume 16GB
    Some(16 * 1024 * 1024 * 1024)
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let monitor = MemoryMonitor::new(MemoryThresholds::default());
        let stats = monitor.get_stats();

        // Should have some memory usage
        println!("RSS: {}", format_bytes(stats.rss_bytes));
        println!("VMS: {}", format_bytes(stats.vms_bytes));
        println!("Percent: {:.1}%", stats.percent_used);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
