use serde::{Deserialize, Serialize};
use statrs::statistics::{Data, Distribution, Max, Min, OrderStatistics, Statistics as StatrsStats};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

impl Statistics {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::zeros();
        }

        let mut data = Data::new(samples.to_vec());

        Self {
            mean: data.mean().unwrap_or(0.0),
            median: data.median(),
            p90: data.quantile(0.90),
            p95: data.quantile(0.95),
            p99: data.quantile(0.99),
            p999: data.quantile(0.999),
            min: data.min(),
            max: data.max(),
            stddev: data.std_dev().unwrap_or(0.0),
        }
    }

    pub fn zeros() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            p999: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Statistics::from_samples(&samples);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_empty_statistics() {
        let samples: Vec<f64> = vec![];
        let stats = Statistics::from_samples(&samples);

        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.median, 0.0);
    }
}
