use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Point {
    pub id: String,
    pub name: String,
    pub coordinates: Vec<f64>,
    pub labels: HashMap<String, u64>,
}

impl Point {
    pub fn generate_unique_id(&self) -> String {
        uuid::Uuid::new_v4().to_string()
    }

    pub fn calculate_length(point: &Point) -> f64 {
        let sum = point.coordinates.iter().fold(0.0, |acc, x| acc + x * x);
        f64::sqrt(sum)
    }
}
