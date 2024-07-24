use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Dimension {
    pub name: String,
    pub weight: f64,
}
