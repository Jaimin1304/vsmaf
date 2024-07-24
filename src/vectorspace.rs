use crate::{dimension::Dimension, point::Point};
use linfa::{
    dataset::Labels,
    prelude::SilhouetteScore,
    traits::{Fit, Predict, Transformer},
    Dataset, DatasetBase,
};
use linfa_clustering::{Dbscan, KMeans};
use linfa_reduction::Pca;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug)]
pub struct VectorSpace {
    pub dimensions: Vec<Dimension>,
    pub points: HashMap<String, Point>,
}

impl VectorSpace {
    pub fn add_point(&mut self, mut point: Point) -> Result<(), String> {
        if point.coordinates.len() != self.dimensions.len() {
            let e = format!(
                "Point dimensions ({}) do not match VectorSpace dimensions ({})",
                point.coordinates.len(),
                self.dimensions.len()
            );
            return Err(e);
        }

        while self.points.contains_key(&point.id) {
            point.id = point.generate_unique_id();
        }
        self.points.insert(point.id.clone(), point);
        Ok(())
    }

    pub fn remove_point(&mut self, point_id: &str) {
        self.points.remove(point_id);
    }

    pub fn calculate_distance(&self, point1_id: &str, point2_id: &str) -> Result<f64, String> {
        if !self.points.contains_key(point1_id) || !self.points.contains_key(point2_id) {
            return Err("One or both point IDs are not found in VectorSpace.".to_string());
        }

        let point1 = self.points.get(point1_id).unwrap();
        let point2 = self.points.get(point2_id).unwrap();
        let ret: Vec<((&f64, &f64), &Dimension)> = point1
            .coordinates
            .iter()
            .zip(point2.coordinates.iter())
            .zip(self.dimensions.iter())
            .collect();
        let mut sum = 0.0;
        for ((p1, p2), dim) in ret {
            sum += (p1 - p2).exp2() * dim.weight;
        }
        Ok(f64::sqrt(sum))
    }

    pub fn find_points_within_radius(
        &self,
        center_point_id: &str,
        radius: f64,
    ) -> Result<Vec<Point>, String> {
        if !self.points.contains_key(center_point_id) {
            return Err(format!(
                "No point with id {} found in VectorSpace.",
                center_point_id
            ));
        }
        let mut points_within_radius: Vec<Point> = Vec::new();

        for point in self.points.values() {
            let dis = self.calculate_distance(center_point_id, &point.id);
            if dis.is_err() {
                println!("{}", dis.unwrap_err());
                continue;
            }
            let dis = dis.unwrap();
            if dis <= radius {
                points_within_radius.push(point.clone());
            }
        }

        Ok(points_within_radius)
    }
    pub fn sort_points_by_dimension(
        &self,
        dimension_name: &str,
        ascending: bool,
    ) -> Result<Vec<Point>, String> {
        let pos = self
            .dimensions
            .iter()
            .position(|d| d.name == dimension_name);
        if pos.is_none() {
            return Err(format!("Dimension name {} not found.", dimension_name));
        }
        let dimension_index = pos.unwrap();

        let mut points: Vec<Point> = self.points.iter().map(|(_, v)| v.clone()).collect();
        points.sort_by(|a, b| {
            if ascending {
                a.coordinates[dimension_index].total_cmp(&b.coordinates[dimension_index])
            } else {
                b.coordinates[dimension_index].total_cmp(&a.coordinates[dimension_index])
            }
        });

        Ok(points)
    }

    pub fn filter_points_by_ranges(&self, ranges: Vec<(f64, f64)>) -> Result<Vec<Point>, String> {
        if ranges.len() != self.dimensions.len() {
            return Err(
                "The size of ranges must match the number of dimensions in the VectorSpace."
                    .to_string(),
            );
        }

        for (min, max) in ranges.iter() {
            if min > max {
                return Err(format!(
                    "Invalid range: [{}, {}]. Min value cannot be greater than max value.",
                    min, max
                ));
            }
        }

        let mut filtered_points = Vec::new();

        for point in self.points.values() {
            let mut include_point = true;

            for (dim_index, (min, max)) in ranges.iter().enumerate() {
                if point.coordinates[dim_index] < *min || point.coordinates[dim_index] > *max {
                    include_point = false;
                    break;
                }
            }

            if include_point {
                filtered_points.push(point.clone());
            }
        }

        Ok(filtered_points)
    }

    pub fn pca_transform(&self, n_components: usize) -> Result<VectorSpace, String> {
        if n_components > self.dimensions.len() {
            return Err(
                "n_components cannot be greater than the number of dimensions in the vector space."
                    .to_string(),
            );
        }
        let data: Vec<Vec<f64>> = self
            .points
            .iter()
            .map(|(_, p)| p.coordinates.clone())
            .collect();

        let rown = data.len();
        let coln = data[0].len();

        let mut arr = ndarray::Array2::zeros((rown, coln));

        for r in 0..rown {
            for c in 0..coln {
                arr[[r, c]] = data[r][c];
            }
        }
        let target = ndarray::Array2::<f64>::zeros((rown, coln));
        let dataset = Dataset::new(arr, target);

        let pca = Pca::params(n_components);
        let embedding: Pca<f64> = pca.fit(&dataset).unwrap();

        let transformed_data = embedding.predict(&dataset);

        let (row, col) = transformed_data.dim();

        let mut tran_data = Vec::new();
        for r in 0..row {
            let col_vec: Vec<f64> = (0..col).map(|c| transformed_data[(r, c)]).collect();
            tran_data.push(col_vec);
        }

        let mut new_dimensions = Vec::new();
        (0..n_components).into_iter().for_each(|i| {
            new_dimensions.push(Dimension {
                name: format!("Dim_{}", i + 1),
                weight: 1.0,
            })
        });

        let mut new_space = VectorSpace {
            dimensions: new_dimensions,
            points: HashMap::new(),
        };
        // Add transformed points to the new VectorSpace
        for ((_, point), new_coords) in self.points.iter().zip(tran_data.iter()) {
            let new_point = Point {
                id: point.id.clone(),
                name: point.name.clone(),
                coordinates: new_coords.clone(),
                labels: HashMap::new(),
            };

            let ret = new_space.add_point(new_point);
            if ret.is_err() {
                println!("{}", ret.unwrap_err());
            }
        }

        Ok(new_space)
    }

    pub fn tsne_transform(
        &self,
        n_components: usize,
        perplexity: f64,
        max_iter: usize,
    ) -> Result<VectorSpace, String> {
        if n_components > self.dimensions.len() {
            return Err(
                "n_components cannot be greater than the number of dimensions in the vector space."
                    .to_string(),
            );
        }

        // Collect coordinates of all points
        let data: Vec<Vec<f64>> = self
            .points
            .iter()
            .map(|(_, p)| p.coordinates.clone())
            .collect();
        if perplexity >= data.len() as f64 {
            return Err("perplexity must be less than the number of samples!".to_string());
        }
        let rown = data.len();
        let coln = data[0].len();
        let mut arr = ndarray::Array2::zeros((rown, coln));
        for r in 0..rown {
            for c in 0..coln {
                arr[[r, c]] = data[r][c];
            }
        }
        let target = ndarray::Array2::<f64>::zeros((rown, coln));
        let dataset = Dataset::new(arr, target);
        // Perform t-SNE
        let ds = linfa_tsne::TSneParams::embedding_size(2)
            .perplexity(perplexity)
            .max_iter(max_iter)
            .transform(dataset)
            .unwrap();

        let mut tran_data = Vec::new();
        for (x, _y) in ds.sample_iter() {
            let mut tmp = Vec::new();
            tmp.push(x[0]);
            tmp.push(x[1]);
            tran_data.push(tmp);
        }

        let mut new_dimensions = Vec::new();
        (0..n_components).into_iter().for_each(|i| {
            new_dimensions.push(Dimension {
                name: format!("Dim_{}", i + 1),
                weight: 1.0,
            })
        });
        let mut new_space = VectorSpace {
            dimensions: new_dimensions,
            points: HashMap::new(),
        };
        // Add transformed points to the new VectorSpace
        for ((_, point), new_coords) in self.points.iter().zip(tran_data.iter()) {
            let new_point = Point {
                id: point.id.clone(),
                name: point.name.clone(),
                coordinates: new_coords.clone(),
                labels: HashMap::new(),
            };

            let ret = new_space.add_point(new_point);
            if ret.is_err() {
                println!("{}", ret.unwrap_err());
            }
        }

        Ok(new_space)
    }

    pub fn perform_kmeans(&mut self, n_clusters: usize) -> Vec<u64> {
        // Collect coordinates of all points
        let data: Vec<Vec<f64>> = self
            .points
            .iter()
            .map(|(_, p)| p.coordinates.clone())
            .collect();

        let rown = data.len();
        let coln = data[0].len();
        let mut arr = ndarray::Array2::zeros((rown, coln));
        for r in 0..rown {
            for c in 0..coln {
                arr[[r, c]] = data[r][c];
            }
        }
        let target = ndarray::Array2::<f64>::zeros((rown, coln));
        let dataset = Dataset::new(arr, target);

        let model = KMeans::params(n_clusters).fit(&dataset).unwrap();
        let dataset = model.predict(dataset);
        let DatasetBase { targets, .. } = dataset;
        let ret: Vec<u64> = targets.map(|&x| x as u64).into_iter().collect();
        for ((_, point), label) in self.points.iter_mut().zip(ret.iter()) {
            point.labels.insert("keans_clusters".to_string(), *label);
        }
        return ret;
    }

    pub fn perform_dbscan(&self, eps: f64, min_samples: usize) {
        // Collect coordinates of all points
        let data: Vec<Vec<f64>> = self
            .points
            .iter()
            .map(|(_, p)| p.coordinates.clone())
            .collect();
        let rown = data.len();
        let coln = data[0].len();
        let mut arr = ndarray::Array2::zeros((rown, coln));
        for r in 0..rown {
            for c in 0..coln {
                arr[[r, c]] = data[r][c];
            }
        }
        let target = ndarray::Array2::<f64>::zeros((rown, coln));
        let dataset = Dataset::new(arr, target);
        // Infer an optimal set of centroids based on the training data distribution

        let cluster_memberships = Dbscan::params(min_samples)
            .tolerance(eps)
            .transform(dataset)
            .unwrap();

        // sigle target dataset
        let label_count = cluster_memberships.label_count().remove(0);

        println!();
        println!("Result: ");
        for (label, count) in label_count {
            match label {
                None => println!(" - {} noise points", count),
                Some(i) => println!(" - {} points in cluster {}", count, i),
            }
        }

        let silhouette_score = cluster_memberships.silhouette_score().unwrap();

        println!("Silhouette score: {}", silhouette_score);

        let (records, cluster_memberships) =
            (cluster_memberships.records, cluster_memberships.targets);

        println!("{}", records);
        println!("{:?}", cluster_memberships);
    }

    pub fn from_json(filepath: &str) -> VectorSpace {
        let data = std::fs::read_to_string(filepath).unwrap();
        let data: serde_json::Value = serde_json::from_str(&data).unwrap();

        let points = data.get("points").unwrap();
        let points = points.as_array().unwrap();

        let dimen = data.get("dimensions").unwrap();
        let dimen = dimen.as_array().unwrap();
        let mut dimensions = Vec::new();
        for dim_data in dimen {
            let dim = Dimension::deserialize(dim_data).unwrap();
            dimensions.push(dim);
        }
        let mut vector_space = VectorSpace {
            dimensions,
            points: HashMap::new(),
        };

        for point_data in points {
            let p = Point::deserialize(point_data).unwrap();
            vector_space.add_point(p).unwrap();
        }

        vector_space
    }

    pub fn to_json(&self, filepath: &str) {
        let point: Vec<&Point> = self.points.values().collect();
        let data = json!({
            "dimensions":self.dimensions,
            "points":point
        });
        let data = data.to_string();
        std::fs::write(filepath, data).unwrap();
    }
}
