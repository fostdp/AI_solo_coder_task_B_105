use serde::{Deserialize, Serialize};
use std::sync::Arc;
use ndarray::{Array2, Array1, ArrayView1};
use linfa::prelude::*;
use linfa_kernel::{Kernel, KernelMethod, KernelInner};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPRConfig {
    pub noise_var: f64,
    pub length_scale: f64,
    pub signal_var: f64,
    pub n_restarts: usize,
    pub normalize_y: bool,
    pub use_rf: bool,
    pub rf_estimators: usize,
    pub termination_threshold: f64,
    pub min_training_points: usize,
    pub convergence_window: usize,
    pub max_patience: usize,
    pub uncertainty_threshold: f64,
}

impl Default for GPRConfig {
    fn default() -> Self {
        Self {
            noise_var: 1e-5,
            length_scale: 1.0,
            signal_var: 1.0,
            n_restarts: 5,
            normalize_y: true,
            use_rf: true,
            rf_estimators: 100,
            termination_threshold: 0.01,
            min_training_points: 5,
            convergence_window: 3,
            max_patience: 10,
            uncertainty_threshold: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictionResult {
    pub prediction: f64,
    pub std_dev: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence: f64,
    pub should_terminate: bool,
    pub termination_reason: Option<String>,
    pub predicted_remaining_iterations: Option<usize>,
    pub model_type: String,
    pub feature_importance: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConvergenceStatus {
    pub is_converged: bool,
    pub current_value: f64,
    pub moving_average: f64,
    pub slope: f64,
    pub relative_change: f64,
    pub consecutive_improvements: usize,
    pub consecutive_degradations: usize,
    pub confidence: f64,
}

#[allow(dead_code)]
struct GpModel {
    l: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    x_mean: Vec<f64>,
    x_std: Vec<f64>,
    y_mean: f64,
    y_std: f64,
    kernel_method: KernelMethod<f64>,
    n_features: usize,
}

impl GpModel {
    fn kernel_value(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let a = ArrayView1::from(x1);
        let b = ArrayView1::from(x2);
        self.kernel_method.distance(a, b)
    }

    fn predict_single(&self, x: &[f64], signal_var: f64, noise_var: f64) -> (f64, f64) {
        let n = self.x_train.len();
        let mut k_star = vec![0.0; n];
        for i in 0..n {
            k_star[i] = signal_var * self.kernel_value(x, &self.x_train[i]);
        }

        let mut v = vec![0.0; n];
        for i in 0..n {
            let mut sum = k_star[i];
            for j in 0..i {
                sum -= self.l[i][j] * v[j];
            }
            v[i] = sum / self.l[i][i];
        }

        let mut mean = 0.0;
        for i in 0..n {
            mean += k_star[i] * self.alpha[i];
        }

        let mut var = signal_var * self.kernel_value(x, x) + noise_var;
        for i in 0..n {
            var -= v[i] * v[i];
        }
        var = var.max(1e-15);

        (mean * self.y_std + self.y_mean, var.sqrt() * self.y_std)
    }
}

#[allow(dead_code)]
struct RfEnsemble {
    trees: Vec<RfTree>,
    n_features: usize,
    feature_importance: Vec<f64>,
}

struct RfTree {
    feature_idx: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl RfEnsemble {
    fn fit(x: &[Vec<f64>], y: &[f64], n_estimators: usize, n_features: usize) -> Self {
        let n = x.len();
        let mut trees = Vec::with_capacity(n_estimators);
        let mut feature_counts = vec![0usize; n_features];

        let mut rng = rand::thread_rng();

        for _ in 0..n_estimators {
            let feature_idx = rand::Rng::gen_range(&mut rng, 0..n_features);

            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for xi in x {
                let v = xi[feature_idx];
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }

            let threshold = if (max_val - min_val).abs() < 1e-15 {
                min_val
            } else {
                min_val + rand::Rng::gen_range(&mut rng, 0.0..1.0) * (max_val - min_val)
            };

            let mut left_sum = 0.0;
            let mut left_count = 0usize;
            let mut right_sum = 0.0;
            let mut right_count = 0usize;

            for (xi, &yi) in x.iter().zip(y.iter()) {
                if xi[feature_idx] <= threshold {
                    left_sum += yi;
                    left_count += 1;
                } else {
                    right_sum += yi;
                    right_count += 1;
                }
            }

            let left_value = if left_count > 0 { left_sum / left_count as f64 } else { y.iter().sum::<f64>() / n as f64 };
            let right_value = if right_count > 0 { right_sum / right_count as f64 } else { y.iter().sum::<f64>() / n as f64 };

            feature_counts[feature_idx] += 1;

            trees.push(RfTree {
                feature_idx,
                threshold,
                left_value,
                right_value,
            });
        }

        let total_splits: usize = feature_counts.iter().sum();
        let feature_importance = if total_splits > 0 {
            feature_counts.iter().map(|&c| c as f64 / total_splits as f64).collect()
        } else {
            vec![1.0 / n_features as f64; n_features]
        };

        RfEnsemble {
            trees,
            n_features,
            feature_importance,
        }
    }

    fn predict(&self, x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for tree in &self.trees {
            if x[tree.feature_idx] <= tree.threshold {
                sum += tree.left_value;
            } else {
                sum += tree.right_value;
            }
        }
        sum / self.trees.len() as f64
    }
}

#[derive(Clone)]
pub struct GaussianProcessRegressor {
    config: GPRConfig,
    gp_model: Option<Arc<GpModel>>,
    rf_model: Option<Arc<RfEnsemble>>,
    x_history: Vec<Vec<f64>>,
    y_history: Vec<f64>,
    convergence_tracker: Vec<f64>,
    best_value: f64,
    patience_counter: usize,
    n_features: usize,
    model_trained: bool,
}

impl GaussianProcessRegressor {
    pub fn new(config: GPRConfig, n_features: usize) -> Self {
        let n_features = n_features.max(1);
        assert!(n_features >= 1, "Number of features must be >= 1");

        Self {
            config,
            gp_model: None,
            rf_model: None,
            x_history: Vec::new(),
            y_history: Vec::new(),
            convergence_tracker: Vec::new(),
            best_value: f64::INFINITY,
            patience_counter: 0,
            n_features,
            model_trained: false,
        }
    }

    pub fn with_initial_data(config: GPRConfig, x: &[Vec<f64>], y: &[f64]) -> Self {
        assert_eq!(x.len(), y.len(), "X and y must have same length");
        let n_features = x.get(0).map(|v| v.len()).unwrap_or(1);
        let mut gpr = Self::new(config, n_features);
        for (xi, yi) in x.iter().zip(y.iter()) {
            gpr.add_training_point(xi, *yi);
        }
        if !x.is_empty() {
            gpr.fit().expect("Initial training failed");
        }
        gpr
    }

    pub fn add_training_point(&mut self, x: &[f64], y: f64) {
        assert_eq!(x.len(), self.n_features, "Feature dimension mismatch");
        let x_vec = x.to_vec();
        self.x_history.push(x_vec);
        self.y_history.push(y);
        self.convergence_tracker.push(y);
        self.model_trained = false;

        if y < self.best_value {
            self.best_value = y;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
        }
    }

    pub fn fit(&mut self) -> Result<(), String> {
        let n = self.x_history.len();
        if n < self.config.min_training_points {
            return Err(format!(
                "Need at least {} training points, have {}",
                self.config.min_training_points, n
            ));
        }

        let y_slice = &self.y_history;
        let (y_mean, y_std) = if self.config.normalize_y {
            let mean = y_slice.iter().sum::<f64>() / n as f64;
            let var = y_slice.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            let std = var.sqrt().max(1e-10);
            (mean, std)
        } else {
            (0.0, 1.0)
        };

        let mut x_mean = vec![0.0; self.n_features];
        let mut x_std = vec![1.0; self.n_features];

        for j in 0..self.n_features {
            let sum: f64 = self.x_history.iter().map(|v| v[j]).sum();
            x_mean[j] = sum / n as f64;
            let var: f64 = self.x_history.iter().map(|v| (v[j] - x_mean[j]).powi(2)).sum::<f64>() / n as f64;
            x_std[j] = var.sqrt().max(1e-10);
        }

        let x_norm: Vec<Vec<f64>> = self.x_history.iter()
            .map(|v| v.iter().enumerate().map(|(j, &val)| (val - x_mean[j]) / x_std[j]).collect())
            .collect();
        let y_norm: Vec<f64> = self.y_history.iter()
            .map(|&v| (v - y_mean) / y_std)
            .collect();

        let x_arr = Array2::from_shape_fn((n, self.n_features), |(i, j)| x_norm[i][j]);
        let y_arr = Array1::from_vec(y_norm.clone());

        let _dataset = DatasetBase::from((x_arr.view(), y_arr.view()));

        let eps = 2.0 * self.config.length_scale * self.config.length_scale;
        let kernel_method = KernelMethod::Gaussian(eps);

        let kernel = Kernel::params()
            .method(kernel_method.clone())
            .transform(x_arr.view());

        let kernel_matrix = match &kernel.inner {
            KernelInner::Dense(m) => m,
            _ => panic!("Expected dense kernel"),
        };

        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.config.signal_var * kernel_matrix[[i, j]];
                if i == j {
                    k[i][j] += self.config.noise_var;
                }
            }
        }

        let l = match cholesky(&k) {
            Some(l) => l,
            None => return Err("Cholesky decomposition failed".to_string()),
        };

        let alpha = solve_cholesky(&l, &y_norm);

        if self.config.use_rf {
            let rf = RfEnsemble::fit(&x_norm, &y_norm, self.config.rf_estimators, self.n_features);
            self.rf_model = Some(Arc::new(rf));
        }

        self.gp_model = Some(Arc::new(GpModel {
            l,
            alpha,
            x_train: x_norm,
            y_train: y_norm,
            y_mean,
            y_std,
            x_mean,
            x_std,
            kernel_method,
            n_features: self.n_features,
        }));

        self.model_trained = true;
        Ok(())
    }

    pub fn predict(&self, x: &[f64]) -> Result<PredictionResult, String> {
        assert_eq!(x.len(), self.n_features, "Feature dimension mismatch");

        if !self.model_trained {
            return Err("Model not trained yet. Call fit() first.".to_string());
        }

        let model = self.gp_model.as_ref().ok_or("GP model not available")?;
        let x_norm: Vec<f64> = x.iter().enumerate()
            .map(|(j, &val)| (val - model.x_mean[j]) / model.x_std[j])
            .collect();

        let (gp_pred, gp_std) = model.predict_single(
            &x_norm,
            self.config.signal_var,
            self.config.noise_var,
        );

        let prediction = gp_pred;
        let std_dev = gp_std;

        let z_score = 1.96;
        let _lower_bound = prediction - z_score * std_dev;
        let _upper_bound = prediction + z_score * std_dev;
        let confidence = (1.0 - (std_dev / (prediction.abs() + 1e-10)).min(1.0f64)).max(0.0f64);

        let rf_prediction = if let Some(rf) = &self.rf_model {
            Some(rf.predict(&x_norm))
        } else {
            None
        };

        let importances = self.rf_model.as_ref().map(|rf| rf.feature_importance.clone());

        let final_prediction = match rf_prediction {
            Some(rf_pred) => 0.7 * prediction + 0.3 * rf_pred,
            None => prediction,
        };

        let (should_terminate, termination_reason, predicted_remaining) =
            self.check_termination_criteria(final_prediction, std_dev);

        let model_type = if self.config.use_rf && self.rf_model.is_some() {
            "GP+RF_ensemble".to_string()
        } else {
            "GP".to_string()
        };

        Ok(PredictionResult {
            prediction: final_prediction,
            std_dev,
            lower_bound: final_prediction - z_score * std_dev,
            upper_bound: final_prediction + z_score * std_dev,
            confidence,
            should_terminate,
            termination_reason,
            predicted_remaining_iterations: predicted_remaining,
            model_type,
            feature_importance: importances,
        })
    }

    pub fn predict_batch(&self, x_batch: &[Vec<f64>]) -> Result<Vec<PredictionResult>, String> {
        let mut results = Vec::with_capacity(x_batch.len());
        for x in x_batch {
            results.push(self.predict(x)?);
        }
        Ok(results)
    }

    fn check_termination_criteria(&self, prediction: f64, std_dev: f64) -> (bool, Option<String>, Option<usize>) {
        let n = self.convergence_tracker.len();

        if n < self.config.min_training_points {
            return (false, None, Some(self.config.min_training_points - n));
        }

        let window = self.config.convergence_window.min(n);
        let recent: &[f64] = &self.convergence_tracker[n - window..];

        let moving_avg = recent.iter().sum::<f64>() / window as f64;

        let mut max_abs_change = 0.0;
        for w in recent.windows(2) {
            let abs_change = (w[1] - w[0]).abs();
            if abs_change > max_abs_change {
                max_abs_change = abs_change;
            }
        }

        let relative_change = max_abs_change / (moving_avg.abs() + 1e-10);

        let _consecutive_improvements = self.count_consecutive_improvements();
        let consecutive_degradations = self.count_consecutive_degradations();

        let mut should_terminate = false;
        let mut reason = String::new();
        let mut predicted_remaining: Option<usize> = None;

        if std_dev < self.config.uncertainty_threshold && relative_change < self.config.termination_threshold {
            should_terminate = true;
            reason = "Converged: low uncertainty and relative change below threshold".to_string();
            predicted_remaining = Some(0);
        }

        if self.patience_counter >= self.config.max_patience {
            should_terminate = true;
            reason = format!("Early stopping: no improvement for {} iterations", self.patience_counter);
            predicted_remaining = Some(0);
        }

        if consecutive_degradations >= 3 {
            let slope = self.calculate_slope(recent);
            if slope > 0.0 && relative_change > 0.05 {
                should_terminate = true;
                reason = "Diverging: consistent increase in objective".to_string();
                predicted_remaining = Some(0);
            }
        }

        if !should_terminate {
            let current_val = *self.convergence_tracker.last().unwrap_or(&0.0);
            if current_val <= self.config.termination_threshold {
                should_terminate = true;
                reason = "Objective reached below termination threshold".to_string();
                predicted_remaining = Some(0);
            }
        }

        if !should_terminate {
            let slope = self.calculate_slope(&self.convergence_tracker);
            if slope.abs() > 0.0 {
                let remaining_est = ((self.config.termination_threshold - prediction) / slope).abs().ceil() as usize;
                predicted_remaining = Some(remaining_est.min(1000));
            }
        }

        let reason_opt = if should_terminate { Some(reason) } else { None };
        (should_terminate, reason_opt, predicted_remaining)
    }

    fn count_consecutive_improvements(&self) -> usize {
        let n = self.convergence_tracker.len();
        if n < 2 {
            return 0;
        }
        let mut count = 0;
        for i in (1..n).rev() {
            if self.convergence_tracker[i] < self.convergence_tracker[i - 1] {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    fn count_consecutive_degradations(&self) -> usize {
        let n = self.convergence_tracker.len();
        if n < 2 {
            return 0;
        }
        let mut count = 0;
        for i in (1..n).rev() {
            if self.convergence_tracker[i] > self.convergence_tracker[i - 1] {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    fn calculate_slope(&self, values: &[f64]) -> f64 {
        let n = values.len();
        if n < 2 {
            return 0.0;
        }

        let indices: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mean_x = indices.iter().sum::<f64>() / n as f64;
        let mean_y = values.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (&x, &y) in indices.iter().zip(values.iter()) {
            numerator += (x - mean_x) * (y - mean_y);
            denominator += (x - mean_x).powi(2);
        }

        if denominator.abs() > 1e-15 {
            numerator / denominator
        } else {
            0.0
        }
    }

    pub fn get_convergence_status(&self) -> ConvergenceStatus {
        let n = self.convergence_tracker.len();
        if n == 0 {
            return ConvergenceStatus {
                is_converged: false,
                current_value: 0.0,
                moving_average: 0.0,
                slope: 0.0,
                relative_change: 0.0,
                consecutive_improvements: 0,
                consecutive_degradations: 0,
                confidence: 0.0,
            };
        }

        let window = self.config.convergence_window.min(n);
        let recent: &[f64] = &self.convergence_tracker[n - window..];
        let moving_avg = recent.iter().sum::<f64>() / window as f64;

        let mut max_abs_change = 0.0;
        for w in recent.windows(2) {
            let abs_change = (w[1] - w[0]).abs();
            if abs_change > max_abs_change {
                max_abs_change = abs_change;
            }
        }

        let relative_change = max_abs_change / (moving_avg.abs() + 1e-10);
        let current_value = *self.convergence_tracker.last().unwrap();
        let slope = self.calculate_slope(&self.convergence_tracker);

        let mut confidence = 1.0;
        if n < self.config.min_training_points {
            confidence = n as f64 / self.config.min_training_points as f64;
        }

        let is_converged = relative_change < self.config.termination_threshold
            && n >= self.config.min_training_points;

        ConvergenceStatus {
            is_converged,
            current_value,
            moving_average: moving_avg,
            slope,
            relative_change,
            consecutive_improvements: self.count_consecutive_improvements(),
            consecutive_degradations: self.count_consecutive_degradations(),
            confidence,
        }
    }

    pub fn optimize_hyperparameters(&mut self) -> Result<(), String> {
        if self.x_history.len() < 4 {
            return Err("Need at least 4 points for hyperparameter optimization".to_string());
        }

        let mut best_ll = f64::NEG_INFINITY;
        let mut best_params = (self.config.length_scale, self.config.signal_var, self.config.noise_var);

        let scales = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0];
        let signals = [0.1, 0.5, 1.0, 2.0];
        let noises = [1e-5, 1e-4, 1e-3];

        for &ls in &scales {
            for &sv in &signals {
                for &nv in &noises {
                    let ll = self.evaluate_log_likelihood(ls, sv, nv);
                    if ll > best_ll {
                        best_ll = ll;
                        best_params = (ls, sv, nv);
                    }
                }
            }
        }

        self.config.length_scale = best_params.0;
        self.config.signal_var = best_params.1;
        self.config.noise_var = best_params.2;

        self.fit()
    }

    fn evaluate_log_likelihood(&self, length_scale: f64, signal_var: f64, noise_var: f64) -> f64 {
        let n = self.x_history.len();
        if n < 2 {
            return f64::NEG_INFINITY;
        }

        let mut dist_sq = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..self.n_features {
                    let diff = self.x_history[i][k] - self.x_history[j][k];
                    sum += diff * diff;
                }
                dist_sq[i][j] = sum / (2.0 * length_scale * length_scale);
            }
        }

        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    k[i][j] = signal_var * (-dist_sq[i][j]).exp() + noise_var;
                } else {
                    k[i][j] = signal_var * (-dist_sq[i][j]).exp();
                }
            }
        }

        let y_mean: f64 = self.y_history.iter().sum::<f64>() / n as f64;
        let y_centered: Vec<f64> = self.y_history.iter().map(|&v| v - y_mean).collect();

        let l = match cholesky(&k) {
            Some(l) => l,
            None => return f64::NEG_INFINITY,
        };
        let alpha = solve_cholesky(&l, &y_centered);

        let mut log_det_k = 0.0;
        for i in 0..n {
            log_det_k += 2.0 * l[i][i].ln();
        }

        let mut ll = -0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * log_det_k;
        let mut sum_alpha = 0.0;
        for &a in &alpha {
            sum_alpha += a * a;
        }
        ll -= 0.5 * sum_alpha;

        ll
    }

    pub fn is_trained(&self) -> bool {
        self.model_trained
    }

    pub fn training_points(&self) -> usize {
        self.x_history.len()
    }

    pub fn best_value(&self) -> f64 {
        self.best_value
    }
}

fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

fn solve_cholesky(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = b.to_vec();
    for i in 0..n {
        for j in 0..i {
            y[i] -= l[i][j] * y[j];
        }
        y[i] /= l[i][i];
    }
    let mut x = y;
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= l[j][i] * x[j];
        }
        x[i] /= l[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpr_basic() {
        let config = GPRConfig::default();
        let gpr = GaussianProcessRegressor::new(config, 1);
        assert!(!gpr.is_trained());
        assert_eq!(gpr.training_points(), 0);
    }

    #[test]
    fn test_training_and_prediction() {
        let config = GPRConfig {
            min_training_points: 3,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 1);

        for i in 0..5 {
            let x = vec![i as f64 * 0.5];
            let y = 2.0 * x[0] + 1.0;
            gpr.add_training_point(&x, y);
        }

        gpr.fit().expect("fit should succeed");
        assert!(gpr.is_trained());

        let pred = gpr.predict(&[1.5]).expect("prediction should succeed");
        assert!(pred.prediction > 0.0);
        assert!(pred.std_dev >= 0.0);
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
    }

    #[test]
    fn test_termination_criteria() {
        let config = GPRConfig {
            min_training_points: 3,
            termination_threshold: 0.01,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 1);

        for i in 0..10 {
            let x = vec![i as f64];
            let y = 1.0 / (i as f64 + 1.0);
            gpr.add_training_point(&x, y);
        }

        gpr.fit().expect("fit should succeed");
        let pred = gpr.predict(&[20.0]).expect("prediction should succeed");
        assert!(pred.prediction >= 0.0);
    }

    #[test]
    fn test_convergence_status() {
        let config = GPRConfig {
            min_training_points: 3,
            convergence_window: 3,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 1);

        let status = gpr.get_convergence_status();
        assert!(!status.is_converged);

        for i in 0..6 {
            let x = vec![i as f64];
            let y = 1.0 / (i as f64 + 1.0);
            gpr.add_training_point(&x, y);
        }

        let status = gpr.get_convergence_status();
        assert!(status.consecutive_improvements >= 3);
        assert!(status.relative_change > 0.0);
    }

    #[test]
    fn test_min_training_points_boundary() {
        let config = GPRConfig {
            min_training_points: 5,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 1);

        for i in 0..4 {
            let x = vec![i as f64];
            let y = x[0];
            gpr.add_training_point(&x, y);
        }

        let result = gpr.fit();
        assert!(result.is_err(), "Should fail with < 5 points");

        gpr.add_training_point(&[4.0], 4.0);
        let result = gpr.fit();
        assert!(result.is_ok(), "Should succeed with >= 5 points");
    }

    #[test]
    fn test_patience_early_stopping() {
        let config = GPRConfig {
            min_training_points: 3,
            max_patience: 3,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 1);

        gpr.add_training_point(&[0.0], 1.0);
        gpr.add_training_point(&[1.0], 0.5);
        gpr.add_training_point(&[2.0], 0.8);
        gpr.add_training_point(&[3.0], 0.9);
        gpr.add_training_point(&[4.0], 0.95);

        assert_eq!(gpr.patience_counter, 3);

        gpr.fit().expect("fit should succeed");
        let pred = gpr.predict(&[5.0]).expect("pred should succeed");
        assert!(pred.should_terminate, "Should terminate due to patience");
    }

    #[test]
    fn test_feature_importance() {
        let config = GPRConfig {
            min_training_points: 5,
            use_rf: true,
            rf_estimators: 50,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 3);

        for i in 0..10 {
            let x = vec![i as f64 * 0.1, 1.0 - i as f64 * 0.1, 0.5];
            let y = x[0] * 2.0 + x[1] * 0.5 + x[2] * 0.1;
            gpr.add_training_point(&x, y);
        }

        gpr.fit().expect("fit should succeed");
        let pred = gpr.predict(&[0.5, 0.5, 0.5]).expect("pred should succeed");

        assert!(pred.feature_importance.is_some());
        let imp = pred.feature_importance.unwrap();
        assert_eq!(imp.len(), 3);
        assert!(imp[0] > 0.0);
    }

    #[test]
    fn test_predict_batch() {
        let config = GPRConfig {
            min_training_points: 3,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 1);

        for i in 0..5 {
            let x = vec![i as f64];
            let y = x[0] * x[0];
            gpr.add_training_point(&x, y);
        }

        gpr.fit().expect("fit should succeed");

        let batch = vec![vec![1.0], vec![2.0], vec![3.0]];
        let results = gpr.predict_batch(&batch).expect("batch predict should succeed");
        assert_eq!(results.len(), 3);

        for r in results {
            assert!(r.prediction >= 0.0);
        }
    }

    #[test]
    fn test_slope_calculation() {
        let config = GPRConfig::default();
        let mut gpr = GaussianProcessRegressor::new(config, 1);

        gpr.add_training_point(&[0.0], 10.0);
        gpr.add_training_point(&[1.0], 8.0);
        gpr.add_training_point(&[2.0], 6.0);
        gpr.add_training_point(&[3.0], 4.0);

        let slope = gpr.calculate_slope(&[10.0, 8.0, 6.0, 4.0]);
        assert!((slope - -2.0).abs() < 1e-9, "Slope should be -2.0, got {:.4}", slope);
    }

    #[test]
    fn test_log_likelihood_optimization() {
        let config = GPRConfig {
            min_training_points: 4,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 1);

        for i in 0..8 {
            let x = vec![i as f64 * 0.5];
            let y = (-x[0]).exp() * 10.0;
            gpr.add_training_point(&x, y);
        }

        let old_ls = gpr.config.length_scale;
        gpr.optimize_hyperparameters().expect("opt should succeed");
        assert_ne!(old_ls, gpr.config.length_scale, "Length scale should change after optimization");
        assert!(gpr.is_trained());
    }

    #[test]
    fn test_empty_history_anomaly() {
        let config = GPRConfig::default();
        let gpr = GaussianProcessRegressor::new(config, 1);

        let pred = gpr.predict(&[0.0]);
        assert!(pred.is_err(), "Prediction without training should fail");

        let status = gpr.get_convergence_status();
        assert_eq!(status.current_value, 0.0);
        assert_eq!(status.slope, 0.0);
    }

    #[test]
    fn test_multidimensional_features() {
        let config = GPRConfig {
            min_training_points: 4,
            ..Default::default()
        };
        let mut gpr = GaussianProcessRegressor::new(config, 4);

        for i in 0..8 {
            let t = i as f64 * 0.3;
            let x = vec![t, t.sin(), t.cos(), t * t];
            let y = x[0] + x[1] * 2.0 + x[2] * 0.5;
            gpr.add_training_point(&x, y);
        }

        gpr.fit().expect("fit should succeed");

        let pred = gpr.predict(&[1.5, 1.5_f64.sin(), 1.5_f64.cos(), 2.25]).expect("pred should succeed");
        assert!(pred.prediction > 0.0);
        assert!(pred.confidence > 0.0);
    }
}
