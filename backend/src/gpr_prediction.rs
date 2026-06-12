use serde::{Deserialize, Serialize};
use gp_termination as gp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPRConfig {
    pub length_scale: f64,
    pub signal_variance: f64,
    pub noise_variance: f64,
    pub kernel_type: String,
    pub target_moisture: f64,
    pub confidence_level: f64,
    pub max_prediction_hours: f64,
}

impl Default for GPRConfig {
    fn default() -> Self {
        Self {
            length_scale: 100.0,
            signal_variance: 25.0,
            noise_variance: 0.5,
            kernel_type: "matern52".to_string(),
            target_moisture: 15.0,
            confidence_level: 0.95,
            max_prediction_hours: 5000.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GPRPredictionResult {
    pub predicted_end_time_hours: f64,
    pub confidence_lower_hours: f64,
    pub confidence_upper_hours: f64,
    pub remaining_hours: f64,
    pub confidence_interval_lower: f64,
    pub confidence_interval_upper: f64,
    pub predicted_curve_time: Vec<f64>,
    pub predicted_curve_mean: Vec<f64>,
    pub predicted_curve_lower: Vec<f64>,
    pub predicted_curve_upper: Vec<f64>,
    pub training_data_time: Vec<f64>,
    pub training_data_moisture: Vec<f64>,
    pub r_squared: f64,
    pub log_marginal_likelihood: f64,
    pub uncertainty_at_target: f64,
}

#[derive(Clone)]
pub struct GaussianProcessRegressor {
    config: GPRConfig,
    train_x: Vec<f64>,
    train_y: Vec<f64>,
    internal: Option<gp::GaussianProcessRegressor>,
    is_trained: bool,
    log_marginal_likelihood: f64,
    r_squared: f64,
}

impl GaussianProcessRegressor {
    pub fn new(config: GPRConfig) -> Self {
        Self {
            config,
            train_x: Vec::new(),
            train_y: Vec::new(),
            internal: None,
            is_trained: false,
            log_marginal_likelihood: 0.0,
            r_squared: 0.0,
        }
    }

    pub fn kernel(&self, x1: f64, x2: f64) -> f64 {
        let ls = self.config.length_scale;
        let sv = self.config.signal_variance;
        match self.config.kernel_type.as_str() {
            "rbf" => {
                let dist = (x1 - x2).powi(2);
                sv * (-dist / (2.0 * ls.powi(2))).exp()
            }
            "matern32" => {
                let dist = (x1 - x2).abs();
                let sqrt3 = 3.0_f64.sqrt();
                let arg = sqrt3 * dist / ls;
                sv * (1.0 + arg) * (-arg).exp()
            }
            "matern52" => {
                let dist = (x1 - x2).abs();
                let sqrt5 = 5.0_f64.sqrt();
                let arg = sqrt5 * dist / ls;
                sv * (1.0 + arg + arg * arg / 3.0) * (-arg).exp()
            }
            _ => {
                let dist = (x1 - x2).powi(2);
                sv * (-dist / (2.0 * ls.powi(2))).exp()
            }
        }
    }

    pub fn fit(&mut self, x: &[f64], y: &[f64]) -> Result<(), &'static str> {
        if x.len() != y.len() {
            return Err("Input and output dimensions mismatch");
        }
        if x.len() < 2 {
            return Err("Need at least 2 training points");
        }

        let n = x.len();
        self.train_x = x.to_vec();
        self.train_y = y.to_vec();

        let internal_config = gp::GPRConfig {
            noise_var: self.config.noise_variance,
            length_scale: self.config.length_scale,
            signal_var: self.config.signal_variance,
            n_restarts: 5,
            normalize_y: true,
            use_rf: true,
            rf_estimators: 50,
            termination_threshold: 0.01,
            min_training_points: 2,
            convergence_window: 3,
            max_patience: 50,
            uncertainty_threshold: 0.1,
        };

        let mut internal = gp::GaussianProcessRegressor::new(internal_config, 1);
        for i in 0..n {
            internal.add_training_point(&[x[i]], y[i]);
        }
        internal.fit().map_err(|_| "GPR fitting failed")?;

        self.log_marginal_likelihood = self.evaluate_lml(x, y);
        self.r_squared = self.calculate_r_squared(x, y);
        self.internal = Some(internal);
        self.is_trained = true;

        Ok(())
    }

    pub fn optimize_hyperparameters(&mut self, x: &[f64], y: &[f64]) -> Result<(), &'static str> {
        if x.len() < 4 {
            return Err("Need at least 4 points for hyperparameter optimization");
        }

        let mut best_lml = f64::NEG_INFINITY;
        let mut best_params = (self.config.length_scale, self.config.signal_variance, self.config.noise_variance);

        let scales = [10.0, 50.0, 100.0, 200.0, 500.0];
        let signals = [5.0, 15.0, 25.0, 50.0];
        let noises = [0.1, 0.5, 1.0];

        let orig_ls = self.config.length_scale;
        let orig_sv = self.config.signal_variance;
        let orig_nv = self.config.noise_variance;

        for &ls in &scales {
            for &sv in &signals {
                for &nv in &noises {
                    self.config.length_scale = ls;
                    self.config.signal_variance = sv;
                    self.config.noise_variance = nv;
                    let lml = self.evaluate_lml(x, y);
                    if lml > best_lml {
                        best_lml = lml;
                        best_params = (ls, sv, nv);
                    }
                }
            }
        }

        self.config.length_scale = best_params.0;
        self.config.signal_variance = best_params.1;
        self.config.noise_variance = best_params.2;

        if best_lml <= f64::NEG_INFINITY {
            self.config.length_scale = orig_ls;
            self.config.signal_variance = orig_sv;
            self.config.noise_variance = orig_nv;
        }

        self.fit(x, y)
    }

    fn evaluate_lml(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 2 {
            return f64::NEG_INFINITY;
        }

        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.kernel(x[i], x[j]);
            }
            k[i][i] += self.config.noise_variance;
        }

        let l = match cholesky_decomposition(&k) {
            Some(l) => l,
            None => return f64::NEG_INFINITY,
        };

        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_centered: Vec<f64> = y.iter().map(|&v| v - y_mean).collect();
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

    fn calculate_r_squared(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() < 2 {
            return 0.0;
        }

        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for i in 0..x.len() {
            let yi = y[i];
            let y_pred = self.predict_point(x[i]).unwrap_or(yi);
            ss_tot += (yi - y_mean).powi(2);
            ss_res += (yi - y_pred).powi(2);
        }

        if ss_tot.abs() < 1e-15 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    fn predict_point(&self, x: f64) -> Option<f64> {
        let internal = self.internal.as_ref()?;
        let result = internal.predict(&[x]).ok()?;
        Some(result.prediction)
    }

    pub fn predict_endpoint(&self) -> Result<GPRPredictionResult, &'static str> {
        if !self.is_trained {
            return Err("Model not trained yet");
        }

        let internal = self.internal.as_ref().ok_or("Internal model not available")?;
        let target = self.config.target_moisture;
        let last_time = *self.train_x.last().ok_or("No training data")?;

        let time_step = 1.0;
        let max_time = self.config.max_prediction_hours.max(last_time + 1000.0);
        let mut times = Vec::new();
        let mut means = Vec::new();
        let mut lowers = Vec::new();
        let mut uppers = Vec::new();

        let z = match self.config.confidence_level {
            cl if cl >= 0.99 => 2.576,
            cl if cl >= 0.95 => 1.96,
            cl if cl >= 0.90 => 1.645,
            _ => 1.96,
        };

        let mut t = 0.0;
        let mut end_time: Option<f64> = None;
        let mut uncertainty_at_end = 0.0;

        while t <= max_time {
            times.push(t);
            let result = internal.predict(&[t]).map_err(|_| "Prediction failed")?;

            let mean = result.prediction;
            let std = result.std_dev.max(1e-10);
            let lower = mean - z * std;
            let upper = mean + z * std;

            means.push(mean);
            lowers.push(lower);
            uppers.push(upper);

            if end_time.is_none() && (mean - target).abs() < 0.01 {
                end_time = Some(t);
                uncertainty_at_end = std;
            } else if end_time.is_none() && means.len() >= 2 {
                let prev_mean = means[means.len() - 2];
                if (prev_mean - target) * (mean - target) < 0.0 {
                    let t_prev = times[times.len() - 2];
                    let t_interp = t_prev + (target - prev_mean) * (t - t_prev) / (mean - prev_mean);
                    end_time = Some(t_interp);
                    uncertainty_at_end = std;
                }
            }

            t += time_step;
        }

        if end_time.is_none() && !means.is_empty() {
            let n = means.len();
            if n >= 5 {
                let recent = &means[n - 5..];
                let last = recent[4];
                if last < target {
                    for i in (0..n).rev() {
                        if means[i] >= target {
                            let t1 = times[i];
                            let t2 = times[i + 1];
                            let m1 = means[i];
                            let m2 = means[i + 1];
                            end_time = Some(t1 + (target - m1) * (t2 - t1) / (m2 - m1));
                            uncertainty_at_end = (uppers[i] - lowers[i]) / (2.0 * z);
                            break;
                        }
                    }
                } else {
                    let last_mean = means[n - 1];
                    let slope = if n >= 50 {
                        (last_mean - means[n - 50]) / (times[n - 1] - times[n - 50])
                    } else {
                        (last_mean - means[0]) / (times[n - 1] - times[0])
                    };
                    if slope.abs() > 1e-10 {
                        end_time = Some(times[n - 1] + (target - last_mean) / slope);
                        uncertainty_at_end = (uppers[n - 1] - lowers[n - 1]) / (2.0 * z);
                    }
                }
            }
        }

        let predicted_end_time = end_time.unwrap_or(max_time);
        let remaining = (predicted_end_time - last_time).max(0.0);

        let ci_lower_hours = predicted_end_time - z * uncertainty_at_end;
        let ci_upper_hours = predicted_end_time + z * uncertainty_at_end;

        Ok(GPRPredictionResult {
            predicted_end_time_hours: predicted_end_time,
            confidence_lower_hours: ci_lower_hours,
            confidence_upper_hours: ci_upper_hours,
            remaining_hours: remaining,
            confidence_interval_lower: predicted_end_time - z * uncertainty_at_end,
            confidence_interval_upper: predicted_end_time + z * uncertainty_at_end,
            predicted_curve_time: times,
            predicted_curve_mean: means,
            predicted_curve_lower: lowers,
            predicted_curve_upper: uppers,
            training_data_time: self.train_x.clone(),
            training_data_moisture: self.train_y.clone(),
            r_squared: self.r_squared,
            log_marginal_likelihood: self.log_marginal_likelihood,
            uncertainty_at_target: uncertainty_at_end,
        })
    }
}

fn cholesky_decomposition(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
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