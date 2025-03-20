use super::bias::Bias;

pub enum WeightDistribution {
    Xavier,
}

impl WeightDistribution {
    pub fn get_weight_distribution(&self, num_inputs: u64, num_outputs: u64) -> Vec<f32> {
        use WeightDistribution::*;
        match self {
            Xavier => {
                let (lower_bound, upper_bound) = (
                    -(1.0 / f32::sqrt(num_inputs as f32)),
                    (1.0 / f32::sqrt(num_inputs as f32)),
                );

                // Compute the random numbers in the xavier distribution to use as weights
                (0..(num_inputs * num_outputs))
                    .into_iter()
                    .map(|_| {
                        lower_bound + rand::random_range(0.0..=1.0) * (upper_bound - lower_bound)
                    })
                    .collect()
            }
        }
    }

    pub fn get_bias_distribution(&self, num_nodes: u64) -> Vec<Bias> {
        use WeightDistribution::*;
        match self {
            Xavier => {
                let (lower_bound, upper_bound) = (
                    -(1.0 / f32::sqrt(num_nodes as f32)),
                    (1.0 / f32::sqrt(num_nodes as f32)),
                );

                // Compute the random numbers in the xavier distribution to use as weights
                (0..num_nodes)
                    .into_iter()
                    .map(|_| Bias {
                        bias: lower_bound
                            + rand::random_range(0.0..=1.0) * (upper_bound - lower_bound),
                    })
                    .collect()
            }
        }
    }
}
