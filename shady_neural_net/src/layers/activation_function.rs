use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    Step,
    Threshold(f32),
    BinarySigmoid(f32),
    BipolarSigmoid(f32),
    ReLU,
    LeakyReLU(f32),
    HyperbolicTangent,
    Softmax,
    Custom,
}

impl ActivationFunction {
    pub fn get_scalar(&self) -> f32 {
        match self {
            Self::Threshold(val)
            | Self::BinarySigmoid(val)
            | Self::BipolarSigmoid(val)
            | Self::LeakyReLU(val) => *val,
            _ => 0.0,
        }
    }
}
