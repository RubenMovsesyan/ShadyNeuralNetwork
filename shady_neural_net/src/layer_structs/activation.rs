use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    Step,
    Threshold(ThresholdFunction),
    BinarySigmoid(BinarySigmoidFunction),
    BipolarSigmoid(BipolarSigmoidFunction),
    ReLU,
    LeakyReLU(LeakyReLUFunction),
    HyperbolicTangent,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Copy, Clone, Serialize, Deserialize)]
pub struct ThresholdFunction {
    pub threshold_value: f32,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Copy, Clone, Serialize, Deserialize)]
pub struct BinarySigmoidFunction {
    pub k: f32,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Copy, Clone, Serialize, Deserialize)]
pub struct BipolarSigmoidFunction {
    pub k: f32,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Copy, Clone, Serialize, Deserialize)]
pub struct LeakyReLUFunction {
    pub a: f32,
}
