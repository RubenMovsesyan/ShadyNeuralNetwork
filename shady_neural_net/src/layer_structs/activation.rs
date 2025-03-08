use bytemuck::{Pod, Zeroable};

#[derive(Debug)]
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
#[derive(Debug, Pod, Zeroable, Copy, Clone)]
pub struct ThresholdFunction {
    pub threshold_value: f32,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Copy, Clone)]
pub struct BinarySigmoidFunction {
    pub k: f32,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Copy, Clone)]
pub struct BipolarSigmoidFunction {
    pub k: f32,
}

#[repr(C)]
#[derive(Debug, Pod, Zeroable, Copy, Clone)]
pub struct LeakyReLUFunction {
    pub a: f32,
}
