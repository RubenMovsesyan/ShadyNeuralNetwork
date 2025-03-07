use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
pub struct Bias {
    bias: f32,
    weight: f32,
}

impl Bias {
    pub fn new(bias: f32, weight: f32) -> Self {
        Self { bias, weight }
    }
}
