use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Bias {
    pub bias: f32,
}

impl Bias {
    pub fn new(bias: f32) -> Self {
        Self { bias }
    }
}
