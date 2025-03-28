use std::{cell::RefCell, rc::Rc};

use gpu_math::math::matrix::Matrix;
use wgpu::{Device, Queue};

use super::MatrixRef;

#[derive(Debug)]
pub struct Input {
    inputs_matrix: MatrixRef,
}

impl Input {
    pub fn new(input_matrix: Matrix, device: Rc<Device>, queue: Rc<Queue>) -> Self {
        Self {
            inputs_matrix: { Rc::new(RefCell::new(input_matrix.buf(device, queue))) },
        }
    }

    pub fn get_inputs(&self) -> MatrixRef {
        self.inputs_matrix.clone()
    }
}
