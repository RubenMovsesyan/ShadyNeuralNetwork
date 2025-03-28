use std::error::Error;
use std::rc::Rc;

use gpu_math::math::matrix::Matrix;
use wgpu::{Device, Queue, include_wgsl};

use super::loss_function::LossFunction;

#[derive(Debug)]
pub struct Output {
    expected_matrix: Matrix,
    loss_function_gradient: Matrix,
    loss_function: LossFunction,
    loss_function_index: usize,
    loss_function_gradient_index: usize,
}

impl Output {
    pub fn new(
        mut expected_matrix: Matrix,
        loss_function: LossFunction,
        device: Rc<Device>,
        queue: Rc<Queue>,
    ) -> Self {
        expected_matrix = expected_matrix.buf(device.clone(), queue.clone());
        let loss_function_gradient =
            Matrix::with_shape((expected_matrix.rows(), expected_matrix.cols())).buf(device, queue);

        let loss_function_index = match loss_function {
            LossFunction::LogLoss => expected_matrix
                .add_custom_multi_op_pipeline(include_wgsl!(
                    "../shaders/loss_functions/log_loss.wgsl"
                ))
                .expect("Failed to set loss function on expected"),
        };

        let loss_function_gradient_index = match loss_function {
            LossFunction::LogLoss => expected_matrix
                .add_custom_multi_op_pipeline(include_wgsl!(
                    "../shaders/loss_function_gradients/d_log_loss.wgsl"
                ))
                .expect("Failed to set the loss function gradient on expected"),
        };

        Self {
            expected_matrix,
            loss_function_gradient,
            loss_function,
            loss_function_index,
            loss_function_gradient_index,
        }
    }

    pub fn get_loss_gradient(&mut self, predicted: &Matrix) -> Result<&Matrix, Box<dyn Error>> {
        Matrix::run_custom_multi_op_pipeline_into(
            &self.expected_matrix,
            predicted,
            self.loss_function_gradient_index,
            &mut self.loss_function_gradient,
        )?;

        Ok(&self.loss_function_gradient)
    }

    pub fn get_cost(&self, predicted: &Matrix) -> Result<f32, Box<dyn Error>> {
        let cost = self
            .expected_matrix
            .run_custom_multi_op_pipeline(predicted, self.loss_function_index)?;

        Ok(cost.sum()? / (cost.rows() * cost.cols()) as f32)
    }
}
