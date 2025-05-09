use anyhow::Result;
use gpu_math::{GpuMath, math::matrix::Matrix};
use std::error::Error;

use super::loss_function::LossFunction;

#[derive(Debug)]
pub struct Output {
    expected_matrix: Matrix,
    loss_function_gradient: Matrix,
    cost: Matrix,
    loss_function_index: usize,
    loss_function_gradient_index: usize,
}

impl Output {
    pub fn new(
        expected_matrix: Matrix,
        loss_function: LossFunction,
        gpu_math: &mut GpuMath,
    ) -> Result<Self> {
        let loss_function_gradient =
            // Matrix::with_shape((expected_matrix.rows(), expected_matrix.cols())).buf(device, queue);
            Matrix::new(gpu_math, (expected_matrix.rows, expected_matrix.cols), None)?;

        let loss_function_index = match loss_function {
            LossFunction::LogLoss => gpu_math.create_custom_matrix_matrix_pipeline(include_str!(
                "../shaders/loss_functions/log_loss.wgsl"
            )),
        };

        let loss_function_gradient_index = match loss_function {
            LossFunction::LogLoss => gpu_math.create_custom_matrix_matrix_pipeline(include_str!(
                "../shaders/loss_function_gradients/d_log_loss.wgsl"
            )),
        };

        let cost = Matrix::new(gpu_math, (expected_matrix.rows, expected_matrix.cols), None)?;

        Ok(Self {
            expected_matrix,
            loss_function_gradient,
            cost,
            loss_function_index,
            loss_function_gradient_index,
        })
    }

    pub fn get_loss_gradient(&mut self, predicted: &Matrix) -> Result<&Matrix, Box<dyn Error>> {
        // Matrix::run_custom_multi_op_pipeline_into(
        //     &self.expected_matrix,
        //     predicted,
        //     self.loss_function_gradient_index,
        //     &mut self.loss_function_gradient,
        // )?;
        Matrix::run_custom_matrix_matrix(
            &self.expected_matrix,
            predicted,
            &self.loss_function_gradient,
            self.loss_function_gradient_index,
        )?;

        Ok(&self.loss_function_gradient)
    }

    pub fn get_cost(&self, predicted: &Matrix) -> Result<f32, Box<dyn Error>> {
        // let cost = self
        //     .expected_matrix
        //     .run_custom_multi_op_pipeline(predicted, self.loss_function_index)?;
        Matrix::run_custom_matrix_matrix(
            &self.expected_matrix,
            predicted,
            &self.cost,
            self.loss_function_index,
        )?;

        Ok(Matrix::sum(&self.cost)? / (self.cost.rows * self.cost.cols) as f32)
    }
}
