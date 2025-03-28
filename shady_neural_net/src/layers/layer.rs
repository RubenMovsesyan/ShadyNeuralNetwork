use std::{cell::RefCell, error::Error, rc::Rc};

use gpu_math::math::matrix::Matrix;
use wgpu::{Device, Queue, include_wgsl};

use super::{MatrixRef, activation_function::ActivationFunction};

#[derive(Debug)]
pub struct Layer {
    weights: Matrix,
    biases: Matrix,
    inputs: MatrixRef,
    outputs: MatrixRef,
    num_nodes: usize,
    batch_size: usize,
    activation_function: ActivationFunction,
    activation_function_index: usize,
    activation_function_gradient_index: usize,

    // Back Propogation Matrices
    output_gradient: Matrix,
    inner_gradient: Matrix,
    weights_gradient: Matrix,
    bias_gradient: f32,
    back_prop_gradient: Matrix,
}

impl Layer {
    pub fn new(
        num_inputs: usize,
        num_nodes: usize,
        linked_inputs: MatrixRef,
        batch_size: usize,
        activation_function: ActivationFunction,
        device: Rc<Device>,
        queue: Rc<Queue>,
    ) -> Self {
        // Random distribution for now, add in the ability for custom distributions later
        let mut weights = Matrix::with_shape((num_nodes, num_inputs));
        let mut biases = Matrix::with_shape((num_nodes, 1));
        let mut outputs = Matrix::with_shape((num_nodes, batch_size));
        let mut output_gradient = Matrix::with_shape((num_nodes, batch_size));
        let mut inner_gradient = Matrix::with_shape((num_nodes, batch_size));
        let mut weights_gradient = Matrix::with_shape((num_nodes, num_inputs));
        let mut back_prop_gradient = Matrix::with_shape((num_inputs, batch_size));

        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                weights[(i, j)] = rand::random_range(-0.5..=0.5);
            }
        }

        for i in 0..biases.rows() {
            biases[(i, 0)] = rand::random_range(-0.5..=0.5);
        }

        weights = weights.buf(device.clone(), queue.clone());
        biases = biases.buf(device.clone(), queue.clone());
        outputs = outputs.buf(device.clone(), queue.clone());
        output_gradient = output_gradient.buf(device.clone(), queue.clone());
        inner_gradient = inner_gradient.buf(device.clone(), queue.clone());
        weights_gradient = weights_gradient.buf(device.clone(), queue.clone());
        back_prop_gradient = back_prop_gradient.buf(device, queue);

        use ActivationFunction::*;
        let activation_function_index = match activation_function {
            Step => outputs.add_custom_single_op_in_place_pipeline(include_wgsl!(
                "../shaders/activation_functions/step.wgsl"
            )),
            Threshold(_) => outputs.add_custom_single_op_in_place_pipeline(include_wgsl!(
                "../shaders/activation_functions/threshold.wgsl"
            )),
            BinarySigmoid(_) => outputs.add_custom_single_op_in_place_pipeline(include_wgsl!(
                "../shaders/activation_functions/binary_sigmoid.wgsl"
            )),
            BipolarSigmoid(_) => outputs.add_custom_single_op_in_place_pipeline(include_wgsl!(
                "../shaders/activation_functions/bipolar_sigmoid.wgsl"
            )),
            ReLU => outputs.add_custom_single_op_in_place_pipeline(include_wgsl!(
                "../shaders/activation_functions/relu.wgsl"
            )),
            LeakyReLU(_) => outputs.add_custom_single_op_in_place_pipeline(include_wgsl!(
                "../shaders/activation_functions/leaky_relu.wgsl"
            )),
            HyperbolicTangent => outputs.add_custom_single_op_in_place_pipeline(include_wgsl!(
                "../shaders/activation_functions/hyperbolic_tangent.wgsl"
            )),
            Softmax => outputs.add_custom_single_op_in_place_pipeline(include_wgsl!(
                "../shaders/activation_functions/softmax.wgsl"
            )),
            Custom => {
                todo!()
            }
        }
        .unwrap();

        let activation_function_gradient_index = match activation_function {
            Step => outputs.add_custom_single_op_pipeline(include_wgsl!(
                "../shaders/activation_function_gradients/d_step.wgsl"
            )),
            Threshold(_) => outputs.add_custom_single_op_pipeline(include_wgsl!(
                "../shaders/activation_function_gradients/d_threshold.wgsl"
            )),
            BinarySigmoid(_) => outputs.add_custom_single_op_pipeline(include_wgsl!(
                "../shaders/activation_function_gradients/d_binary_sigmoid.wgsl"
            )),
            BipolarSigmoid(_) => outputs.add_custom_single_op_pipeline(include_wgsl!(
                "../shaders/activation_function_gradients/d_bipolar_sigmoid.wgsl"
            )),
            ReLU => outputs.add_custom_single_op_pipeline(include_wgsl!(
                "../shaders/activation_function_gradients/d_relu.wgsl"
            )),
            LeakyReLU(_) => outputs.add_custom_single_op_pipeline(include_wgsl!(
                "../shaders/activation_function_gradients/d_leaky_relu.wgsl"
            )),
            HyperbolicTangent => outputs.add_custom_single_op_pipeline(include_wgsl!(
                "../shaders/activation_function_gradients/d_hyperbolic_tangent.wgsl"
            )),
            Softmax => outputs.add_custom_multi_op_pipeline(include_wgsl!(
                "../shaders/activation_function_gradients/d_softmax.wgsl"
            )),
            Custom => {
                todo!()
            }
        }
        .unwrap();

        Self {
            weights,
            biases,
            inputs: linked_inputs,
            outputs: Rc::new(RefCell::new(outputs)),
            num_nodes,
            batch_size,
            activation_function,
            activation_function_index,
            activation_function_gradient_index,

            // -------------------------
            output_gradient,
            inner_gradient,
            weights_gradient,
            bias_gradient: 0.0,
            back_prop_gradient,
        }
    }

    pub fn get_num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn feed_forward(&mut self) -> Result<(), Box<dyn Error>> {
        Matrix::dot_into(
            &self.weights,
            &self.inputs.borrow(),
            &mut self.outputs.borrow_mut(),
        )?;
        self.outputs
            .borrow_mut()
            .vectored_add_in_place(&self.biases)?;
        self.outputs
            .borrow()
            .write_to_scalar(self.activation_function.get_scalar())?;
        self.outputs
            .borrow_mut()
            .run_custom_single_op_pipeline_in_place(self.activation_function_index)?;

        Ok(())
    }

    pub fn back_propogate(
        &mut self,
        // inputs: &mut Matrix,
        next_layer_back_prop: &Matrix,
    ) -> Result<&Matrix, Box<dyn Error>> {
        // println!("nlbp: {}", next_layer_back_prop);
        // Get the output gradient to find the weights gradient with
        match self.activation_function {
            ActivationFunction::Softmax => {
                Matrix::run_custom_multi_op_pipeline_into(
                    &self.outputs.borrow(),
                    next_layer_back_prop,
                    self.activation_function_gradient_index,
                    &mut self.output_gradient,
                )?;
            }
            _ => {
                Matrix::run_custom_single_op_pipeline_into(
                    &self.outputs.borrow(),
                    self.activation_function_gradient_index,
                    &mut self.output_gradient,
                )?;
            }
        }

        Matrix::elem_mult_into(
            next_layer_back_prop,
            &self.output_gradient,
            &mut self.inner_gradient,
        )?;

        self.inputs.borrow_mut().transpose_in_place();
        Matrix::dot_into(
            &self.inner_gradient,
            &self.inputs.borrow(),
            &mut self.weights_gradient,
        )?;
        self.inputs.borrow_mut().transpose_in_place();

        self.weights_gradient
            .mult_in_place(1.0 / self.batch_size as f32)?;

        self.bias_gradient = self.inner_gradient.sum()?;

        self.weights.transpose_in_place();
        Matrix::dot_into(
            &self.weights,
            &self.inner_gradient,
            &mut self.back_prop_gradient,
        )?;
        self.weights.transpose_in_place();

        Ok(&self.back_prop_gradient)
    }

    pub fn update_parameters(&mut self, learning_rate: f32) -> Result<(), Box<dyn Error>> {
        self.weights_gradient.mult_in_place(learning_rate)?;
        println!("wg a: {}", self.weights_gradient);
        self.weights.sub_in_place(&self.weights_gradient)?;
        self.biases
            .sub_scalar_in_place(learning_rate * self.bias_gradient)?;

        Ok(())
    }

    pub fn output_link(&self) -> MatrixRef {
        self.outputs.clone()
    }
}
