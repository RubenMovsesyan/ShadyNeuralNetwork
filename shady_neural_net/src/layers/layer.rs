use std::{cell::RefCell, error::Error, rc::Rc};

use anyhow::Result;
use gpu_math::{GpuMath, math::matrix::Matrix};

use crate::neural_network::Parameters;

use super::{MatrixRef, activation_function::ActivationFunction};

#[derive(Debug)]
pub struct Layer {
    weights: Matrix,
    biases: Matrix,
    inputs: MatrixRef,
    outputs: MatrixRef,
    num_inputs: u32,
    num_nodes: u32,
    batch_size: u32,
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
    /// Creates a new layer with the given parameter
    ///
    /// # Arguments
    ///
    /// * `num_inputs` - number of inputs this layer has
    /// * `num_nodes` - the number of nodes on this layer
    /// * `linked_inputs` - Reference to the outputs of the previous layer
    /// * `batch_size` - the size of the batch that the layer will be training with
    /// * `activattion_function` - The activation function for this layer (make sure to use the proper activation function for the type of layer this is, i.e. ReLU for a hidden layer or softmax for an output layer)
    /// * `device` - reference counted wgpu device for this layer to use when creating all the necessary buffers
    /// * `queue` - reference counted wgpu queue for this layer to use when creting all the necessary buffers
    ///
    /// # Returns
    ///
    /// `Layer` with the parameters created and set by the defined parameters
    pub fn new(
        num_inputs: u32,
        num_nodes: u32,
        linked_inputs: MatrixRef,
        batch_size: u32,
        activation_function: ActivationFunction,
        gpu_math: &mut GpuMath,
    ) -> Result<Self> {
        // Random distribution for now, add in the ability for custom distributions later
        let weights = Matrix::new(
            gpu_math,
            (num_nodes, num_inputs),
            Some({
                (0..(num_nodes * num_inputs))
                    .into_iter()
                    .map(|_| rand::random_range(-0.5..=0.5))
                    .collect::<Vec<_>>()
            }),
        )?;

        let biases = Matrix::new(
            gpu_math,
            (num_nodes, 1),
            Some({
                (0..(num_nodes))
                    .into_iter()
                    .map(|_| rand::random_range(-0.5..=0.5))
                    .collect::<Vec<_>>()
            }),
        )?;

        let outputs = Matrix::new(gpu_math, (num_nodes, batch_size), None)?;
        let output_gradient = Matrix::new(gpu_math, (num_nodes, batch_size), None)?;
        let inner_gradient = Matrix::new(gpu_math, (num_nodes, batch_size), None)?;
        let weights_gradient = Matrix::new(gpu_math, (num_nodes, num_inputs), None)?;
        let back_prop_gradient = Matrix::new(gpu_math, (num_inputs, batch_size), None)?;

        use ActivationFunction::*;
        // Set the custom matrix operation for the activation function described
        let activation_function_index = match activation_function {
            Step => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_functions/step.wgsl"
            )),
            Threshold(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(include_str!(
                "../shaders/activation_functions/threshold.wgsl"
            )),
            BinarySigmoid(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(
                include_str!("../shaders/activation_functions/binary_sigmoid.wgsl"),
            ),
            BipolarSigmoid(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(
                include_str!("../shaders/activation_functions/bipolar_sigmoid.wgsl"),
            ),
            ReLU => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_functions/relu.wgsl"
            )),
            LeakyReLU(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(include_str!(
                "../shaders/activation_functions/leaky_relu.wgsl"
            )),
            HyperbolicTangent => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_functions/hyperbolic_tangent.wgsl"
            )),
            Softmax => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_functions/softmax.wgsl"
            )),
            Custom => {
                todo!()
            }
        };

        // Set the gradient for the activation function described
        let activation_function_gradient_index = match activation_function {
            Step => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_step.wgsl"
            )),
            Threshold(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_threshold.wgsl"
            )),
            BinarySigmoid(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(
                include_str!("../shaders/activation_function_gradients/d_binary_sigmoid.wgsl"),
            ),
            BipolarSigmoid(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(
                include_str!("../shaders/activation_function_gradients/d_bipolar_sigmoid.wgsl"),
            ),
            ReLU => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_relu.wgsl"
            )),
            LeakyReLU(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_leaky_relu.wgsl"
            )),
            HyperbolicTangent => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_hyperbolic_tangent.wgsl"
            )),
            Softmax => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_softmax.wgsl"
            )),
            Custom => {
                todo!()
            }
        };

        Ok(Self {
            weights,
            biases,
            inputs: linked_inputs,
            outputs: Rc::new(RefCell::new(outputs)),
            num_inputs,
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
        })
    }

    /// Creates a layer from the weights and biases described in `parameters`
    /// everything else is still needs to be specified when creating the network
    ///
    /// # Arguments
    ///
    /// * `parameters` - information about the weights and biases for this layer
    /// * `batch_size` - the size of the batch that is going to be sent though this network
    /// * `linked_inputs` - the outputs from the previous layer
    /// * `device` - wgpu device to create all the gpu matrices
    /// * `queue` - wgpu queue to create all the gpu matrices
    ///
    /// # Returns
    ///
    /// `Layer` with the weights and biases specified in `parameters`
    pub fn from_parameters(
        parameters: &Parameters,
        batch_size: u32,
        linked_inputs: MatrixRef,
        gpu_math: &mut GpuMath,
    ) -> Result<Self> {
        let (weights, biases, activation_function, inputs, outputs) = parameters;

        let weights_matrix = Matrix::new(gpu_math, (*outputs, *inputs), Some(weights.clone()))?;
        let biases_matrix = Matrix::new(gpu_math, (*outputs, 1), Some(biases.clone()))?;

        let layer_outputs = Matrix::new(gpu_math, (*outputs, batch_size), None)?;
        let output_gradient = Matrix::new(gpu_math, (*outputs, batch_size), None)?;
        let inner_gradient = Matrix::new(gpu_math, (*outputs, batch_size), None)?;
        let weights_gradient = Matrix::new(gpu_math, (*outputs, *inputs), None)?;
        let back_prop_gradient = Matrix::new(gpu_math, (*inputs, batch_size), None)?;

        use ActivationFunction::*;
        // Set the custom matrix operation for the activation function described
        let activation_function_index = match activation_function {
            Step => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_functions/step.wgsl"
            )),
            Threshold(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(include_str!(
                "../shaders/activation_functions/threshold.wgsl"
            )),
            BinarySigmoid(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(
                include_str!("../shaders/activation_functions/binary_sigmoid.wgsl"),
            ),
            BipolarSigmoid(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(
                include_str!("../shaders/activation_functions/bipolar_sigmoid.wgsl"),
            ),
            ReLU => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_functions/relu.wgsl"
            )),
            LeakyReLU(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(include_str!(
                "../shaders/activation_functions/leaky_relu.wgsl"
            )),
            HyperbolicTangent => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_functions/hyperbolic_tangent.wgsl"
            )),
            Softmax => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_functions/softmax.wgsl"
            )),
            Custom => {
                todo!()
            }
        };

        // Set the gradient for the activation function described
        let activation_function_gradient_index = match activation_function {
            Step => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_step.wgsl"
            )),
            Threshold(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_threshold.wgsl"
            )),
            BinarySigmoid(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(
                include_str!("../shaders/activation_function_gradients/d_binary_sigmoid.wgsl"),
            ),
            BipolarSigmoid(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(
                include_str!("../shaders/activation_function_gradients/d_bipolar_sigmoid.wgsl"),
            ),
            ReLU => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_relu.wgsl"
            )),
            LeakyReLU(_) => gpu_math.create_custom_matrix_scalar_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_leaky_relu.wgsl"
            )),
            HyperbolicTangent => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_hyperbolic_tangent.wgsl"
            )),
            Softmax => gpu_math.create_custom_matrix_in_place_pipeline(include_str!(
                "../shaders/activation_function_gradients/d_softmax.wgsl"
            )),
            Custom => {
                todo!()
            }
        };

        Ok(Self {
            weights: weights_matrix,
            biases: biases_matrix,
            inputs: linked_inputs,
            outputs: Rc::new(RefCell::new(layer_outputs)),
            num_inputs: *inputs,
            num_nodes: *outputs,
            batch_size,
            activation_function: *activation_function,
            activation_function_index,
            activation_function_gradient_index,

            // -------------------------
            output_gradient,
            inner_gradient,
            weights_gradient,
            bias_gradient: 0.0,
            back_prop_gradient,
        })
    }

    /// Gets the number of nodes that are in this layer
    ///
    /// # Returns
    ///
    /// `usize` of the number of nodes
    pub fn get_num_nodes(&self) -> u32 {
        self.num_nodes
    }

    /// Runs the feed forward algorithm on this layer by multiplying the inputs by the weight matrix,
    /// running that through the activation function, and adding the bias vector to the result
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the feed forward was successful, and `Err` if failed
    pub fn feed_forward(&mut self) -> Result<(), Box<dyn Error>> {
        // Matrix::dot_into(
        //     &self.weights,
        //     &self.inputs.borrow(),
        //     &mut self.outputs.borrow_mut(),
        // )?;
        Matrix::dot(&self.weights, &self.inputs.borrow(), &self.outputs.borrow())?;

        // self.outputs
        //     .borrow_mut()
        //     .vectored_add_in_place(&self.biases)?;
        Matrix::vectored_add_in_place(&self.outputs.borrow(), &self.biases)?;

        // self.outputs
        //     .borrow()
        //     .write_to_scalar(self.activation_function.get_scalar())?;

        // self.outputs
        //     .borrow_mut()
        //     .run_custom_single_op_pipeline_in_place(self.activation_function_index)?;

        use ActivationFunction::*;
        match self.activation_function {
            BinarySigmoid(a) | BipolarSigmoid(a) | LeakyReLU(a) | Threshold(a) => {
                Matrix::run_custom_matrix_scalar_in_place(
                    &self.outputs.borrow(),
                    a,
                    self.activation_function_index,
                )?;
            }
            _ => Matrix::run_custom_matrix_in_place(
                &self.outputs.borrow(),
                self.activation_function_index,
            )?,
        }

        Ok(())
    }

    /// Runs the back propogate algorithm on this layer storing the weights gradient and the bias gradient in
    /// their respective buffers
    ///
    /// # Arguments
    ///
    /// `next_layer_back_prop` - reference to a `Matrix` that contains the next layers back propogation gradient multiplied by the next layers weights, in the case of the output layer, it is the derivative of the loss function
    ///
    /// # Returns
    ///
    /// `Result` with `Ok(&Matrix)` that contains the gradient that needs to be send back to the previous layer and `Err` if it failed
    pub fn back_propogate(
        &mut self,
        next_layer_back_prop: &Matrix,
    ) -> Result<&Matrix, Box<dyn Error>> {
        // Get the output gradient to find the weights gradient with
        match self.activation_function {
            ActivationFunction::Softmax => {
                // In the case of softmax we don't need to multiply by the next layer back prop
                // because we need to use it to compute the Jacobian matrix anyways
                // Matrix::run_custom_multi_op_pipeline_into(
                //     &self.outputs.borrow(),
                //     next_layer_back_prop,
                //     self.activation_function_gradient_index,
                //     &mut self.inner_gradient,
                // )?;
                Matrix::run_custom_matrix_matrix(
                    &self.outputs.borrow(),
                    next_layer_back_prop,
                    &self.inner_gradient,
                    self.activation_function_gradient_index,
                )?;
            }
            _ => {
                // Matrix::run_custom_single_op_pipeline_into(
                //     &self.outputs.borrow(),
                //     self.activation_function_gradient_index,
                //     &mut self.output_gradient,
                // )?;
                Matrix::run_custom_matrix(
                    &self.outputs.borrow(),
                    &self.output_gradient,
                    self.activation_function_gradient_index,
                )?;

                // Matrix::elem_mult_into(
                //     next_layer_back_prop,
                //     &self.output_gradient,
                //     &mut self.inner_gradient,
                // )?;
                Matrix::mult(
                    next_layer_back_prop,
                    &self.output_gradient,
                    &self.inner_gradient,
                )?;
            }
        }

        // Dot the inputs with the outputs of this layer that have been scaled by the loss gradient
        // dZ * h^T
        // self.inputs.borrow_mut().transpose_in_place();
        self.inputs.borrow_mut().transpose()?;
        // Matrix::dot_into(
        //     &self.inner_gradient,
        //     &self.inputs.borrow(),
        //     &mut self.weights_gradient,
        // )?;
        Matrix::dot(
            &self.inner_gradient,
            &self.inputs.borrow(),
            &self.weights_gradient,
        )?;
        // self.inputs.borrow_mut().transpose_in_place();
        self.inputs.borrow_mut().transpose()?; // Do I need to transpose it back?

        // Make sure to normalize the weight gradient by the batch size
        // self.weights_gradient
        //     .mult_in_place(1.0 / self.batch_size as f32)?;
        Matrix::mult_scalar_in_place(&self.weights_gradient, 1.0 / self.batch_size as f32)?;

        // Also make sure to normalize the bias gradient by the batch size
        // self.bias_gradient = self.inner_gradient.sum()? / self.batch_size as f32;
        self.bias_gradient = Matrix::sum(&self.inner_gradient)? / self.batch_size as f32;

        // Compute the gradient that gets sent back to the previous layer
        // W^T * dZ
        // self.weights.transpose_in_place();
        self.weights.transpose()?;
        // Matrix::dot_into(
        //     &self.weights,
        //     &self.inner_gradient,
        //     &mut self.back_prop_gradient,
        // )?;
        Matrix::dot(
            &self.weights,
            &self.inner_gradient,
            &self.back_prop_gradient,
        )?;
        // self.weights.transpose_in_place();
        self.weights.transpose()?;

        Ok(&self.back_prop_gradient)
    }

    /// Updates the weights and biases of this layer based on the gradients computed in the back propogation stage
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - learning rate to scale the weight and bias gradients by
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the update was successful and `Err` if the update failed
    pub fn update_parameters(&mut self, learning_rate: f32) -> Result<(), Box<dyn Error>> {
        // self.weights_gradient.mult_in_place(learning_rate)?;
        Matrix::mult_scalar_in_place(&self.weights_gradient, learning_rate)?;
        // self.weights.sub_in_place(&self.weights_gradient)?;
        Matrix::sub_in_place(&self.weights, &self.weights_gradient)?;
        // self.biases
        //     .sub_scalar_in_place(learning_rate * self.bias_gradient)?;
        Matrix::sub_scalar_in_place(&self.biases, learning_rate * self.bias_gradient)?;

        Ok(())
    }

    /// Gets a reference to the outputs of this layer to be used in the next layer
    ///
    /// # Returns
    ///
    /// `MatrixRef` of the outputs to be referenced by the next layer
    pub fn output_link(&self) -> MatrixRef {
        self.outputs.clone()
    }

    /// Sets the input link for this layer
    ///
    /// # Arguments
    ///
    /// * `input` - refernce to the matrix of the inputs
    pub fn input_link(&mut self, input: MatrixRef) {
        self.inputs = input;
    }

    /// Saves the parameters of this layer to be used later to recreate this layer
    ///
    /// # Returns
    ///
    /// `Ok` with `(Vec<f32>, Vec<f32>, ActivationFunction)` representing the weights, biases, and activation function respectively
    /// if successful, and `Err` if failed
    pub fn save_parameters(&mut self) -> Result<Parameters, Box<dyn Error>> {
        Ok((
            self.weights.get_inner()?,
            self.biases.get_inner()?,
            self.activation_function,
            self.num_inputs,
            self.num_nodes,
        ))
    }
}
