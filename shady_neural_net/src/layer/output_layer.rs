use std::rc::Rc;

use crate::create_buffer_bind_group;
use crate::layer::compute_2d_workgroup_size;
use crate::layer_structs::regularization::*;
use crate::utils::{get_buffer, print_buffer, read_buffer};

use super::{BackPropogationLayer, D2_WORK_GROUP_SIZE};
use super::{FeedForwardConnection, WORK_GROUP_SIZE, bias::Bias, compute_workgroup_size};
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderStages,
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct OutputLayer {
    pub num_inputs: u64,
    pub num_outputs: u64,

    // Buffers associated in feed forward computation
    dimensions_buffer: Rc<Buffer>,
    weights_buffer: Rc<Buffer>,
    bias_buffer: Buffer,
    intermediary_buffer: Buffer,
    output_buffer: Buffer,

    // Cost function buffer
    loss_function_buffer: Buffer,
    expected_values_buffer: Buffer,

    // buffers used in back propogation
    l_1_norm_buffer: Buffer,
    frobenius_norm_buffer: Buffer,
    regularization_info_buffer: Buffer,
    regularization_output_buffer: Buffer,
    gradient_buffer: Buffer,
    gradient_coefficient_buffer: Rc<Buffer>,

    // Bind group information
    input_buffer: Rc<Buffer>,
    input_bind_group_layout: BindGroupLayout,
    input_bind_group: BindGroup,

    // Main bind group information
    feed_forward_bind_group_layout: BindGroupLayout,
    feed_forward_bind_group: BindGroup,

    // Cost function bind group information
    loss_function_bind_group_layout: BindGroupLayout,
    loss_function_bind_group: BindGroup,

    // Back Propogation bind groups
    back_propogation_bind_group_layout: BindGroupLayout,
    back_propogation_bind_group: BindGroup,

    // Gradient Descent Bind groups
    gradient_descent_bind_group_layout: Option<BindGroupLayout>,
    gradient_descent_bind_group: Option<BindGroup>,

    // GPU Pipeline Information
    feed_forward_pipeline: ComputePipeline,
    loss_function_pipeline: ComputePipeline,
    regularization_pipeline: ComputePipeline,
    gradient_descent_pipeline: Option<ComputePipeline>,
}

impl OutputLayer {
    /// Initialize a new output layer with random weights and biases
    ///
    /// # Arguments
    ///
    /// * `input_connecting_bind_group` - Bind group reference from the previous layer
    /// * `num_outputs` - number of outputs in this layer
    /// * `device` - a reference to wgpu device to create necessary buffers
    ///
    /// # Returns
    ///
    /// A new instance of `OutputLayer`
    pub fn new(
        feed_forward_input: &FeedForwardConnection,
        num_outputs: u64,
        device: &Device,
    ) -> Self {
        let (input_bind_group_layout, input_bind_group) = create_buffer_bind_group!(
            device,
            "Output Layer Input Bind Group",
            (
                0,
                &feed_forward_input.buffer,
                Bbt::Storage { read_only: true }
            )
        );

        // Create all the buffers necessary in this layer
        let (
            dimensions_buffer,
            weights_buffer,
            bias_buffer,
            intermediary_buffer,
            output_buffer,
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            regularization_output_buffer,
            gradient_buffer,
            gradient_coefficient_buffer,
            loss_function_buffer,
            expected_values_buffer,
        ) = {
            let dimensions_buffer = {
                let mut dimensions = Vec::new();
                dimensions.push(feed_forward_input.num_inputs as u32);
                dimensions.push(num_outputs as u32);

                Rc::new(device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Output Layer Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&dimensions),
                    usage: BufferUsages::UNIFORM,
                }))
            };

            let weights_buffer = {
                let mut weights: Vec<f32> = Vec::new();
                for _ in 0..feed_forward_input.num_inputs {
                    for _ in 0..num_outputs {
                        weights.push(rand::random_range(-1.0..=1.0));
                    }
                }

                Rc::new(device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Output Layer Weights Buffer"),
                    contents: bytemuck::cast_slice(&weights),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                }))
            };

            let bias_buffer = {
                let mut biases = Vec::new();
                for _ in 0..num_outputs {
                    biases.push(Bias::new(
                        rand::random_range(-1.0..=1.0),
                        rand::random_range(-1.0..=1.0),
                    ));
                }

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Output Layer Bias Buffer"),
                    contents: bytemuck::cast_slice(&biases),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                })
            };

            let intermediary_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Intermediary Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

            let output_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

            let l_1_norm_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer L1 Norm Buffer"),
                mapped_at_creation: false,
                size: std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let frobenius_norm_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Frobenius Norm Buffer"),
                mapped_at_creation: false,
                size: std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let regularization_info_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Regularization Info Buffer"),
                mapped_at_creation: false,
                size: std::mem::size_of::<(u32, f32, f32)>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let regularization_output_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Regularization Output Buffer"),
                mapped_at_creation: false,
                size: num_outputs
                    * feed_forward_input.num_inputs
                    * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let gradient_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Ouput Layer Gradient Buffer"),
                mapped_at_creation: false,
                size: num_outputs
                    * feed_forward_input.num_inputs
                    * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let gradient_coefficient_buffer = Rc::new(device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Gradient Coefficient Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }));

            let loss_function_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Loss Function Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let expected_values_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Expected Values Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

            (
                dimensions_buffer,
                weights_buffer,
                bias_buffer,
                intermediary_buffer,
                output_buffer,
                l_1_norm_buffer,
                frobenius_norm_buffer,
                regularization_info_buffer,
                regularization_output_buffer,
                gradient_buffer,
                gradient_coefficient_buffer,
                loss_function_buffer,
                expected_values_buffer,
            )
        };

        let (feed_forward_bind_group_layout, feed_forward_bind_group) = create_buffer_bind_group!(
            device,
            "Output Layer Feed Forward Bind Group",
            (0, &dimensions_buffer, Bbt::Uniform),
            (1, &weights_buffer, Bbt::Storage { read_only: true }),
            (2, &bias_buffer, Bbt::Storage { read_only: true }),
            (3, &intermediary_buffer, Bbt::Storage { read_only: false }),
            (4, &output_buffer, Bbt::Storage { read_only: false })
        );

        let (loss_function_bind_group_layout, loss_function_bind_group) = create_buffer_bind_group!(
            device,
            "Output Layer Loss Function Bind Group",
            (0, &dimensions_buffer, Bbt::Uniform),
            (1, &output_buffer, Bbt::Storage { read_only: true }),
            (2, &expected_values_buffer, Bbt::Storage { read_only: true }),
            (3, &loss_function_buffer, Bbt::Storage { read_only: false }),
            (
                4,
                &gradient_coefficient_buffer,
                Bbt::Storage { read_only: false }
            )
        );

        let (back_propogation_bind_group_layout, back_propogation_bind_group) = create_buffer_bind_group!(
            device,
            "Output Lyaer Back Propogation Bind Group",
            (0, &l_1_norm_buffer, Bbt::Uniform),
            (1, &frobenius_norm_buffer, Bbt::Uniform),
            (2, &regularization_info_buffer, Bbt::Uniform),
            (
                3,
                &regularization_output_buffer,
                Bbt::Storage { read_only: false }
            ),
            (4, &dimensions_buffer, Bbt::Uniform),
            (5, &weights_buffer, Bbt::Storage { read_only: true }),
            (6, &gradient_buffer, Bbt::Storage { read_only: false }),
            (
                7,
                &gradient_coefficient_buffer,
                Bbt::Storage { read_only: true }
            )
        );

        // This is the main pipeline that is used when feeding information forward
        // through the neural network. This pipeline will not effect any of the
        // weights or biases that are created in this layer
        let feed_forward_pipeline = {
            let shader = device.create_shader_module(include_wgsl!(
                "../shaders/output_layer/output_layer_feed_forward.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Output Layer Feed Forward Compute Pipeline Layout"),
                bind_group_layouts: &[&input_bind_group_layout, &feed_forward_bind_group_layout],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Output Layer Feed Forward Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("output_layer_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        // This is the cost function computing pipeline. This will not effect
        // any of the weights or biases in this layer. It is used for computing
        // the cost function associated from the data that is given
        let loss_function_pipeline = {
            let shader = device.create_shader_module(include_wgsl!(
                "../shaders/output_layer/output_layer_cost_function.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Output Layer Cost Function Compute Pipeline Layout"),
                bind_group_layouts: &[&loss_function_bind_group_layout],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Output Layer Cost Function Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("output_layer_cost_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let regularization_pipeline = {
            let shader = device.create_shader_module(include_wgsl!(
                "../shaders/output_layer/output_layer_regularization.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Output Layer Regularization Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &back_propogation_bind_group_layout,
                    &input_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Output Layer Regularization Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("output_layer_regularization_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        Self {
            num_inputs: feed_forward_input.num_inputs,
            num_outputs,
            // -------------------------------
            dimensions_buffer,
            weights_buffer,
            bias_buffer,
            intermediary_buffer,
            output_buffer,
            // -------------------------------
            loss_function_buffer,
            expected_values_buffer,
            // -------------------------------
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            regularization_output_buffer,
            gradient_buffer,
            gradient_coefficient_buffer,
            // -------------------------------
            input_buffer: feed_forward_input.buffer.clone(),
            input_bind_group_layout,
            input_bind_group,
            // -------------------------------
            feed_forward_bind_group_layout,
            feed_forward_bind_group,
            // -------------------------------
            loss_function_bind_group_layout,
            loss_function_bind_group,
            // -------------------------------
            back_propogation_bind_group_layout,
            back_propogation_bind_group,
            // -------------------------------
            gradient_descent_bind_group_layout: None,
            gradient_descent_bind_group: None,
            // -------------------------------
            feed_forward_pipeline,
            loss_function_pipeline,
            regularization_pipeline,
            gradient_descent_pipeline: None,
        }
    }

    /// Runs the feed forward algorithm to the end of the output layer
    /// and store the output as the softmax vector of the output computation
    /// in the output buffer
    ///
    /// # Arguments
    ///
    /// * `device` - reference to the wgpu device to run shaders
    /// * `queue` - refernce to the wgpu queue to submit commands to the gpu
    pub fn feed_forward(&self, device: &Device, queue: &Queue) -> Vec<f32> {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Input Layer Command Encoder"),
        });

        let before = read_buffer(
            &self.input_buffer,
            self.num_inputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        // Run the pipeline
        {
            let dispatch_size = compute_workgroup_size(self.num_outputs as u32, WORK_GROUP_SIZE);

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Output Layer Feed Forward Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.feed_forward_pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &self.input_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.feed_forward_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder.insert_debug_marker("Sync Point: Ouptut Feed Forward Pipeline Finished");
        device.poll(Maintain::Wait);

        let output = read_buffer(
            &self.output_buffer,
            self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        print_buffer(&before, device, "Output Buffer Before");

        get_buffer(&output, device)
    }

    /// Computes the loss of the model based on some expected values
    /// Stores the loss computations in the loss buffer
    ///
    /// # Arguments
    ///
    /// * `expected_values` - slice of values expected for the given input
    /// * `device` - reference to the wgpu device for dispatching workgroups
    /// * `queue` - reference to the adapter queue for dispatching workgroups
    ///
    /// # Returns
    ///
    /// `f32` cost value: average of all loss values
    pub fn compute_loss(&self, expected_values: &[f32], device: &Device, queue: &Queue) -> f32 {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Output Layer Cost Function Command Encoder"),
        });

        // Write the expected values to the buffer
        queue.write_buffer(
            &self.expected_values_buffer,
            0,
            bytemuck::cast_slice(&expected_values),
        );

        // Run the pipeline
        {
            let dispatch_size = compute_workgroup_size(self.num_outputs as u32, WORK_GROUP_SIZE);

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Output Layer Cost Function Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.loss_function_pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &self.loss_function_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder.insert_debug_marker("Sync Point: Ouptut Cost Function Pipeline Finished");
        device.poll(Maintain::Wait);

        let loss = read_buffer(
            &self.loss_function_buffer,
            self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        // Get the average of all the loss values
        let loss_vector = get_buffer(&loss, device);

        loss_vector.iter().sum::<f32>() / loss_vector.len() as f32
    }

    /// Generates the regularization for the layer with the
    /// chosen regularization function and store the result
    /// in the gpu buffer to be used for back propogation
    ///
    /// # Arguments
    ///
    /// * `regularization` - A regularization container that contains the regularization function and the hyper paramter
    /// * `device` - a reference to wgpu device to send commands to
    /// * `queue` - a reference to wgpu queue to send command with
    ///
    /// # Returns
    ///
    /// `Vec<f32>` of the computed value of the regularization function for this layer
    pub fn back_propogate(&self, regularization: Regularization, device: &Device, queue: &Queue) {
        // representation of struct to send to gpu
        #[repr(C)]
        #[derive(Pod, Zeroable, Copy, Clone)]
        struct RegRepr {
            function: u32,
            hyper_parameter_1: f32,
            hyper_parameter_2: f32,
        }

        match regularization.function {
            RegularizationFunction::Lasso => {
                _ = self.generate_weights_l_1_norm(device, queue);
                queue.write_buffer(
                    &self.regularization_info_buffer,
                    0,
                    bytemuck::cast_slice(&[RegRepr {
                        function: 0,
                        hyper_parameter_1: regularization.hyper_parameter_1,
                        hyper_parameter_2: 0.0,
                    }]),
                );
            }
            RegularizationFunction::Ridge => {
                _ = self.generate_weights_frobenius_norm(device, queue);
                queue.write_buffer(
                    &self.regularization_info_buffer,
                    0,
                    bytemuck::cast_slice(&[RegRepr {
                        function: 1,
                        hyper_parameter_1: regularization.hyper_parameter_1,
                        hyper_parameter_2: 0.0,
                    }]),
                );
            }
            RegularizationFunction::ElasticNetRegression => {
                _ = self.generate_weights_l_1_norm(device, queue);
                _ = self.generate_weights_frobenius_norm(device, queue);
                queue.write_buffer(
                    &self.regularization_info_buffer,
                    0,
                    bytemuck::cast_slice(&[RegRepr {
                        function: 2,
                        hyper_parameter_1: regularization.hyper_parameter_1,
                        hyper_parameter_2: regularization.hyper_parameter_2,
                    }]),
                );
            }
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Dense Layer Regularization Command Encoder"),
        });

        {
            let (dispatch_width, dispatch_height) = compute_2d_workgroup_size(
                (self.num_inputs as u32, self.num_outputs as u32),
                (D2_WORK_GROUP_SIZE, D2_WORK_GROUP_SIZE),
            );

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Output Layer Regulariation Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.regularization_pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &self.back_propogation_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.input_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        encoder.insert_debug_marker("Sync Point: Output Regularization Pipeline Finished");
        device.poll(Maintain::Wait);

        let gradient = read_buffer(
            &self.gradient_buffer,
            self.num_inputs * self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        print_buffer(&gradient, device, "Output Layer Gradient Buffer");
    }

    /// Links the learning rate buffer to the layer and generates the bind group
    /// information for the gradient descent pass
    ///
    /// # Arguments
    ///
    /// * `device` - reference to the wgpu device to create the buffers
    /// * `learning_rate_buffer` - buffer containing the learning rate uniform
    pub fn link_gradient_descent_pipeline(
        &mut self,
        device: &Device,
        learning_rate_buffer: &Buffer,
    ) {
        let (gradient_descent_bind_group_layout, gradient_descent_bind_group) = create_buffer_bind_group!(
            device,
            "Output Layer Gradient Descent Bind Group",
            (0, learning_rate_buffer, Bbt::Uniform),
            (1, &self.gradient_buffer, Bbt::Storage { read_only: true }),
            (2, &self.weights_buffer, Bbt::Storage { read_only: false }),
            (3, &self.dimensions_buffer, Bbt::Uniform)
        );

        let gradient_descent_pipeline = {
            let shader = device.create_shader_module(include_wgsl!(
                "../shaders/output_layer/output_layer_gradient_descent.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Output Layer Gradient Descent Compute Pipeline Layout"),
                bind_group_layouts: &[&gradient_descent_bind_group_layout],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Output Layer Gradient Descent Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("output_layer_gradient_descent_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        self.gradient_descent_bind_group_layout = Some(gradient_descent_bind_group_layout);
        self.gradient_descent_bind_group = Some(gradient_descent_bind_group);
        self.gradient_descent_pipeline = Some(gradient_descent_pipeline);
    }

    /// Performs the gradient descent pass after the buffers have been linked
    /// and back propogation has been performed
    ///
    /// # Arguments
    ///
    /// * `device` - reference to the wgpu device for creating the command encoder
    /// * `queue` - reference to the wgpu queue to send command to the gpu
    pub fn gradient_descent(&self, device: &Device, queue: &Queue) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Output Layer Gradient Descent Command Encoder"),
        });

        // Run the gradient descent pass
        {
            let (dispatch_width, dispatch_height) = compute_2d_workgroup_size(
                (self.num_inputs as u32, self.num_outputs as u32),
                (D2_WORK_GROUP_SIZE, D2_WORK_GROUP_SIZE),
            );

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Output Layer Gradient Descent Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(self.gradient_descent_pipeline.as_ref().unwrap());

            // Set the bind groups
            compute_pass.set_bind_group(0, self.gradient_descent_bind_group.as_ref().unwrap(), &[]);

            // Dispatch the work groups
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        encoder.insert_debug_marker("Sync Point: Output Regularization Pipeline Finished");
        device.poll(Maintain::Wait);

        let weights = read_buffer(
            &self.weights_buffer,
            self.num_inputs * self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        print_buffer(&weights, device, "Output Layer New Weights Buffer");
    }

    /// Generates the frobenius norm of the weight matrix
    /// and stores it in the GPU buffer
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device reference to send commands to
    /// * `queue` - wgpu queue reference to write to the norm buffer
    ///
    /// # Returns
    ///
    /// `f32` value for the frobenius norm
    pub fn generate_weights_frobenius_norm(&self, device: &Device, queue: &Queue) -> f32 {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Output Layer Generate Weight Frobenius Norm Command Encoder"),
        });

        let weights_buffer = read_buffer(
            &self.weights_buffer,
            self.num_outputs * self.num_inputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        // sends the commands to copy the buffer to the gpu
        queue.submit(Some(encoder.finish()));

        let weights = get_buffer(&weights_buffer, device);

        let frobenius_norm = {
            // Σ of all squared values in matrix
            let squared_sum = weights
                .iter()
                .map(|weight| f32::powi(f32::abs(*weight), 2)) // WARN do I really need abs here?
                .sum();

            f32::sqrt(squared_sum)
        };

        // Write the norm to the buffer
        queue.write_buffer(
            &self.frobenius_norm_buffer,
            0,
            bytemuck::cast_slice(&[frobenius_norm]),
        );

        frobenius_norm
    }

    /// Generatea the L~1~ norm of the weight matrix
    /// and stores it in the GPU buffer
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device reference to send command to
    /// * `queue` - wgpu queue reference to write to the norm buffer
    ///
    /// # Returns
    ///
    /// `f32` value for the L~1~ norm
    pub fn generate_weights_l_1_norm(&self, device: &Device, queue: &Queue) -> f32 {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Dense Layer Generate Weight L1 Norm Command Encoder"),
        });

        let weights_buffer = read_buffer(
            &self.weights_buffer,
            self.num_outputs * self.num_inputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        // sends the command to copy the buffer to the gpu
        queue.submit(Some(encoder.finish()));

        let weights = get_buffer(&weights_buffer, device);

        // Σ of all absolute values in matrix
        let l_1_norm = weights.iter().map(|weight| f32::abs(*weight)).sum();

        // Write the norm to the buffer
        queue.write_buffer(&self.l_1_norm_buffer, 0, bytemuck::cast_slice(&[l_1_norm]));

        l_1_norm
    }
}

impl BackPropogationLayer for OutputLayer {
    fn get_gradient_coefficient_buffer(&self) -> Rc<Buffer> {
        self.gradient_coefficient_buffer.clone()
    }

    fn get_weights_buffer(&self) -> Rc<Buffer> {
        self.weights_buffer.clone()
    }

    fn get_dimensions_buffer(&self) -> Rc<Buffer> {
        self.dimensions_buffer.clone()
    }
}
