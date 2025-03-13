use std::rc::Rc;

use crate::{
    create_buffer_bind_group,
    layer::{D2_WORK_GROUP_SIZE, compute_2d_workgroup_size},
    regularization::RegularizationFunction,
    utils::{get_buffer, print_buffer, read_buffer},
};

use super::{
    BackPropogationConnection, BackPropogationLayer, FeedForwardConnection, FeedForwardLayer,
    WORK_GROUP_SIZE, activation::ActivationFunction, bias::Bias, compute_workgroup_size,
    regularization::Regularization,
};
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderModule,
    ShaderStages, include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

/// Dense layer struct used in neural net
#[allow(dead_code)]
#[derive(Debug)]
pub struct DenseLayer {
    pub num_nodes: u64,
    pub num_inputs: u64,
    activation_function: ActivationFunction,

    // Feed forward buffers
    dimensions_buffer: Rc<Buffer>,
    weights_buffer: Rc<Buffer>,
    bias_buffer: Buffer,
    intermediary_buffer: Buffer,
    output_buffer: Rc<Buffer>,

    // buffers used in back propogation
    l_1_norm_buffer: Buffer,
    frobenius_norm_buffer: Buffer,
    regularization_info_buffer: Buffer,
    regularization_output_buffer: Buffer,
    gradient_buffer: Buffer,
    gradient_coefficient_buffer: Rc<Buffer>,

    // Input Bind group information
    input_buffer: Rc<Buffer>,
    input_bind_group_layout: BindGroupLayout,
    input_bind_group: BindGroup,

    // Feed forward bind group information
    feed_forward_bind_group_layout: BindGroupLayout,
    feed_forward_bind_group: BindGroup,

    // Back Propogation bind groups
    back_propogation_bind_group_layout: BindGroupLayout,
    back_propogation_bind_group: BindGroup,

    // Gradient descent bind groups
    gradient_descent_bind_group_layout: Option<BindGroupLayout>,
    gradient_descent_bind_group: Option<BindGroup>,

    // GPU pipeline information
    feed_forward_pipeline: ComputePipeline,
    coefficient_forming_pipeline: Option<ComputePipeline>,
    regularization_pipeline: Option<ComputePipeline>,
    gradient_descent_pipeline: Option<ComputePipeline>,

    // Buffer information that needs to be linked after creation
    next_layer_gradient_coefficient_buffer: Option<Rc<Buffer>>,
    next_layer_weights_buffer: Option<Rc<Buffer>>,
    next_layer_dimensions_buffer: Option<Rc<Buffer>>,
    next_layer_bind_group_layout: Option<BindGroupLayout>,
    next_layer_bind_group: Option<BindGroup>,

    // Buffer information that is needed to get the gradient coefficient
    coefficient_forming_bind_group_layout: BindGroupLayout,
    coefficient_forming_bind_group: BindGroup,
}

impl DenseLayer {
    /// Initialize a new dense layer with random weights and biases
    ///
    /// # Arguments
    ///
    /// * `input_connection_bind_group` - Bind group reference from the previous layer
    /// * `num_nodes` - number of nodes in this layer
    /// * `activation_function` - activation function type and function parameter if necessary
    /// * `device` - a reference to wgpu device to create necessary buffers
    ///
    /// # Returns
    ///
    /// A new instance of `DenseLayer`
    pub fn new(
        input_connecting_bind_group: &FeedForwardConnection,
        num_nodes: u64,
        activation_function: ActivationFunction,
        device: &Device,
    ) -> Self {
        let (input_bind_group_layout, input_bind_group) = create_buffer_bind_group!(
            device,
            "Dense Layer Input Bind Group",
            (
                0,
                &input_connecting_bind_group.buffer,
                Bbt::Storage { read_only: true }
            )
        );

        // Create all the buffers necessary in this layer
        let (
            dimensions_buffer,
            weights_buffer,
            bias_buffer,
            activation_function_buffer,
            intermediary_buffer,
            output_buffer,
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            regularization_output_buffer,
            gradient_buffer,
            gradient_coefficient_buffer,
        ) = {
            let dimensions_buffer = {
                let mut dimensions = Vec::new();
                dimensions.push(input_connecting_bind_group.num_inputs as u32);
                dimensions.push(num_nodes as u32);

                Rc::new(device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Dense Layer Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&dimensions),
                    usage: BufferUsages::UNIFORM,
                }))
            };

            // Initialize the weights matrix buffer with random values from -1.0 to 1.0
            // containts a matrix with num_nodes sets of weights
            // each with num_inputs weights in them
            let weights_buffer = {
                let mut weights: Vec<f32> = Vec::new();

                for _ in 0..input_connecting_bind_group.num_inputs {
                    for _ in 0..num_nodes {
                        weights.push(rand::random_range(-1.0..=1.0));
                    }
                }

                Rc::new(device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Dense Layer Weights Buffer"),
                    contents: bytemuck::cast_slice(&weights),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                }))
            };

            // Initialize the bias vector buffer with random values from -1.0 to 1.0
            // each Bias is a bias value and a bias weight
            let bias_buffer = {
                let mut biases = Vec::new();
                for _ in 0..num_nodes {
                    biases.push(Bias::new(
                        rand::random_range(-1.0..=1.0),
                        rand::random_range(-1.0..=1.0),
                    ));
                }

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Dense Layer Bias Buffer"),
                    contents: bytemuck::cast_slice(&biases),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                })
            };

            let activation_function_buffer = {
                use ActivationFunction::*;

                #[repr(C)]
                #[derive(Debug, Pod, Zeroable, Copy, Clone)]
                struct ActivationFunctionRepresentation {
                    function_type: u32,
                    function_parameter: f32,
                }

                // Create a struct of data to send to the GPU that contains
                // The information needed for the activation function
                let data = match activation_function {
                    Step => ActivationFunctionRepresentation {
                        function_type: 0,
                        function_parameter: 0.0,
                    },
                    Threshold(threshold_function) => ActivationFunctionRepresentation {
                        function_type: 1,
                        function_parameter: threshold_function.threshold_value,
                    },
                    BinarySigmoid(binary_sigmoid_function) => ActivationFunctionRepresentation {
                        function_type: 2,
                        function_parameter: binary_sigmoid_function.k,
                    },
                    BipolarSigmoid(bipolar_sigmoid_function) => ActivationFunctionRepresentation {
                        function_type: 3,
                        function_parameter: bipolar_sigmoid_function.k,
                    },
                    ReLU => ActivationFunctionRepresentation {
                        function_type: 4,
                        function_parameter: 0.0,
                    },
                    LeakyReLU(leaky_relu_function) => ActivationFunctionRepresentation {
                        function_type: 5,
                        function_parameter: leaky_relu_function.a,
                    },
                    HyperbolicTangent => ActivationFunctionRepresentation {
                        function_type: 6,
                        function_parameter: 0.0,
                    },
                };

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Dense Layer Activation Function Uniform Buffer"),
                    contents: bytemuck::cast_slice(&[data]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                })
            };

            let intermediary_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer Intermediary Buffer"),
                mapped_at_creation: false,
                size: num_nodes * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

            let output_buffer = Rc::new(device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer Output Buffer"),
                mapped_at_creation: false,
                size: num_nodes * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            }));

            let l_1_norm_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer L1 Norm Buffer"),
                mapped_at_creation: false,
                size: std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

            let frobenius_norm_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer Frobenius Norm Buffer"),
                mapped_at_creation: false,
                size: std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

            let regularization_info_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer Regularization Buffer"),
                mapped_at_creation: false,
                size: std::mem::size_of::<(u32, f32, f32)>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let regularization_output_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer Regularization Output Buffer"),
                mapped_at_creation: false,
                size: input_connecting_bind_group.num_inputs
                    * num_nodes
                    * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let gradient_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer Gradient Buffer"),
                mapped_at_creation: false,
                size: input_connecting_bind_group.num_inputs
                    * num_nodes
                    * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let gradient_coefficient_buffer = Rc::new(device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer Gradient Coefficient Buffer"),
                mapped_at_creation: false,
                size: num_nodes * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }));

            (
                dimensions_buffer,
                weights_buffer,
                bias_buffer,
                activation_function_buffer,
                intermediary_buffer,
                output_buffer,
                l_1_norm_buffer,
                frobenius_norm_buffer,
                regularization_info_buffer,
                regularization_output_buffer,
                gradient_buffer,
                gradient_coefficient_buffer,
            )
        };

        let (feed_forward_bind_group_layout, feed_forward_bind_group) = create_buffer_bind_group!(
            device,
            "Dense Layer Feed Forward Bind Group",
            (0, &dimensions_buffer, Bbt::Uniform),
            (1, &weights_buffer, Bbt::Storage { read_only: true }),
            (2, &bias_buffer, Bbt::Storage { read_only: true }),
            (3, &activation_function_buffer, Bbt::Uniform),
            (4, &intermediary_buffer, Bbt::Storage { read_only: false }),
            (5, &output_buffer, Bbt::Storage { read_only: false })
        );

        let (back_propogation_bind_group_layout, back_propogation_bind_group) = create_buffer_bind_group!(
            device,
            "Dense Layer Back Propogation Bind Group",
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

        let (coefficient_forming_bind_group_layout, coefficient_forming_bind_group) = create_buffer_bind_group!(
            device,
            "Dense Layer Coefficient Forming Bind Group",
            (
                0,
                &gradient_coefficient_buffer,
                Bbt::Storage { read_only: false }
            ),
            (1, &activation_function_buffer, Bbt::Uniform),
            (
                2,
                &input_connecting_bind_group.buffer,
                Bbt::Storage { read_only: true }
            )
        );

        // Create the pipeline from the bind group layout
        let feed_forward_pipeline = {
            let shader: ShaderModule = device.create_shader_module(include_wgsl!(
                "../shaders/dense_layer/dense_layer_feed_forward.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Dense Layer Compute Pipeline Layout"),
                bind_group_layouts: &[&input_bind_group_layout, &feed_forward_bind_group_layout],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Dense Layer Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("dense_layer_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        Self {
            num_nodes,
            num_inputs: input_connecting_bind_group.num_inputs,
            activation_function,
            // ------------------------------------
            dimensions_buffer,
            weights_buffer,
            bias_buffer,
            intermediary_buffer,
            output_buffer,
            // ------------------------------------
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            regularization_output_buffer,
            gradient_buffer,
            gradient_coefficient_buffer,
            // ------------------------------------
            input_buffer: input_connecting_bind_group.buffer.clone(),
            input_bind_group_layout,
            input_bind_group,
            // ------------------------------------
            feed_forward_bind_group_layout,
            feed_forward_bind_group,
            // ------------------------------------
            back_propogation_bind_group_layout,
            back_propogation_bind_group,
            // ------------------------------------
            gradient_descent_bind_group_layout: None,
            gradient_descent_bind_group: None,
            // ------------------------------------
            feed_forward_pipeline,
            coefficient_forming_pipeline: None,
            regularization_pipeline: None,
            gradient_descent_pipeline: None,
            // ------------------------------------
            next_layer_gradient_coefficient_buffer: None,
            next_layer_weights_buffer: None,
            next_layer_dimensions_buffer: None,
            next_layer_bind_group_layout: None,
            next_layer_bind_group: None,
            coefficient_forming_bind_group_layout,
            coefficient_forming_bind_group,
        }
    }

    /// Links the weights from the next layer to this layer to be used
    /// during back propogation
    ///
    /// # Arguments
    ///
    /// * `device` - reference to the wgpu device to create bind groups
    /// * `back_propogation_connection` - Link Descriptor to the next layers weights
    pub fn link_next_layer(
        &mut self,
        device: &Device,
        back_propogation_connection: &BackPropogationConnection,
    ) {
        self.next_layer_gradient_coefficient_buffer = Some(
            back_propogation_connection
                .gradient_coefficient_buffer
                .clone(),
        );

        self.next_layer_weights_buffer = Some(back_propogation_connection.weights_buffer.clone());

        let (next_layer_bind_group_layout, next_layer_bind_group) = create_buffer_bind_group!(
            device,
            "Dense Layer Next Layer Bind Group",
            (
                0,
                &back_propogation_connection.gradient_coefficient_buffer,
                Bbt::Storage { read_only: true }
            ),
            (
                1,
                &back_propogation_connection.weights_buffer,
                Bbt::Storage { read_only: true }
            ),
            (
                2,
                &back_propogation_connection.dimensions_buffer,
                Bbt::Uniform
            )
        );

        // Create the pipeline to compute the coefficient
        let coefficient_forming_pipeline = {
            let shader = device.create_shader_module(include_wgsl!(
                "../shaders/dense_layer/dense_layer_coefficient_forming.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Dense Layer Coefficient Forming Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &next_layer_bind_group_layout,
                    &self.coefficient_forming_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Dense Layer Coefficient Forming Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("dense_layer_coefficient_forming_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        // Create the pipeline to compute the regularization function
        let regularization_pipeline = {
            let shader = device.create_shader_module(include_wgsl!(
                "../shaders/dense_layer/dense_layer_regularization.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Dense Layer Regularization Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.input_bind_group_layout,
                    &self.back_propogation_bind_group_layout,
                    &next_layer_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Dense Layer Regularization Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("dense_layer_regularization_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        self.next_layer_bind_group_layout = Some(next_layer_bind_group_layout);
        self.next_layer_bind_group = Some(next_layer_bind_group);

        self.regularization_pipeline = Some(regularization_pipeline);
        self.coefficient_forming_pipeline = Some(coefficient_forming_pipeline);
    }

    /// Runs the feed forward algorithm through the dense layer and stores
    /// the output before the activation function in the intermediary buffer
    /// the the output after the activation function in the output buffer
    ///
    /// # Arguments
    ///
    /// * `device` - reference to the wgpu device to run shaders
    /// * `queue` - reference to the wgpu queue to submit commands to the gpu
    pub fn feed_forward(&self, device: &Device, queue: &Queue) {
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
            let dispatch_size = compute_workgroup_size(self.num_nodes as u32, WORK_GROUP_SIZE);

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Dense Layer Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.feed_forward_pipeline);

            // Set the bind group
            compute_pass.set_bind_group(0, &self.input_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.feed_forward_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        let after_int = read_buffer(
            &self.intermediary_buffer,
            self.num_nodes * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        let after_out = read_buffer(
            &self.output_buffer,
            self.num_nodes * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        encoder.insert_debug_marker("Sync Point: Input Pipeline Finished");
        device.poll(Maintain::Wait);

        queue.submit(Some(encoder.finish()));

        print_buffer(&before, device, "Dense Layer Before");
        print_buffer(&after_int, device, "Dense Layer Inter");
        print_buffer(&after_out, device, "Dense Layer Output");
    }

    /// Generates the frobenius norm of the weight matrix
    /// and stores it in the GPU buffer
    ///
    /// # Arguments
    ///
    /// * `queue` - wgpu queue reference to write to the frobenius norm buffer
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
            self.num_nodes * self.num_inputs * std::mem::size_of::<f32>() as u64,
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
                .map(|weight| f32::powi(f32::abs(*weight), 2))
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
            self.num_nodes * self.num_inputs * std::mem::size_of::<f32>() as u64,
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

    /// Generates the regularization for the layer with the
    /// chosen regularization function and store the result
    /// in the gpu buffer to be used for back propogation
    ///
    /// # Arguments
    ///
    /// * `regularization` - A regularization container that contains the regularization function and the hyper paramter
    /// * `device` - a reference to wgpu device to send commands to
    /// * `queue` - a reference to wgpu queue to send command with
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

        // Compute the gradient coefficient first
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Dense Layer Gradient Coefficient Command Encoder"),
        });

        {
            let dispatch_size = compute_workgroup_size(self.num_nodes as u32, WORK_GROUP_SIZE);

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Dense Layer Gradient Coefficient Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(self.coefficient_forming_pipeline.as_ref().unwrap());

            // Set the Bind Groups
            compute_pass.set_bind_group(0, self.next_layer_bind_group.as_ref().unwrap(), &[]);
            compute_pass.set_bind_group(1, &self.coefficient_forming_bind_group, &[]);

            // Dispatch the work groups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder
            .insert_debug_marker("Sync Point: Dense Layer Gradient Coefficient Pipeline Finished");
        device.poll(Maintain::Wait);

        queue.submit(Some(encoder.finish()));

        // The main Back propogation pass
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Dense Layer Back Propogation Command Encoder"),
        });

        {
            let (dispatch_width, dispatch_height) = compute_2d_workgroup_size(
                (self.num_inputs as u32, self.num_nodes as u32),
                (D2_WORK_GROUP_SIZE, D2_WORK_GROUP_SIZE),
            );

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Dense Layer Back Propogation Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(self.regularization_pipeline.as_ref().unwrap());

            // Set the bind groups
            compute_pass.set_bind_group(0, &self.input_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.back_propogation_bind_group, &[]);
            compute_pass.set_bind_group(2, self.next_layer_bind_group.as_ref().unwrap(), &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        encoder.insert_debug_marker("Sync Point: Dense Layer Back Propogation Pipeline Finished");
        device.poll(Maintain::Wait);

        let gradient = read_buffer(
            &self.gradient_buffer,
            self.num_inputs * self.num_nodes * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        let total = self.num_inputs * self.num_nodes;

        let output_string = format!(
            "Dense Layer Gradient Buffer: inputs: {} nodes: {} total: {}",
            self.num_inputs, self.num_nodes, total
        );

        print_buffer(&gradient, device, &output_string);
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
            "Dense Layer Gradient Descent Bind Group",
            (0, learning_rate_buffer, Bbt::Uniform),
            (1, &self.gradient_buffer, Bbt::Storage { read_only: true }),
            (2, &self.weights_buffer, Bbt::Storage { read_only: false }),
            (3, &self.dimensions_buffer, Bbt::Uniform)
        );

        let gradient_descent_pipeline = {
            let shader = device.create_shader_module(include_wgsl!(
                "../shaders/dense_layer/dense_layer_gradient_descent.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Dense Layer Gradient Descent Compute Pipeline Layout"),
                bind_group_layouts: &[&gradient_descent_bind_group_layout],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Dense Layer Gradient Descent Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("dense_layer_gradient_descent_main"),
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
            label: Some("Dense Layer Gradient Descent Command Encoder"),
        });

        let before = read_buffer(
            &self.weights_buffer,
            self.num_inputs * self.num_nodes * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        // Run the gradient descent pass
        {
            let (dispatch_width, dispatch_height) = compute_2d_workgroup_size(
                (self.num_inputs as u32, self.num_nodes as u32),
                (D2_WORK_GROUP_SIZE, D2_WORK_GROUP_SIZE),
            );

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Dense Layer Gradient Descent Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(self.gradient_descent_pipeline.as_ref().unwrap());

            // Set the bind groups
            compute_pass.set_bind_group(0, self.gradient_descent_bind_group.as_ref().unwrap(), &[]);

            // Dispatch the work groups
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        encoder.insert_debug_marker("Sync Point: Dense Layer Gradient Descent Pipeline Finished");
        device.poll(Maintain::Wait);

        let weights = read_buffer(
            &self.weights_buffer,
            self.num_inputs * self.num_nodes * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        print_buffer(&before, device, "Dense Layer Old Weights Buffer");
        print_buffer(&weights, device, "Dense Layer New Weights Buffer");
    }
}

impl FeedForwardLayer for DenseLayer {
    fn get_output_buffer(&self) -> Rc<Buffer> {
        self.output_buffer.clone()
    }
}

impl BackPropogationLayer for DenseLayer {
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
