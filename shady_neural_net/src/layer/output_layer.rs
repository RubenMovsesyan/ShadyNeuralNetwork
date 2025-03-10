use std::rc::Rc;

use crate::utils::{get_buffer, print_buffer, read_buffer};

use super::{ConnectingBindGroup, WORK_GROUP_SIZE, bias::Bias, compute_workgroup_size};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue,
    ShaderStages, include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct OutputLayer {
    num_inputs: u64,
    num_outputs: u64,

    // Buffers associated in feed forward computation
    weights_buffer: Buffer,
    bias_buffer: Buffer,
    intermediary_buffer: Buffer,
    buffer: Buffer,

    // Cost function buffer
    loss_function_buffer: Buffer,
    expected_values_buffer: Buffer,

    // buffers used in back propogation
    frobenius_norm_buffer: Buffer,

    // Buffer for reading the output values
    read_buffer: Buffer,
    loss_read_buffer: Buffer,

    // Bind group information
    input_buffer: Rc<Buffer>,
    input_bind_group_layout: Rc<BindGroupLayout>,
    input_bind_group: Rc<BindGroup>,

    // Main bind group information
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,

    // Cost function bind group information
    loss_function_bind_group_layout: BindGroupLayout,
    loss_function_bind_group: BindGroup,

    // Back Propogation bind groups
    back_propogation_bind_group_layout: BindGroupLayout,
    back_propogation_bind_group: BindGroup,

    // GPU Pipeline Information
    feed_forward_pipeline: ComputePipeline,
    loss_function_pipeline: ComputePipeline,
}

impl OutputLayer {
    /// Initialize a new output layer with random weights and biases
    ///
    /// # Arguments
    ///
    /// * `input_connecting_bind_group` - Bind group reference from the previous layer
    /// * `num_outputs` - number out outputs in this layer
    /// * `device` - a reference to wgpu device to create necessary buffers
    ///
    /// # Returs
    ///
    /// A new instance of `OutputLayer`
    pub fn new(
        input_connecting_bind_group: &ConnectingBindGroup,
        num_outputs: u64,
        device: &Device,
    ) -> Self {
        // Create the main feed forward bind group and buffer information for this layer
        let (
            bind_group_layout,
            bind_group,
            weights_buffer,
            bias_buffer,
            intermediary_buffer,
            buffer,
            read_buffer,
        ) = {
            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Output Layer Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        // Output Buffer
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        // Dimensions Buffer
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        // Weights Buffer
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        // Bias Buffer
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        // Intermediary Buffer
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

            let read_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Copy Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            });

            let dimensions_buffer = {
                let mut dimensions = Vec::new();
                dimensions.push(input_connecting_bind_group.num_inputs as u32);
                dimensions.push(num_outputs as u32);

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Output Layer Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&dimensions),
                    usage: BufferUsages::UNIFORM,
                })
            };

            let weights_buffer = {
                let mut weights: Vec<f32> = Vec::new();
                for _ in 0..input_connecting_bind_group.num_inputs {
                    for _ in 0..num_outputs {
                        weights.push(rand::random_range(-1.0..=1.0));
                    }
                }

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Output Layer Weights Buffer"),
                    contents: bytemuck::cast_slice(&weights),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                })
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
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Output Layer Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: dimensions_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: weights_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: bias_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: intermediary_buffer.as_entire_binding(),
                    },
                ],
            });

            (
                bind_group_layout,
                bind_group,
                weights_buffer,
                bias_buffer,
                intermediary_buffer,
                buffer,
                read_buffer,
            )
        };

        // Create the cost funciton bind group and buffer information
        let (
            loss_function_bind_group_layout,
            loss_function_bind_group,
            loss_function_buffer,
            expected_values_buffer,
            loss_read_buffer,
        ) = {
            let loss_function_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Output Layer Cost Function Bind Group Layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let loss_function_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Cost Funciton Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            });

            let expected_values_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Expected Values Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });

            let loss_read_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Cost Function Read Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            });

            let loss_function_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Output Layer Cost Function Bind Group"),
                layout: &loss_function_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: loss_function_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: expected_values_buffer.as_entire_binding(),
                    },
                ],
            });

            (
                loss_function_bind_group_layout,
                loss_function_bind_group,
                loss_function_buffer,
                expected_values_buffer,
                loss_read_buffer,
            )
        };

        // Create the bind group and buffers for back propogation
        let (
            back_propogation_bind_group_layout,
            back_propogation_bind_group,
            frobenius_norm_buffer,
        ) = {
            let back_propogation_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Output Layer Back Propogation Bind Group Layout"),
                    entries: &[BindGroupLayoutEntry {
                        // Frobenius norm buffer
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            let frobenius_norm_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Frobenius Norm Buffer"),
                mapped_at_creation: false,
                size: std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

            let back_propgation_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Output Layer Back Propogation Bind Group"),
                layout: &back_propogation_bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: frobenius_norm_buffer.as_entire_binding(),
                }],
            });

            (
                back_propogation_bind_group_layout,
                back_propgation_bind_group,
                frobenius_norm_buffer,
            )
        };

        // This is the main pipeline that is used when feeding information forward
        // through the neural network. This pipeline will not effect any of the
        // weights or biases that are created in this layer
        let feed_forward_pipeline = {
            let shader = device.create_shader_module(include_wgsl!(
                "../shaders/output_layer/output_layer_feed_forward.wgsl"
            ));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Output Layer Feed Forward Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &input_connecting_bind_group.bind_group_layout,
                    &bind_group_layout,
                ],
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
                bind_group_layouts: &[
                    &input_connecting_bind_group.bind_group_layout,
                    &bind_group_layout,
                    &loss_function_bind_group_layout,
                ],
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

        Self {
            num_inputs: input_connecting_bind_group.num_inputs,
            num_outputs,
            // -------------------------------
            weights_buffer,
            bias_buffer,
            intermediary_buffer,
            buffer,
            // -------------------------------
            loss_function_buffer,
            expected_values_buffer,
            // -------------------------------
            frobenius_norm_buffer,
            // -------------------------------
            read_buffer,
            loss_read_buffer,
            // -------------------------------
            input_buffer: input_connecting_bind_group.buffer.clone(),
            input_bind_group_layout: input_connecting_bind_group.bind_group_layout.clone(),
            input_bind_group: input_connecting_bind_group.bind_group.clone(),
            // -------------------------------
            bind_group_layout,
            bind_group,
            // -------------------------------
            loss_function_bind_group_layout,
            loss_function_bind_group,
            // -------------------------------
            back_propogation_bind_group_layout,
            back_propogation_bind_group,
            // -------------------------------
            feed_forward_pipeline,
            loss_function_pipeline,
        }
    }

    pub fn recieve(&self, device: &Device, queue: &Queue) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Input Layer Command Encoder"),
        });

        let before = read_buffer(
            &self.input_buffer,
            self.num_inputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        let weights = read_buffer(
            &self.weights_buffer,
            self.num_inputs * self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );
        let biases = read_buffer(
            &self.bias_buffer,
            self.num_outputs * std::mem::size_of::<f32>() as u64 * 2,
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

            // Set the bind group
            compute_pass.set_bind_group(0, self.input_bind_group.as_ref(), &[]);
            compute_pass.set_bind_group(1, &self.bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        let intermediary = read_buffer(
            &self.intermediary_buffer,
            self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        let after = read_buffer(
            &self.buffer,
            self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        encoder.insert_debug_marker("Sync Point: Ouptut Feed Forward Pipeline Finished");
        device.poll(Maintain::Wait);

        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &self.read_buffer,
            0,
            self.num_outputs * std::mem::size_of::<f32>() as u64,
        );

        queue.submit(Some(encoder.finish()));

        print_buffer(&weights, device, "Output layer weights");
        print_buffer(&biases, device, "Output layer biases");
        print_buffer(&before, device, "Output layer inputs");
        print_buffer(&intermediary, device, "Output layer intermediary");
        print_buffer(&after, device, "Output layer outputs");
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
            compute_pass.set_bind_group(0, self.input_bind_group.as_ref(), &[]);
            compute_pass.set_bind_group(1, &self.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.loss_function_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder.insert_debug_marker("Sync Point: Ouptut Cost Function Pipeline Finished");
        device.poll(Maintain::Wait);

        encoder.copy_buffer_to_buffer(
            &self.loss_function_buffer,
            0,
            &self.loss_read_buffer,
            0,
            self.num_outputs * std::mem::size_of::<f32>() as u64,
        );

        queue.submit(Some(encoder.finish()));

        // Get the average of all the loss values
        let loss_vector = self.get_loss(device);

        println!("{:#?}", loss_vector);
        loss_vector.iter().sum::<f32>() / loss_vector.len() as f32
    }

    /// Gets a vector of the output from the previous feed forward call
    ///
    /// # Arguments
    ///
    /// * `device` - reference to wgpu device to read output buffer
    ///
    /// # Returns
    /// `Vec<f32>` of the computed output values    
    pub fn get_output(&self, device: &Device) -> Vec<f32> {
        let slice = self.read_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(Maintain::Wait);

        let data = slice.get_mapped_range();

        let new_slice: &[f32] = bytemuck::cast_slice(&data);

        new_slice.to_vec()
    }

    /// Reads the loss buffer from the last `compute_cost` call
    ///
    /// # Arguments
    ///
    /// * `device` - refernce to the wgpu device to read the buffer
    ///
    /// # Returns
    /// `Vec<f32>` of the computed loss values
    pub fn get_loss(&self, device: &Device) -> Vec<f32> {
        let slice = self.loss_read_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(Maintain::Wait);

        let data = slice.get_mapped_range();

        let new_slice: &[f32] = bytemuck::cast_slice(&data);

        new_slice.to_vec()
    }

    /// Generates the frobenius norm of the weight matrix
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
}
