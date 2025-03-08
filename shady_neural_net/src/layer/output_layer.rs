use std::rc::Rc;

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

    weights_buffer: Buffer,
    bias_buffer: Buffer,
    buffer: Buffer,

    // Cost function buffer
    loss_function_buffer: Buffer,
    expected_values_buffer: Buffer,

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

    // GPU Pipeline Information
    feed_forward_pipeline: ComputePipeline,
    loss_function_pipeline: ComputePipeline,
}

impl OutputLayer {
    pub fn new(
        input_connecting_bind_group: &ConnectingBindGroup,
        num_outputs: u64,
        device: &Device,
    ) -> Self {
        // Create the main feed forward bind group and buffer information for this layer
        let (bind_group_layout, bind_group, weights_buffer, bias_buffer, buffer, read_buffer) = {
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
                ],
            });

            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Output Layer Buffer"),
                mapped_at_creation: false,
                size: num_outputs * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
                let mut weights = Vec::new();
                for _ in 0..input_connecting_bind_group.num_inputs {
                    for _ in 0..num_outputs {
                        weights.push(rand::random_range(-1.0..=1.0));
                    }
                }

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Output Layer Weights Buffer"),
                    contents: bytemuck::cast_slice(&weights),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                })
            };

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
                ],
            });

            (
                bind_group_layout,
                bind_group,
                weights_buffer,
                bias_buffer,
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
                label: Some("Output Layer Compute Pipeline"),
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
            input_buffer: input_connecting_bind_group.buffer.clone(),
            input_bind_group_layout: input_connecting_bind_group.bind_group_layout.clone(),
            input_bind_group: input_connecting_bind_group.bind_group.clone(),
            weights_buffer,
            bias_buffer,
            buffer,
            read_buffer,
            loss_read_buffer,
            expected_values_buffer,
            loss_function_bind_group_layout,
            loss_function_bind_group,
            loss_function_buffer,
            bind_group_layout,
            bind_group,
            feed_forward_pipeline,
            loss_function_pipeline,
        }
    }

    pub fn recieve(&self, device: &Device, queue: &Queue) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Input Layer Command Encoder"),
        });

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
    }

    pub fn compute_cost(&self, expected_values: &[f32], device: &Device, queue: &Queue) -> f32 {
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

    /// Gets a Vec<f32> of the output that was computed
    /// from the last feed forward Call of the layer
    pub fn get_output(&self, device: &Device) -> Vec<f32> {
        let slice = self.read_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(Maintain::Wait);

        let data = slice.get_mapped_range();

        let new_slice: &[f32] = bytemuck::cast_slice(&data);

        new_slice.to_vec()
    }

    /// Gets a Vec<f32> of the result of the cost function
    /// that was computed from the last cost function call
    /// of the layer
    pub fn get_loss(&self, device: &Device) -> Vec<f32> {
        let slice = self.loss_read_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(Maintain::Wait);

        let data = slice.get_mapped_range();

        let new_slice: &[f32] = bytemuck::cast_slice(&data);

        new_slice.to_vec()
    }
}
