use std::{error::Error, fmt::Display, rc::Rc};

#[allow(unused_imports)]
use log::*;

use bias::Bias;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue,
    ShaderModule, ShaderStages, include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

const WORK_GROUP_SIZE: u32 = 256;

// Helper functions
fn compute_workgroup_size(nodes: u32, work_group_size: u32) -> u32 {
    (nodes + work_group_size - 1) / work_group_size
}

// Error Structs
#[derive(Debug)]
pub struct InputLengthMismatchError;

impl Error for InputLengthMismatchError {}

impl Display for InputLengthMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input Length is not the same as the buffer")
    }
}

// Module for inner structures in the Neural Network
use crate::layer_structs::*;

// Trait for each layer to get the connecting buffer
pub trait Layer {
    fn get_connecting_bind_group(&self) -> Rc<BindGroup>;

    fn get_connecting_bind_group_layout(&self) -> Rc<BindGroupLayout>;

    fn get_connecting_buffer(&self) -> Rc<Buffer>;
}

pub struct ConnectingBindGroup {
    pub bind_group_layout: Rc<BindGroupLayout>,
    pub bind_group: Rc<BindGroup>,
    pub buffer: Rc<Buffer>,
    pub buffer_len: u64,
}

#[derive(Debug)]
pub enum NeuralNetLayer {
    Input(InputLayer),
    Dense(DenseLayer),
    Output(OutputLayer),
}

impl NeuralNetLayer {
    pub fn get_connecting_bind_group(&self) -> Option<ConnectingBindGroup> {
        use NeuralNetLayer::*;
        match self {
            Input(input_layer) => Some(ConnectingBindGroup {
                bind_group_layout: input_layer.get_connecting_bind_group_layout(),
                bind_group: input_layer.get_connecting_bind_group(),
                buffer: input_layer.get_connecting_buffer(),
                buffer_len: input_layer.num_inputs,
            }),
            Dense(dense_layer) => Some(ConnectingBindGroup {
                bind_group_layout: dense_layer.get_connecting_bind_group_layout(),
                bind_group: dense_layer.get_connecting_bind_group(),
                buffer: dense_layer.get_connecting_buffer(),
                buffer_len: dense_layer.num_nodes,
            }),
            Output(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
/// Input Layer struct used in neural net layer
#[derive(Debug)]
pub struct InputLayer {
    num_inputs: u64,
    buffer: Rc<Buffer>,

    bind_group: Rc<BindGroup>,
    bind_group_layout: Rc<BindGroupLayout>,
    pipeline: ComputePipeline,
}

impl InputLayer {
    pub fn new(num_inputs: u64, device: &Device) -> Self {
        // Create the buffer from the input data
        let bind_group_layout =
            Rc::new(device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Input Layer Bind Group Layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }));

        let buffer = Rc::new(device.create_buffer(&BufferDescriptor {
            label: Some("Input Layer Buffer"),
            mapped_at_creation: false,
            size: num_inputs * std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        }));

        let bind_group = Rc::new(device.create_bind_group(&BindGroupDescriptor {
            label: Some("Input Layer Bind Group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        }));

        let shader = device.create_shader_module(include_wgsl!("./shaders/input_layer.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Input Layer Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Input Layer Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("input_layer_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            num_inputs,
            buffer,
            bind_group,
            bind_group_layout,
            pipeline,
        }
    }

    pub fn set_inputs(
        &self,
        inputs: Vec<f32>,
        device: &Device,
        queue: &Queue,
    ) -> Result<(), InputLengthMismatchError> {
        if inputs.len() != self.num_inputs as usize {
            return Err(InputLengthMismatchError);
        }

        queue.write_buffer(self.buffer.as_ref(), 0, bytemuck::cast_slice(&inputs));

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Input Layer Command Encoder"),
        });

        // Run the pipeline
        {
            let dispatch_size = compute_workgroup_size(self.num_inputs as u32, WORK_GROUP_SIZE);

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Input Layer Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.pipeline);

            compute_pass.set_bind_group(0, self.bind_group.as_ref(), &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder.insert_debug_marker("Sync Point: Input Pipeline Finished");
        queue.submit(Some(encoder.finish()));
        device.poll(Maintain::Wait);

        Ok(())
    }
}

impl Layer for InputLayer {
    fn get_connecting_bind_group(&self) -> Rc<BindGroup> {
        self.bind_group.clone()
    }

    fn get_connecting_bind_group_layout(&self) -> Rc<BindGroupLayout> {
        self.bind_group_layout.clone()
    }

    fn get_connecting_buffer(&self) -> Rc<Buffer> {
        self.buffer.clone()
    }
}

// ---------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------
/// Dense layer struct used in neural net
#[allow(dead_code)]
#[derive(Debug)]
pub struct DenseLayer {
    num_nodes: u64,
    num_inputs: u64,

    weights_buffer: Buffer,
    bias_buffer: Buffer,
    output_buffer: Rc<Buffer>,

    // Bind group information
    input_buffer: Rc<Buffer>,
    input_bind_group_layout: Rc<BindGroupLayout>,
    input_bind_group: Rc<BindGroup>,

    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,

    output_bind_group_layout: Rc<BindGroupLayout>,
    output_bind_group: Rc<BindGroup>,

    // GPU pipeline information
    pipeline: ComputePipeline,
}

impl DenseLayer {
    pub fn new(
        input_connecting_bind_group: &ConnectingBindGroup,
        num_nodes: u64,
        device: &Device,
    ) -> Self {
        let (
            bind_group_layout,
            bind_group,
            output_bind_group_layout,
            output_bind_group,
            weights_buffer,
            bias_buffer,
            output_buffer,
        ) = {
            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Dense Layer Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        // Weights buffer
                        binding: 0,
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
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        // Dimensions
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            // Initialize the weights matrix buffer with random values from -1.0 to 1.0
            // containts a matrix with num_nodes sets of weights
            // each with num_inputs weights in them
            let weights_buffer = {
                let mut weights = Vec::new();

                for _ in 0..input_connecting_bind_group.buffer_len {
                    for _ in 0..num_nodes {
                        weights.push(rand::random_range(-1.0..=1.0));
                        // weights.push(1.0);
                    }
                }

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Dense Layer Weights Buffer"),
                    contents: bytemuck::cast_slice(&weights),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                })
            };

            // Initialize the bias vector buffer with random values from -1.0 to 1.0
            // each Bias is a bias value and a bias weight
            let bias_buffer = {
                let mut biases = Vec::new();
                for _ in 0..num_nodes {
                    biases.push(Bias::new(
                        rand::random_range(-1.0..=1.0),
                        rand::random_range(-1.0..=1.0),
                        // 0.0, 0.0,
                    ));
                }

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Dense Layer Bias Buffer"),
                    contents: bytemuck::cast_slice(&biases),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                })
            };

            let dimensions_buffer = {
                let dimensions = vec![num_nodes, input_connecting_bind_group.buffer_len];

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Dense Layer Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&dimensions),
                    usage: BufferUsages::UNIFORM,
                })
            };

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Dense Layer Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: weights_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: bias_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: dimensions_buffer.as_entire_binding(),
                    },
                ],
            });

            let output_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Dense Layer Output Bind Group Layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            let output_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Dense Layer Output Buffer"),
                mapped_at_creation: false,
                size: num_nodes * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            });

            let output_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Dense Layer Output Bind Group"),
                layout: &output_bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                }],
            });

            (
                bind_group_layout,
                bind_group,
                output_bind_group_layout,
                output_bind_group,
                weights_buffer,
                bias_buffer,
                output_buffer,
            )
        };

        // Create the pipeline from the bind group layout
        let pipeline = {
            let shader: ShaderModule =
                device.create_shader_module(include_wgsl!("./shaders/dense_layer.wgsl"));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Dense Layer Compute Pipeline Layout"),
                bind_group_layouts: &[
                    input_connecting_bind_group.bind_group_layout.as_ref(),
                    &bind_group_layout,
                    &output_bind_group_layout,
                ],
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
            num_inputs: input_connecting_bind_group.buffer_len,
            num_nodes,
            weights_buffer,
            bias_buffer,
            input_buffer: input_connecting_bind_group.buffer.clone(),
            output_buffer: Rc::new(output_buffer),
            input_bind_group_layout: input_connecting_bind_group.bind_group_layout.clone(),
            input_bind_group: input_connecting_bind_group.bind_group.clone(),
            bind_group_layout,
            bind_group,
            output_bind_group_layout: Rc::new(output_bind_group_layout),
            output_bind_group: Rc::new(output_bind_group),
            pipeline,
        }
    }

    pub fn feed_forward(&self, device: &Device, queue: &Queue) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Input Layer Command Encoder"),
        });

        // Run the pipeline
        {
            let dispatch_size = compute_workgroup_size(self.num_nodes as u32, WORK_GROUP_SIZE);

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Dense Layer Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.pipeline);

            // Set the bind group
            compute_pass.set_bind_group(0, self.input_bind_group.as_ref(), &[]);
            compute_pass.set_bind_group(1, &self.bind_group, &[]);
            compute_pass.set_bind_group(2, self.output_bind_group.as_ref(), &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder.insert_debug_marker("Sync Point: Input Pipeline Finished");
        device.poll(Maintain::Wait);

        queue.submit(Some(encoder.finish()));
    }
}

impl Layer for DenseLayer {
    fn get_connecting_bind_group(&self) -> Rc<BindGroup> {
        self.output_bind_group.clone()
    }

    fn get_connecting_bind_group_layout(&self) -> Rc<BindGroupLayout> {
        self.output_bind_group_layout.clone()
    }

    fn get_connecting_buffer(&self) -> Rc<Buffer> {
        self.output_buffer.clone()
    }
}

// ---------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------
#[allow(dead_code)]
#[derive(Debug)]
pub struct OutputLayer {
    num_inputs: u64,
    num_outputs: u64,

    buffer: Buffer,
    read_buffer: Buffer,

    // Bind group information
    input_buffer: Rc<Buffer>,
    input_bind_group_layout: Rc<BindGroupLayout>,
    input_bind_group: Rc<BindGroup>,

    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,

    // GPU Pipeline Information
    pipeline: ComputePipeline,
}

impl OutputLayer {
    pub fn new(
        input_connecting_bind_group: &ConnectingBindGroup,
        num_outputs: u64,
        device: &Device,
    ) -> Self {
        let (bind_group_layout, bind_group, buffer, read_buffer) = {
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
                dimensions.push(input_connecting_bind_group.buffer_len as u32);
                dimensions.push(num_outputs as u32);

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Output Layer Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&dimensions),
                    usage: BufferUsages::UNIFORM,
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
                ],
            });

            (bind_group_layout, bind_group, buffer, read_buffer)
        };

        // Create the pipeline from the bind group layout
        let pipeline = {
            let shader: ShaderModule =
                device.create_shader_module(include_wgsl!("./shaders/output_layer.wgsl"));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Output Layer Compute Pipeline Layout"),
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

        Self {
            num_inputs: input_connecting_bind_group.buffer_len,
            num_outputs,
            input_buffer: input_connecting_bind_group.buffer.clone(),
            input_bind_group_layout: input_connecting_bind_group.bind_group_layout.clone(),
            input_bind_group: input_connecting_bind_group.bind_group.clone(),
            buffer,
            read_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
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
                label: Some("Output Layer Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.pipeline);

            // Set the bind group
            compute_pass.set_bind_group(0, self.input_bind_group.as_ref(), &[]);
            compute_pass.set_bind_group(1, &self.bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder.insert_debug_marker("Sync Point: Input Pipeline Finished");
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

    pub fn get_data(&self, device: &Device) -> Vec<f32> {
        let slice = self.read_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(Maintain::Wait);

        let data = slice.get_mapped_range();

        let new_slice: &[f32] = bytemuck::cast_slice(&data);

        new_slice.to_vec()
    }
}
