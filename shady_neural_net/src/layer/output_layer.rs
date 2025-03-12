use std::rc::Rc;

use crate::layer::compute_2d_workgroup_size;
use crate::layer_structs::regularization::*;
use crate::utils::{get_buffer, read_buffer};

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

/// Creates the input bind group connecting the outputs
/// from the previous layer to the current layer
///
/// # Arguments
///
/// * `device` - wgpu device reference to create bind group
/// * `input_buffer` - buffer of inputs to the current layer
///
/// # Returns
///
/// `(BindGroupLayout, BindGroup)` tuple with the input bind group layout and the bind group
fn create_input_bind_group(
    device: &Device,
    input_buffer: Rc<Buffer>,
) -> (BindGroupLayout, BindGroup) {
    let input_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Dense Layer Input Bind Group Layout"),
        entries: &[BindGroupLayoutEntry {
            // Outputs from previous layer
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let input_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Dense Layer Input Bind Group"),
        layout: &input_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: input_buffer.as_entire_binding(),
        }],
    });

    (input_bind_group_layout, input_bind_group)
}

/// Creates the bind group to be used in the feed forward stage of this layer
///
/// # Arguments
///
/// * `device` - reference to wgpu refernce to create bind group
/// * `dimensions_buffer` - buffer containing the dimensions of the weight buffer
/// * `weights_buffer` - buffer containing the weights in this layer
/// * `bias_buffer` - buffer containing the biases in this layer
/// * `intermediary_buffer` - buffer for storing values before computing the softmax
/// * `output_buffer` - buffer for storing values after computing the softmax
///
/// # Returns
///
/// `(BindGroupLayout, BindGroup)` tuple with the feed forward bind group layout and bind group
fn create_feed_forward_bind_group(
    device: &Device,
    dimensions_buffer: &Buffer,
    weights_buffer: &Buffer,
    bias_buffer: &Buffer,
    intermediary_buffer: &Buffer,
    output_buffer: &Buffer,
) -> (BindGroupLayout, BindGroup) {
    let feed_forward_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Output Layer Feed Forward Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    // Dimensions Buffer
                    binding: 0,
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
                    // Bias Buffer
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
                    // Intermediary Buffer
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    // Output Buffer
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

    let feed_forward_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Output Layer Feed Forward Bind Group"),
        layout: &feed_forward_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: dimensions_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: weights_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: bias_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: intermediary_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    (feed_forward_bind_group_layout, feed_forward_bind_group)
}

/// Creates the bind group to be used in the loss stage of this layer
///
/// # Arguments
///
/// * `device` - reference to wgpu refernce to create bind group
/// * `output_buffer` - reference to the output buffer of this layer
/// * `expected_values_buffer` - reference to the expected values buffer
///
/// # Returns
///
/// `(BindGroupLayout, BindGroup)` tuple with the loss function bind group layout and bind group
fn create_loss_fuction_bind_group(
    device: &Device,
    dimensions_buffer: &Buffer,
    output_buffer: &Buffer,
    expected_values_buffer: &Buffer,
    loss_function_buffer: &Buffer,
) -> (BindGroupLayout, BindGroup) {
    let loss_function_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Ouptut Layer Loss Function Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    // Dimensions Buffer
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    // Output Buffer
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
                    // Expected Values Buffer
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
                    // Loss Function Buffer
                    binding: 3,
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

    let loss_function_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Output Layer Loss Function Bind Group"),
        layout: &loss_function_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: dimensions_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: expected_values_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: loss_function_buffer.as_entire_binding(),
            },
        ],
    });

    (loss_function_bind_group_layout, loss_function_bind_group)
}

/// Creates the bind group to be used in the loss stage of this layer
///
/// # Arguments
///
/// * `device` - reference to wgpu refernce to create bind group
/// * `l_1_norm_buffer` - buffer where the L1 norm is stored
/// * `frobenius_norm_buffer` - buffer where the frobenius norm is stored
/// * `regularization_info_buffer` - buffer where the regularization info is stored
/// * `regularization_output_buffer` - buffer where the output of the regularization derivative goes
///
/// # Returns
///
/// `(BindGroupLayout, BindGroup)` tuple with the back propogation bind group layout and bind group
fn create_back_propogation_bind_group(
    device: &Device,
    l_1_norm_buffer: &Buffer,
    frobenius_norm_buffer: &Buffer,
    regularization_info_buffer: &Buffer,
    regularization_output_buffer: &Buffer,
    dimensions_buffer: &Buffer,
    weights_buffer: &Buffer,
) -> (BindGroupLayout, BindGroup) {
    let back_propogation_bind_group_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Output Layer Back Propogation Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    // L1 Norm Buffer
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    // Frobenius Norm buffer
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
                    // Regularization Info Buffer
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    // Regularization Output Buffer
                    binding: 3,
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
                    binding: 4,
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
                    binding: 5,
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

    let back_propogation_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Output Layer Back Propogation Bind Group"),
        layout: &back_propogation_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: l_1_norm_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: frobenius_norm_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: regularization_info_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: regularization_output_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: dimensions_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: weights_buffer.as_entire_binding(),
            },
        ],
    });

    (
        back_propogation_bind_group_layout,
        back_propogation_bind_group,
    )
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct OutputLayer {
    pub num_inputs: u64,
    pub num_outputs: u64,

    // Buffers associated in feed forward computation
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

    // GPU Pipeline Information
    feed_forward_pipeline: ComputePipeline,
    loss_function_pipeline: ComputePipeline,
    regularization_pipeline: ComputePipeline,
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
        let (input_bind_group_layout, input_bind_group) =
            create_input_bind_group(device, feed_forward_input.buffer.clone());

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
            loss_function_buffer,
            expected_values_buffer,
        ) = {
            let dimensions_buffer = {
                let mut dimensions = Vec::new();
                dimensions.push(feed_forward_input.num_inputs as u32);
                dimensions.push(num_outputs as u32);

                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Output Layer Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&dimensions),
                    usage: BufferUsages::UNIFORM,
                })
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
                loss_function_buffer,
                expected_values_buffer,
            )
        };

        let (feed_forward_bind_group_layout, feed_forward_bind_group) =
            create_feed_forward_bind_group(
                device,
                &dimensions_buffer,
                &weights_buffer,
                &bias_buffer,
                &intermediary_buffer,
                &output_buffer,
            );

        let (loss_function_bind_group_layout, loss_function_bind_group) =
            create_loss_fuction_bind_group(
                device,
                &dimensions_buffer,
                &output_buffer,
                &expected_values_buffer,
                &loss_function_buffer,
            );

        let (back_propogation_bind_group_layout, back_propogation_bind_group) =
            create_back_propogation_bind_group(
                device,
                &l_1_norm_buffer,
                &frobenius_norm_buffer,
                &regularization_info_buffer,
                &regularization_output_buffer,
                &dimensions_buffer,
                &weights_buffer,
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
                bind_group_layouts: &[&back_propogation_bind_group_layout],
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
            feed_forward_pipeline,
            loss_function_pipeline,
            regularization_pipeline,
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
    pub fn generate_regularization_function(
        &self,
        regularization: Regularization,
        device: &Device,
        queue: &Queue,
    ) -> Vec<f32> {
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

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        encoder.insert_debug_marker("Sync Point: Output Regularization Pipeline Finished");
        device.poll(Maintain::Wait);

        let value = read_buffer(
            &self.regularization_output_buffer,
            self.num_inputs * self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        get_buffer(&value, device)
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
    fn get_connecting_weight_buffer(&self) -> Rc<Buffer> {
        self.weights_buffer.clone()
    }
}
