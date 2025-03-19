use std::rc::Rc;

use crate::create_buffer_bind_group;
use crate::layer::compute_2d_workgroup_size;
use crate::layer_structs::loss::*;
use crate::layer_structs::regularization::*;
use crate::utils::{get_buffer, read_buffer};

use super::weight_distribution::WeightDistribution;
use super::{
    BackPropogationLayer, BackPropogationLayerConnection, D2_WORK_GROUP_SIZE, FeedForwardLayer,
};
use super::{FeedForwardConnection, WORK_GROUP_SIZE, bias::Bias, compute_workgroup_size};
use serde::{Deserialize, Serialize};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderStages,
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputLayerDescriptor {
    pub num_inputs: u64,
    pub num_outputs: u64,
    pub weights: Vec<f32>,
    pub biases: Vec<Bias>,
}

fn create_bind_groups(
    device: &Device,
    input_buffer: &Buffer,
    dimensions_buffer: &Buffer,
    weights_buffer: &Buffer,
    bias_buffer: &Buffer,
    intermediary_buffer: &Buffer,
    output_buffer: &Buffer,
    l_1_norm_buffer: &Buffer,
    frobenius_norm_buffer: &Buffer,
    regularization_info_buffer: &Buffer,
    regularization_output_buffer: &Buffer,
    gradient_buffer: &Buffer,
    gradient_coefficient_buffer: &Buffer,
    gradient_back_prop_buffer: &Buffer,
    loss_function_buffer: &Buffer,
    loss_function_info_buffer: &Buffer,
    expected_values_buffer: &Buffer,
    learning_rate_buffer: &Buffer,
) -> (
    (BindGroupLayout, BindGroup),
    (BindGroupLayout, BindGroup),
    (BindGroupLayout, BindGroup),
    (BindGroupLayout, BindGroup),
    (BindGroupLayout, BindGroup),
) {
    let (input_bind_group_layout, input_bind_group) = create_buffer_bind_group!(
        device,
        "Output Layer Input Bind Group",
        (0, input_buffer, Bbt::Storage { read_only: true })
    );

    let (feed_forward_bind_group_layout, feed_forward_bind_group) = create_buffer_bind_group!(
        device,
        "Output Layer Feed Forward Bind Group",
        (0, dimensions_buffer, Bbt::Uniform),
        (1, weights_buffer, Bbt::Storage { read_only: true }),
        (2, bias_buffer, Bbt::Storage { read_only: true }),
        (3, intermediary_buffer, Bbt::Storage { read_only: false }),
        (4, output_buffer, Bbt::Storage { read_only: false })
    );

    let (loss_function_bind_group_layout, loss_function_bind_group) = create_buffer_bind_group!(
        device,
        "Output Layer Loss Function Bind Group",
        (0, dimensions_buffer, Bbt::Uniform),
        (1, output_buffer, Bbt::Storage { read_only: true }),
        (2, expected_values_buffer, Bbt::Storage { read_only: true }),
        (3, loss_function_buffer, Bbt::Storage { read_only: false }),
        (
            4,
            gradient_coefficient_buffer,
            Bbt::Storage { read_only: false }
        ),
        (
            5,
            gradient_back_prop_buffer,
            Bbt::Storage { read_only: false }
        ),
        (6, weights_buffer, Bbt::Storage { read_only: true }),
        (7, loss_function_info_buffer, Bbt::Uniform)
    );

    let (back_propogation_bind_group_layout, back_propogation_bind_group) = create_buffer_bind_group!(
        device,
        "Output Layer Back Propogation Bind Group",
        (0, l_1_norm_buffer, Bbt::Uniform),
        (1, frobenius_norm_buffer, Bbt::Uniform),
        (2, regularization_info_buffer, Bbt::Uniform),
        (
            3,
            regularization_output_buffer,
            Bbt::Storage { read_only: false }
        ),
        (4, dimensions_buffer, Bbt::Uniform),
        (5, weights_buffer, Bbt::Storage { read_only: false }),
        (6, gradient_buffer, Bbt::Storage { read_only: false }),
        (
            7,
            gradient_coefficient_buffer,
            Bbt::Storage { read_only: true }
        )
    );

    let (learning_rate_bind_group_layout, learning_rate_bind_group) = create_buffer_bind_group!(
        device,
        "Output Layer Learning Rate Bind Group",
        (0, learning_rate_buffer, Bbt::Uniform)
    );

    (
        (input_bind_group_layout, input_bind_group),
        (feed_forward_bind_group_layout, feed_forward_bind_group),
        (loss_function_bind_group_layout, loss_function_bind_group),
        (
            back_propogation_bind_group_layout,
            back_propogation_bind_group,
        ),
        (learning_rate_bind_group_layout, learning_rate_bind_group),
    )
}

fn create_buffers(
    device: &Device,
    feed_forward_input: &FeedForwardConnection,
    num_outputs: u64,
) -> (
    Buffer,
    Buffer,
    Buffer,
    Buffer,
    Buffer,
    Buffer,
    Buffer,
    Buffer,
    Buffer,
    Buffer,
    Rc<Buffer>,
    Rc<Buffer>,
    Rc<Buffer>,
) {
    let dimensions_buffer = {
        let mut dimensions = Vec::new();
        dimensions.push(num_outputs as u32);
        dimensions.push(feed_forward_input.num_inputs as u32);

        Rc::new(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Output Layer Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dimensions),
            usage: BufferUsages::UNIFORM,
        }))
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
        size: num_outputs * feed_forward_input.num_inputs * std::mem::size_of::<f32>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });

    let gradient_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Ouput Layer Gradient Buffer"),
        mapped_at_creation: false,
        size: num_outputs * feed_forward_input.num_inputs * std::mem::size_of::<f32>() as u64,
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

    let loss_function_info_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Output Layer Loss Function information Buffer"),
        mapped_at_creation: false,
        size: std::mem::size_of::<LossRepr>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });

    let expected_values_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Output Layer Expected Values Buffer"),
        mapped_at_creation: false,
        size: num_outputs * std::mem::size_of::<f32>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });

    let gradient_back_prop_buffer = Rc::new(device.create_buffer(&BufferDescriptor {
        label: Some("Output Layer Gradient Back Prop Buffer"),
        mapped_at_creation: false,
        size: feed_forward_input.num_inputs * std::mem::size_of::<f32>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    }));

    (
        intermediary_buffer,
        output_buffer,
        l_1_norm_buffer,
        frobenius_norm_buffer,
        regularization_info_buffer,
        regularization_output_buffer,
        gradient_buffer,
        loss_function_buffer,
        loss_function_info_buffer,
        expected_values_buffer,
        gradient_back_prop_buffer,
        dimensions_buffer,
        gradient_coefficient_buffer,
    )
}

fn create_pipelines(
    (
        input_bind_group_layout,
        feed_forward_bind_group_layout,
        loss_function_bind_group_layout,
        back_propogation_bind_group_layout,
        learning_rate_bind_group_layout,
    ): (
        &BindGroupLayout,
        &BindGroupLayout,
        &BindGroupLayout,
        &BindGroupLayout,
        &BindGroupLayout,
    ),
    device: &Device,
) -> (ComputePipeline, ComputePipeline, ComputePipeline) {
    // This is the main pipeline that is used when feeding information forward
    // through the neural network. This pipeline will not effect any of the
    // weights or biases that are created in this layer
    let feed_forward_pipeline = {
        let shader =
            device.create_shader_module(include_wgsl!("../shaders/output_layer/feed_forward.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Output Layer Feed Forward Compute Pipeline Layout"),
            bind_group_layouts: &[input_bind_group_layout, feed_forward_bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Output Layer Feed Forward Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("output_layer_feed_forward_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    // This is the cost function computing pipeline. This will not effect
    // any of the weights or biases in this layer. It is used for computing
    // the cost function associated from the data that is given
    let loss_function_pipeline = {
        let shader =
            device.create_shader_module(include_wgsl!("../shaders/output_layer/loss.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Output Layer Cost Function Compute Pipeline Layout"),
            bind_group_layouts: &[loss_function_bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Output Layer Cost Function Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("output_layer_loss_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let back_propogation_pipeline = {
        let shader = device
            .create_shader_module(include_wgsl!("../shaders/output_layer/back_propogate.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Output Layer Back Propogation Compute Pipeline Layout"),
            bind_group_layouts: &[
                back_propogation_bind_group_layout,
                input_bind_group_layout,
                learning_rate_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Output Layer Back Propogation Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("output_layer_back_propogate_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    (
        feed_forward_pipeline,
        loss_function_pipeline,
        back_propogation_pipeline,
    )
}

#[derive(Debug)]
pub struct OutputLayer {
    pub num_inputs: u64,
    pub num_outputs: u64,

    // Buffers associated in feed forward computation
    weights_buffer: Rc<Buffer>,
    bias_buffer: Buffer,
    output_buffer: Buffer,

    // Cost function buffer
    loss_function_buffer: Buffer,
    loss_function_info_buffer: Buffer,
    expected_values_buffer: Buffer,

    // Buffers used in back propogation
    l_1_norm_buffer: Buffer,
    frobenius_norm_buffer: Buffer,
    regularization_info_buffer: Buffer,
    gradient_back_prop_buffer: Rc<Buffer>,

    // Input Bind group information
    input_bind_group: BindGroup,

    // Main bind group information
    feed_forward_bind_group: BindGroup,

    // Cost function bind group information
    loss_function_bind_group: BindGroup,

    // Back Propogation bind groups
    back_propogation_bind_group: BindGroup,

    // Learning Rate bind groups
    learning_rate_bind_group: BindGroup,

    // GPU Pipeline Information
    feed_forward_pipeline: ComputePipeline,
    loss_function_pipeline: ComputePipeline,
    back_propogation_pipeline: ComputePipeline,
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
        learning_rate_buffer: &Buffer,
        device: &Device,
    ) -> Self {
        // Create all the buffers necessary in this layer
        let (
            intermediary_buffer,
            output_buffer,
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            regularization_output_buffer,
            gradient_buffer,
            loss_function_buffer,
            loss_function_info_buffer,
            expected_values_buffer,
            gradient_back_prop_buffer,
            dimensions_buffer,
            gradient_coefficient_buffer,
        ) = create_buffers(device, feed_forward_input, num_outputs);

        let weights_buffer = {
            let weights = WeightDistribution::Xavier
                .get_weight_distribution(feed_forward_input.num_inputs, num_outputs);

            Rc::new(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Output Layer Weights Buffer"),
                contents: bytemuck::cast_slice(&weights),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }))
        };

        let bias_buffer = {
            let biases = WeightDistribution::Xavier.get_bias_distribution(num_outputs);

            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Output Layer Bias Buffer"),
                contents: bytemuck::cast_slice(&biases),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };

        let (
            (input_bind_group_layout, input_bind_group),
            (feed_forward_bind_group_layout, feed_forward_bind_group),
            (loss_function_bind_group_layout, loss_function_bind_group),
            (back_propogation_bind_group_layout, back_propogation_bind_group),
            (learning_rate_bind_group_layout, learning_rate_bind_group),
        ) = create_bind_groups(
            device,
            &feed_forward_input.buffer,
            &dimensions_buffer,
            &weights_buffer,
            &bias_buffer,
            &intermediary_buffer,
            &output_buffer,
            &l_1_norm_buffer,
            &frobenius_norm_buffer,
            &regularization_info_buffer,
            &regularization_output_buffer,
            &gradient_buffer,
            &gradient_coefficient_buffer,
            &gradient_back_prop_buffer,
            &loss_function_buffer,
            &loss_function_info_buffer,
            &expected_values_buffer,
            learning_rate_buffer,
        );

        let (feed_forward_pipeline, loss_function_pipeline, back_propogation_pipeline) =
            create_pipelines(
                (
                    &input_bind_group_layout,
                    &feed_forward_bind_group_layout,
                    &loss_function_bind_group_layout,
                    &back_propogation_bind_group_layout,
                    &learning_rate_bind_group_layout,
                ),
                device,
            );

        Self {
            num_inputs: feed_forward_input.num_inputs,
            num_outputs,
            // -------------------------------
            weights_buffer,
            bias_buffer,
            output_buffer,
            // -------------------------------
            loss_function_buffer,
            loss_function_info_buffer,
            expected_values_buffer,
            // -------------------------------
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            gradient_back_prop_buffer,
            // -------------------------------
            input_bind_group,
            // -------------------------------
            feed_forward_bind_group,
            // -------------------------------
            loss_function_bind_group,
            // -------------------------------
            learning_rate_bind_group,
            // -------------------------------
            back_propogation_bind_group,
            // -------------------------------
            feed_forward_pipeline,
            loss_function_pipeline,
            back_propogation_pipeline,
        }
    }

    /// Creates a output layer based on a descriptor
    /// Used for deserializing a model
    ///
    /// # Arguments
    ///
    /// * `output_layer_descriptor` - descriptor struct containing the output layers information such as inputs, outputs, weights, and basies
    /// * `feed_forward_input` - `&FeedForwardConnection` detailing the outputs of the previous layer
    /// * `device` - wgpu device for creating the layer buffers
    ///
    /// # Returns
    ///
    /// `OutputLayer` instance with the set weights and biases
    pub fn from_descriptor(
        output_layer_descriptor: &OutputLayerDescriptor,
        feed_forward_input: &FeedForwardConnection,
        learning_rate_buffer: &Buffer,
        device: &Device,
    ) -> Self {
        // Create all the buffers necessary in this layer
        let (
            intermediary_buffer,
            output_buffer,
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            regularization_output_buffer,
            gradient_buffer,
            loss_function_buffer,
            loss_function_info_buffer,
            expected_values_buffer,
            gradient_back_prop_buffer,
            dimensions_buffer,
            gradient_coefficient_buffer,
        ) = create_buffers(
            device,
            feed_forward_input,
            output_layer_descriptor.num_outputs,
        );

        // These buffers are set by the information given in the output layer descriptor
        let weights_buffer = Rc::new(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Output Layer Weights Buffer"),
            contents: bytemuck::cast_slice(&output_layer_descriptor.weights),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));

        let bias_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Output Layer Bias Buffer"),
            contents: bytemuck::cast_slice(&output_layer_descriptor.biases),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let (
            (input_bind_group_layout, input_bind_group),
            (feed_forward_bind_group_layout, feed_forward_bind_group),
            (loss_function_bind_group_layout, loss_function_bind_group),
            (back_propogation_bind_group_layout, back_propogation_bind_group),
            (learning_rate_bind_group_layout, learning_rate_bind_group),
        ) = create_bind_groups(
            device,
            &feed_forward_input.buffer,
            &dimensions_buffer,
            &weights_buffer,
            &bias_buffer,
            &intermediary_buffer,
            &output_buffer,
            &l_1_norm_buffer,
            &frobenius_norm_buffer,
            &regularization_info_buffer,
            &regularization_output_buffer,
            &gradient_buffer,
            &gradient_coefficient_buffer,
            &gradient_back_prop_buffer,
            &loss_function_buffer,
            &loss_function_info_buffer,
            &expected_values_buffer,
            learning_rate_buffer,
        );

        let (feed_forward_pipeline, loss_function_pipeline, back_propogation_pipeline) =
            create_pipelines(
                (
                    &input_bind_group_layout,
                    &feed_forward_bind_group_layout,
                    &loss_function_bind_group_layout,
                    &back_propogation_bind_group_layout,
                    &learning_rate_bind_group_layout,
                ),
                device,
            );
        Self {
            num_inputs: output_layer_descriptor.num_inputs,
            num_outputs: output_layer_descriptor.num_outputs,
            // -------------------------------
            weights_buffer,
            bias_buffer,
            output_buffer,
            // -------------------------------
            loss_function_buffer,
            loss_function_info_buffer,
            expected_values_buffer,
            // -------------------------------
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            gradient_back_prop_buffer,
            // -------------------------------
            input_bind_group,
            // -------------------------------
            feed_forward_bind_group,
            // -------------------------------
            loss_function_bind_group,
            // -------------------------------
            back_propogation_bind_group,
            // -------------------------------
            learning_rate_bind_group,
            // -------------------------------
            feed_forward_pipeline,
            loss_function_pipeline,
            back_propogation_pipeline,
        }
    }

    /// Creates a descriptor of the output layer to be used for serializing the
    /// weights and bias information
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device to get the necessary buffers from gpu memory
    /// * `queue` - wgpu queue to sumbit commands to the gpu
    ///
    /// # Returns
    ///
    /// a `OutputLayerDescriptor` detailing the number of inputs, number of outputs, weights, and biases
    pub fn to_descriptor(&self, device: &Device, queue: &Queue) -> OutputLayerDescriptor {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Output Layer Description Generator Encoder"),
        });

        let weights_buffer = read_buffer(
            &self.weights_buffer,
            self.num_outputs * self.num_inputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        let biases_buffer = read_buffer(
            &self.bias_buffer,
            self.num_outputs * std::mem::size_of::<Bias>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        let (weights, biases) = (
            get_buffer(&weights_buffer, device),
            bytemuck::cast_slice::<f32, Bias>(&get_buffer(&biases_buffer, device)).to_vec(),
        );

        OutputLayerDescriptor {
            num_outputs: self.num_outputs,
            num_inputs: self.num_inputs,
            weights,
            biases,
        }
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

    /// Sets the values of the expected wieghts buffer
    ///
    /// # Arguments
    ///
    /// * `expected_values` - slice of values to send to the gpu
    /// * `queue` - wgpu queue to create command for the gpu
    pub fn set_expected_weights(&self, expected_values: &[f32], queue: &Queue) {
        queue.write_buffer(
            &self.expected_values_buffer,
            0,
            bytemuck::cast_slice(&expected_values),
        );
    }

    /// Sets the information of the loss function buffer to the given loss function
    ///
    /// # Arguments
    ///
    /// * `loss_function` - Loss function to use for this layer
    /// * `queue` - wgpu queue to write the information to the gpu
    pub fn set_loss_function(&self, loss_function: LossFunction, queue: &Queue) {
        queue.write_buffer(
            &self.loss_function_info_buffer,
            0,
            bytemuck::cast_slice(&[loss_function.as_repr()]),
        );
    }

    /// Gets the cost from the loss that has been computed the last time that
    /// the back propogation was ran
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device to create a command encoder
    /// * `queue` - wgpu queue to send commands to the gpu
    ///
    /// # Returns
    ///
    /// `f32` value representing the cost (average of all the loss values)
    pub fn get_cost(&self, device: &Device, queue: &Queue) -> f32 {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Output Cust Read Command Encoder"),
        });

        let loss = read_buffer(
            &self.loss_function_buffer,
            self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        let vals = get_buffer(&loss, device);

        vals.iter().sum::<f32>() / vals.len() as f32
    }

    /// Gets the predicted values from the last time the feed forward has been run
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device to create the command encoder
    /// * `queue` - wgpu queue to senc commands to the gpu
    ///
    /// # Returns
    ///
    /// `Vec<f32>` containing all the predicted values from the last `feed_forward` call
    pub fn get_predicted_values(&self, device: &Device, queue: &Queue) -> Vec<f32> {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Output Layer Predicted Values Getter Command Encoder"),
        });

        let prediction = read_buffer(
            &self.output_buffer,
            self.num_outputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        get_buffer(&prediction, device)
    }
}

impl BackPropogationLayerConnection for OutputLayer {
    fn get_ceoff_back_prop_buffer(&self) -> Rc<Buffer> {
        self.gradient_back_prop_buffer.clone()
    }
}

impl FeedForwardLayer for OutputLayer {
    fn feed_forward(&self, device: &Device, queue: &Queue) {
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

        queue.submit(Some(encoder.finish()));
    }
}

impl BackPropogationLayer for OutputLayer {
    fn back_propogate(&self, regularization: Regularization, device: &Device, queue: &Queue) {
        // Compute the loss
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Output Layer Back Propogation Command Encoder"),
        });

        // Run the loss function pipeline
        {
            let dispatch_size = compute_workgroup_size(
                self.num_outputs.max(self.num_inputs) as u32,
                WORK_GROUP_SIZE,
            );

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Output Layer Loss Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.loss_function_pipeline);

            // Set bind groups
            compute_pass.set_bind_group(0, &self.loss_function_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        encoder.insert_debug_marker("Sync Point: Output Loss Pipeline Complete");
        device.poll(Maintain::Wait);

        // back propogate and gradient descent
        let reg_repr = RegRepr {
            function: match regularization.function {
                RegularizationFunction::Lasso => {
                    _ = self.generate_weights_l_1_norm(device, queue);

                    0
                }
                RegularizationFunction::Ridge => {
                    _ = self.generate_weights_frobenius_norm(device, queue);

                    1
                }
                RegularizationFunction::ElasticNetRegression => {
                    _ = self.generate_weights_l_1_norm(device, queue);
                    _ = self.generate_weights_frobenius_norm(device, queue);

                    2
                }
            },
            hyper_parameter_1: regularization.hyper_parameter_1,
            hyper_parameter_2: regularization.hyper_parameter_2,
        };

        queue.write_buffer(
            &self.regularization_info_buffer,
            0,
            bytemuck::cast_slice(&[reg_repr]),
        );

        // Run the Back propogation pipeline
        {
            let (dispatch_width, dispatch_height) = compute_2d_workgroup_size(
                (self.num_inputs as u32, self.num_outputs as u32),
                (D2_WORK_GROUP_SIZE, D2_WORK_GROUP_SIZE),
            );

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Output Layer Back Propogation Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.back_propogation_pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &self.back_propogation_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.input_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.learning_rate_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        encoder.insert_debug_marker("Sync Point: Output Back Prop Pipeline Finished");
        device.poll(Maintain::Wait);

        queue.submit(Some(encoder.finish()));
    }
}
