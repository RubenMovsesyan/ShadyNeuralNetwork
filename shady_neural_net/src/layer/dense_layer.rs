use std::rc::Rc;

use crate::{
    create_buffer_bind_group,
    layer::{D2_WORK_GROUP_SIZE, compute_2d_workgroup_size},
    regularization::RegularizationFunction,
    utils::{get_buffer, print_buffer, read_buffer},
};

use super::{
    BackPropogationConnection, BackPropogationLayer, BackPropogationLayerConnection,
    FeedForwardConnection, FeedForwardLayer, FeedForwardLayerConnection, WORK_GROUP_SIZE,
    activation::ActivationFunction,
    bias::Bias,
    compute_workgroup_size,
    regularization::{RegRepr, Regularization},
    weight_distribution::WeightDistribution,
};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderModule,
    ShaderStages, include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct DenseLayerDescriptor {
    pub num_nodes: u64,
    pub num_inputs: u64,
    pub activation_function: ActivationFunction,

    pub weights: Vec<f32>,
    pub biases: Vec<Bias>,
}

fn create_bind_groups(
    device: &Device,
    input_buffer: &Buffer,
    dimensions_buffer: &Buffer,
    weights_buffer: &Buffer,
    bias_buffer: &Buffer,
    activation_function_buffer: &Buffer,
    intermediary_buffer: &Buffer,
    output_buffer: &Buffer,
    l_1_norm_buffer: &Buffer,
    frobenius_norm_buffer: &Buffer,
    regularization_info_buffer: &Buffer,
    regularization_output_buffer: &Buffer,
    gradient_buffer: &Buffer,
    gradient_coefficient_buffer: &Buffer,
    learning_rate_buffer: &Buffer,
) -> (
    (BindGroupLayout, BindGroup),
    (BindGroupLayout, BindGroup),
    (BindGroupLayout, BindGroup),
    (BindGroupLayout, BindGroup),
) {
    let (input_bind_group_layout, input_bind_group) = create_buffer_bind_group!(
        device,
        "Dense Layer Input Bind Group",
        (0, input_buffer, Bbt::Storage { read_only: true })
    );

    let (feed_forward_bind_group_layout, feed_forward_bind_group) = create_buffer_bind_group!(
        device,
        "Dense Layer Feed Forward Bind Group",
        (0, dimensions_buffer, Bbt::Uniform),
        (1, weights_buffer, Bbt::Storage { read_only: true }),
        (2, bias_buffer, Bbt::Storage { read_only: true }),
        (3, activation_function_buffer, Bbt::Uniform),
        (4, intermediary_buffer, Bbt::Storage { read_only: false }),
        (5, output_buffer, Bbt::Storage { read_only: false })
    );

    let (back_propogation_bind_group_layout, back_propogation_bind_group) = create_buffer_bind_group!(
        device,
        "Dense Layer Back Propogation Bind Group",
        (0, l_1_norm_buffer, Bbt::Uniform),
        (1, frobenius_norm_buffer, Bbt::Uniform),
        (2, dimensions_buffer, Bbt::Uniform),
        (3, weights_buffer, Bbt::Storage { read_only: false }),
        (4, gradient_buffer, Bbt::Storage { read_only: false }),
        (
            5,
            gradient_coefficient_buffer,
            Bbt::Storage { read_only: true }
        ),
        (6, input_buffer, Bbt::Storage { read_only: true }),
        (7, regularization_info_buffer, Bbt::Uniform),
        (
            8,
            regularization_output_buffer,
            Bbt::Storage { read_only: false }
        )
    );

    let (learning_rate_bind_group_layout, learning_rate_bind_group) = create_buffer_bind_group!(
        device,
        "Dense Layer Learning Rate Bind Group",
        (0, learning_rate_buffer, Bbt::Uniform)
    );

    (
        (input_bind_group_layout, input_bind_group),
        (feed_forward_bind_group_layout, feed_forward_bind_group),
        (
            back_propogation_bind_group_layout,
            back_propogation_bind_group,
        ),
        (learning_rate_bind_group_layout, learning_rate_bind_group),
    )
}

fn create_feed_forward_pipeline(
    input_bind_group_layout: &BindGroupLayout,
    feed_forward_bind_group_layout: &BindGroupLayout,
    device: &Device,
) -> ComputePipeline {
    let shader: ShaderModule =
        device.create_shader_module(include_wgsl!("../shaders/dense_layer/feed_forward.wgsl"));

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Dense Layer Feed Forward Compute Pipeline Layout"),
        bind_group_layouts: &[input_bind_group_layout, feed_forward_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Dense Layer Feed Forward Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("dense_layer_feed_forward_main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn create_back_propogation_pipeline(
    back_prop_bind_group_layout: &BindGroupLayout,
    learning_rate_bind_group_layout: &BindGroupLayout,
    device: &Device,
) -> ComputePipeline {
    let shader: ShaderModule =
        device.create_shader_module(include_wgsl!("../shaders/dense_layer/back_propogate.wgsl"));

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Dense Layer Back Propogation Compute Pipeline Layout"),
        bind_group_layouts: &[back_prop_bind_group_layout, learning_rate_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Dense Layer Back Propogation Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("dense_layer_back_propogation_main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn create_buffers(
    device: &Device,
    input_connecting_bind_group: &FeedForwardConnection,
    num_nodes: u64,
    activation_function: ActivationFunction,
) -> (
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
    Rc<Buffer>,
) {
    let dimensions_buffer = {
        let mut dimensions = Vec::new();
        dimensions.push(num_nodes as u32);
        dimensions.push(input_connecting_bind_group.num_inputs as u32);

        Rc::new(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Dense Layer Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dimensions),
            usage: BufferUsages::UNIFORM,
        }))
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
        size: input_connecting_bind_group.num_inputs * std::mem::size_of::<f32>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    }));

    let gradient_intermediary_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Dense Layer Gradient Intermediary Buffer"),
        mapped_at_creation: false,
        size: num_nodes * std::mem::size_of::<f32>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    });

    let gradient_back_prop_buffer = Rc::new(device.create_buffer(&BufferDescriptor {
        label: Some("Dense Layer Gradent Back Propogation Buffer"),
        mapped_at_creation: false,
        size: input_connecting_bind_group.num_inputs * std::mem::size_of::<f32>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    }));

    (
        activation_function_buffer,
        intermediary_buffer,
        l_1_norm_buffer,
        frobenius_norm_buffer,
        regularization_info_buffer,
        regularization_output_buffer,
        gradient_buffer,
        gradient_intermediary_buffer,
        gradient_back_prop_buffer,
        dimensions_buffer,
        output_buffer,
        gradient_coefficient_buffer,
    )
}

/// Dense layer struct used in neural net
// #[allow(dead_code)]
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
    activation_function_buffer: Buffer,
    gradient_buffer: Buffer,
    gradient_coefficient_buffer: Rc<Buffer>,
    gradient_intermediary_buffer: Buffer,
    gradient_back_prop_buffer: Rc<Buffer>,

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

    learning_rate_bind_group_layout: BindGroupLayout,
    learning_rate_bind_group: BindGroup,

    // Gradient descent bind groups
    gradient_descent_bind_group_layout: Option<BindGroupLayout>,
    gradient_descent_bind_group: Option<BindGroup>,

    // GPU pipeline information
    feed_forward_pipeline: ComputePipeline,
    back_propogation_pipeline: ComputePipeline,
    coeff_pipeline: Option<ComputePipeline>,
    regularization_pipeline: Option<ComputePipeline>,
    gradient_descent_pipeline: Option<ComputePipeline>,

    // Buffer information that needs to be linked after creation
    next_layer_gradient_coefficient_buffer: Option<Rc<Buffer>>,
    next_layer_weights_buffer: Option<Rc<Buffer>>,
    next_layer_dimensions_buffer: Option<Rc<Buffer>>,
    next_layer_bind_group_layout: Option<BindGroupLayout>,
    next_layer_bind_group: Option<BindGroup>,

    // Buffer information that is needed to get the gradient coefficient
    coeff_bind_group_layout: Option<BindGroupLayout>,
    coeff_bind_group: Option<BindGroup>,
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
        learning_rate_buffer: &Buffer,
        device: &Device,
    ) -> Self {
        // Create all the buffers necessary in this layer
        let (
            activation_function_buffer,
            intermediary_buffer,
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            regularization_output_buffer,
            gradient_buffer,
            gradient_intermediary_buffer,
            gradient_back_prop_buffer,
            dimensions_buffer,
            output_buffer,
            gradient_coefficient_buffer,
        ) = create_buffers(
            device,
            input_connecting_bind_group,
            num_nodes,
            activation_function,
        );

        // Initialize the weights matrix buffer with random values from -1.0 to 1.0
        // containts a matrix with num_nodes sets of weights
        // each with num_inputs weights in them
        let weights_buffer = {
            let weights = WeightDistribution::Xavier
                .get_weight_distribution(input_connecting_bind_group.num_inputs, num_nodes);

            Rc::new(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Dense Layer Weights Buffer"),
                contents: bytemuck::cast_slice(&weights),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }))
        };

        // Initialize the bias vector buffer with random values from -1.0 to 1.0
        // each Bias is a bias value and a bias weight
        let bias_buffer = {
            let biases = WeightDistribution::Xavier.get_bias_distribution(num_nodes);

            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Dense Layer Bias Buffer"),
                contents: bytemuck::cast_slice(&biases),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        };

        let (
            (input_bind_group_layout, input_bind_group),
            (feed_forward_bind_group_layout, feed_forward_bind_group),
            (back_propogation_bind_group_layout, back_propogation_bind_group),
            (learning_rate_bind_group_layout, learning_rate_bind_group),
        ) = create_bind_groups(
            device,
            &input_connecting_bind_group.buffer,
            &dimensions_buffer,
            &weights_buffer,
            &bias_buffer,
            &activation_function_buffer,
            &intermediary_buffer,
            &output_buffer,
            &l_1_norm_buffer,
            &frobenius_norm_buffer,
            &regularization_info_buffer,
            &regularization_output_buffer,
            &gradient_buffer,
            &gradient_coefficient_buffer,
            learning_rate_buffer,
        );

        // Create the pipeline from the bind group layout
        let feed_forward_pipeline = create_feed_forward_pipeline(
            &input_bind_group_layout,
            &feed_forward_bind_group_layout,
            device,
        );

        let back_propogation_pipeline = create_back_propogation_pipeline(
            &back_propogation_bind_group_layout,
            &learning_rate_bind_group_layout,
            device,
        );

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
            activation_function_buffer,
            gradient_buffer,
            gradient_coefficient_buffer,
            gradient_intermediary_buffer,
            gradient_back_prop_buffer,
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

            learning_rate_bind_group_layout,
            learning_rate_bind_group,
            // ------------------------------------
            gradient_descent_bind_group_layout: None,
            gradient_descent_bind_group: None,
            // ------------------------------------
            feed_forward_pipeline,
            back_propogation_pipeline,
            coeff_pipeline: None,
            regularization_pipeline: None,
            gradient_descent_pipeline: None,
            // ------------------------------------
            next_layer_gradient_coefficient_buffer: None,
            next_layer_weights_buffer: None,
            next_layer_dimensions_buffer: None,
            next_layer_bind_group_layout: None,
            next_layer_bind_group: None,
            coeff_bind_group_layout: None,
            coeff_bind_group: None,
        }
    }

    /// Creates a dense layer based on a descriptor
    /// Used for deserializing a model
    ///
    /// # Arguments
    ///
    /// * `dense_layer_descriptor` - descriptor struct containing the dense layers information such as inputs, outputs, weights, and basies
    /// * `input_connecting_bind_group` - `&FeedForwardConnection` detailing the outputs of the previous layer
    /// * `device` - wgpu device for creating the layer buffers
    ///
    /// # Returns
    ///
    /// `DenseLayer` instance with the set weights and biases
    pub fn from_descriptor(
        dense_layer_descriptor: &DenseLayerDescriptor,
        input_connecting_bind_group: &FeedForwardConnection,
        learning_rate_buffer: &Buffer,
        device: &Device,
    ) -> Self {
        // Create all the buffers necessary in this layer
        let (
            activation_function_buffer,
            intermediary_buffer,
            l_1_norm_buffer,
            frobenius_norm_buffer,
            regularization_info_buffer,
            regularization_output_buffer,
            gradient_buffer,
            gradient_intermediary_buffer,
            gradient_back_prop_buffer,
            dimensions_buffer,
            output_buffer,
            gradient_coefficient_buffer,
        ) = create_buffers(
            device,
            input_connecting_bind_group,
            dense_layer_descriptor.num_nodes,
            dense_layer_descriptor.activation_function,
        );

        // Initialize the weights matrix buffer with random values from -1.0 to 1.0
        // containts a matrix with num_nodes sets of weights
        // each with num_inputs weights in them
        let weights_buffer = Rc::new(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Dense Layer Weights Buffer"),
            contents: bytemuck::cast_slice(&dense_layer_descriptor.weights),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));

        // Initialize the bias vector buffer with random values from -1.0 to 1.0
        // each Bias is a bias value and a bias weight
        let bias_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Dense Layer Bias Buffer"),
            contents: bytemuck::cast_slice(&dense_layer_descriptor.biases),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let (
            (input_bind_group_layout, input_bind_group),
            (feed_forward_bind_group_layout, feed_forward_bind_group),
            (back_propogation_bind_group_layout, back_propogation_bind_group),
            (learning_rate_bind_group_layout, learning_rate_bind_group),
        ) = create_bind_groups(
            device,
            &input_connecting_bind_group.buffer,
            &dimensions_buffer,
            &weights_buffer,
            &bias_buffer,
            &activation_function_buffer,
            &intermediary_buffer,
            &output_buffer,
            &l_1_norm_buffer,
            &frobenius_norm_buffer,
            &regularization_info_buffer,
            &regularization_output_buffer,
            &gradient_buffer,
            &gradient_coefficient_buffer,
            learning_rate_buffer,
        );

        // Create the pipeline from the bind group layout
        let feed_forward_pipeline = create_feed_forward_pipeline(
            &input_bind_group_layout,
            &feed_forward_bind_group_layout,
            device,
        );

        let back_propogation_pipeline = create_back_propogation_pipeline(
            &back_propogation_bind_group_layout,
            &learning_rate_bind_group_layout,
            device,
        );

        Self {
            num_nodes: dense_layer_descriptor.num_nodes,
            num_inputs: dense_layer_descriptor.num_inputs,
            activation_function: dense_layer_descriptor.activation_function,
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
            activation_function_buffer,
            gradient_buffer,
            gradient_coefficient_buffer,
            gradient_intermediary_buffer,
            gradient_back_prop_buffer,
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

            learning_rate_bind_group_layout,
            learning_rate_bind_group,
            // ------------------------------------
            gradient_descent_bind_group_layout: None,
            gradient_descent_bind_group: None,
            // ------------------------------------
            feed_forward_pipeline,
            back_propogation_pipeline,
            coeff_pipeline: None,
            regularization_pipeline: None,
            gradient_descent_pipeline: None,
            // ------------------------------------
            next_layer_gradient_coefficient_buffer: None,
            next_layer_weights_buffer: None,
            next_layer_dimensions_buffer: None,
            next_layer_bind_group_layout: None,
            next_layer_bind_group: None,
            coeff_bind_group_layout: None,
            coeff_bind_group: None,
        }
    }

    /// Creates a descriptor of the dense layer to be used for serializing the
    /// weights and bias information
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device to get the necessary buffers from gpu memory
    /// * `queue` - wgpu queue to sumbit commands to the gpu
    ///
    /// # Returns
    ///
    /// a `DenseLayerDescriptor` detailing the number of inputs, number of nodes, weights, and biases
    pub fn to_descriptor(&self, device: &Device, queue: &Queue) -> DenseLayerDescriptor {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Dense Layer Description Generator Encoder"),
        });

        let weights_buffer = read_buffer(
            &self.weights_buffer,
            self.num_nodes * self.num_inputs * std::mem::size_of::<f32>() as u64,
            device,
            &mut encoder,
        );

        let biases_buffer = read_buffer(
            &self.bias_buffer,
            self.num_nodes * std::mem::size_of::<Bias>() as u64,
            device,
            &mut encoder,
        );

        queue.submit(Some(encoder.finish()));

        let (weights, biases) = (
            get_buffer(&weights_buffer, device),
            bytemuck::cast_slice::<f32, Bias>(&get_buffer(&biases_buffer, device)).to_vec(),
        );

        DenseLayerDescriptor {
            num_nodes: self.num_nodes,
            num_inputs: self.num_inputs,
            activation_function: self.activation_function.clone(),
            weights,
            biases,
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
        let (coeff_bind_group_layout, coeff_bind_group) = create_buffer_bind_group!(
            device,
            "Dense Layer Coefficient Bind Group",
            (
                0,
                self.intermediary_buffer,
                Bbt::Storage { read_only: true }
            ),
            (1, self.output_buffer, Bbt::Storage { read_only: true }),
            (2, self.dimensions_buffer, Bbt::Uniform),
            (3, self.weights_buffer, Bbt::Storage { read_only: true }),
            (4, self.activation_function_buffer, Bbt::Uniform),
            (
                5,
                self.gradient_intermediary_buffer,
                Bbt::Storage { read_only: false }
            ),
            (
                6,
                &back_propogation_connection.grad_coeff_back_prop_buffer,
                Bbt::Storage { read_only: true }
            ),
            (
                7,
                self.gradient_coefficient_buffer,
                Bbt::Storage { read_only: false }
            ),
            (
                8,
                self.gradient_back_prop_buffer,
                Bbt::Storage { read_only: false }
            )
        );

        // Create the pipeline to compute the coefficient
        let coeff_pipeline = {
            let shader =
                device.create_shader_module(include_wgsl!("../shaders/dense_layer/coeff.wgsl"));

            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Dense Layer Coefficient Compute Pipeline Layout"),
                bind_group_layouts: &[&self.input_bind_group_layout, &coeff_bind_group_layout],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Dense Layer Coefficient Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("dense_layer_coefficient_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        self.coeff_bind_group_layout = Some(coeff_bind_group_layout);
        self.coeff_bind_group = Some(coeff_bind_group);
        self.coeff_pipeline = Some(coeff_pipeline);
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
}

impl FeedForwardLayerConnection for DenseLayer {
    fn get_output_buffer(&self) -> Rc<Buffer> {
        self.output_buffer.clone()
    }
}

impl BackPropogationLayerConnection for DenseLayer {
    fn get_ceoff_back_prop_buffer(&self) -> Rc<Buffer> {
        self.gradient_back_prop_buffer.clone()
    }

    fn get_weights_buffer(&self) -> Rc<Buffer> {
        self.weights_buffer.clone()
    }

    fn get_dimensions_buffer(&self) -> Rc<Buffer> {
        self.dimensions_buffer.clone()
    }
}

impl FeedForwardLayer for DenseLayer {
    fn feed_forward(&self, device: &Device, queue: &Queue) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Dense Layer Feed Forward Command Encoder"),
        });

        // Run the Pipeline
        {
            let dispatch_size = compute_workgroup_size(self.num_nodes as u32, WORK_GROUP_SIZE);

            // Begine the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Dense Layer Feed Forward Compute Pass"),
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

        encoder.insert_debug_marker("Sync Point: Dense Feed Forward Finished");
        device.poll(Maintain::Wait);

        queue.submit(Some(encoder.finish()));
    }
}

impl BackPropogationLayer for DenseLayer {
    fn back_propogate(&self, regularization: Regularization, device: &Device, queue: &Queue) {
        // Compute the gradient coefficient
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Dense Layer Back Propogation Command Encoder"),
        });

        let reg = match regularization.function {
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
        };

        queue.write_buffer(
            &self.regularization_info_buffer,
            0,
            bytemuck::cast_slice(&[RegRepr {
                function: reg,
                hyper_parameter_1: regularization.hyper_parameter_1,
                hyper_parameter_2: regularization.hyper_parameter_2,
            }]),
        );

        // Run the gradient coefficient pipeline
        {
            let dispatch_size =
                compute_workgroup_size(self.num_nodes.max(self.num_inputs) as u32, WORK_GROUP_SIZE);

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Dense Layer Gradient Coefficient Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&self.coeff_pipeline.as_ref().unwrap());

            // Set the bind groups
            compute_pass.set_bind_group(0, &self.input_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.coeff_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        // let gradient = read_buffer(
        //     &self.gradient_buffer,
        //     self.num_inputs * self.num_nodes * std::mem::size_of::<f32>() as u64,
        //     device,
        //     &mut encoder,
        // );

        // let weights = read_buffer(
        //     &self.weights_buffer,
        //     self.num_inputs * self.num_nodes * std::mem::size_of::<f32>() as u64,
        //     device,
        //     &mut encoder,
        // );

        encoder.insert_debug_marker("Sync Point: Dense Layer Coefficient Pipeline");
        device.poll(Maintain::Wait);

        // Back Propogation compute pass
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
            compute_pass.set_pipeline(&self.back_propogation_pipeline);

            // Set the bind Groups
            compute_pass.set_bind_group(0, &self.back_propogation_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.learning_rate_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        encoder.insert_debug_marker("Sync Point: Dense Layer Back Propogation Pipeline");
        device.poll(Maintain::Wait);

        queue.submit(Some(encoder.finish()));

        // print_buffer(&gradient, device, "Dense Layer Gradient Buffer");
        // print_buffer(&weights, device, "Dense Layer Weights Buffer");
    }
}
