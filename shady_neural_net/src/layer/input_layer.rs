use std::rc::Rc;

use serde::{Deserialize, Serialize};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderStages,
    include_wgsl,
};

use crate::create_buffer_bind_group;

use super::{
    FeedForwardLayerConnection, WORK_GROUP_SIZE, compute_workgroup_size, errors::InputLengthMismatchError,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct InputLayerDescriptor {
    pub num_inputs: u64,
}

/// Input Layer struct used in neural net layer
#[allow(dead_code)]
#[derive(Debug)]
pub struct InputLayer {
    pub num_inputs: u64,
    buffer: Rc<Buffer>,

    bind_group: Rc<BindGroup>,
    bind_group_layout: Rc<BindGroupLayout>,
    pipeline: ComputePipeline,
}

impl InputLayer {
    /// Initialize a new input layer with the necessary buffer
    ///
    /// # Arguments
    ///
    /// * `num_inputs` - number of inputs in this layer
    /// * `device` - a reference to wgpu device to create necessary buffers
    ///
    /// # Returns
    ///
    /// A new instance of `InputLayer`
    pub fn new(num_inputs: u64, device: &Device) -> Self {
        // Create the buffer from the input data

        let buffer = Rc::new(device.create_buffer(&BufferDescriptor {
            label: Some("Input Layer Buffer"),
            mapped_at_creation: false,
            size: num_inputs * std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        }));

        let (bind_group_layout, bind_group) = create_buffer_bind_group!(
            device,
            "Input Layer Bind Group",
            (0, &buffer, Bbt::Storage { read_only: false })
        );

        let (bind_group_layout, bind_group) = (Rc::new(bind_group_layout), Rc::new(bind_group));

        let shader = device.create_shader_module(include_wgsl!(
            "../shaders/input_layer/input_layer_feed_forward.wgsl"
        ));

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

    /// Creates an input layer from a descriptor that has been provided
    ///
    /// # Arguments
    ///
    /// * `input_layer_descriptor` - descriptor of the number of inputs in the layer
    /// * `device` - reference to the wgpu device to create the input layer
    ///
    /// # Returns
    ///
    /// A new instance of `InputLayer`
    pub fn from_descriptor(input_layer_descriptor: &InputLayerDescriptor, device: &Device) -> Self {
        let buffer = Rc::new(device.create_buffer(&BufferDescriptor {
            label: Some("Input Layer Buffer"),
            mapped_at_creation: false,
            size: input_layer_descriptor.num_inputs * std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));

        let (bind_group_layout, bind_group) = create_buffer_bind_group!(
            device,
            "Input Layer Bind Group",
            (0, &buffer, Bbt::Storage { read_only: false })
        );

        let (bind_group_layout, bind_group) = (Rc::new(bind_group_layout), Rc::new(bind_group));

        let shader = device.create_shader_module(include_wgsl!(
            "../shaders/input_layer/input_layer_feed_forward.wgsl"
        ));

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
            num_inputs: input_layer_descriptor.num_inputs,
            buffer,
            bind_group,
            bind_group_layout,
            pipeline,
        }
    }

    /// Creates a descriptor for the input layer that can be used to
    /// serialize the data in the layer
    ///
    /// # Returns
    ///
    /// an `InputLayerDescriptor` containing the number of inputs in the layer
    pub fn to_descriptor(&self) -> InputLayerDescriptor {
        InputLayerDescriptor {
            num_inputs: self.num_inputs,
        }
    }

    /// Sets the inputs of the input layer buffer
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of inputs to set as the input layer
    /// * `device` - a reference to wgpu device to set inputs layer
    /// * `queue` - a reference to wgpu queue to set inputs layer
    ///
    /// # Returns
    ///
    /// `Result` of `Ok(())` if the inputs were set successfully
    /// or `Err(InputLengthMismatchError)` if the input vector
    /// is not the same size as the input buffer
    pub fn set_inputs(
        &self,
        mut inputs: Vec<f32>,
        device: &Device,
        queue: &Queue,
    ) -> Result<(), InputLengthMismatchError> {
        if inputs.len() != self.num_inputs as usize {
            return Err(InputLengthMismatchError);
        }

        // Normalize the inputs before sending them through
        {
            let avg = inputs.iter().sum::<f32>() / inputs.len() as f32;
            inputs = inputs.iter_mut().map(|value| *value / avg).collect();
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Input Layer Command Encoder"),
        });

        queue.write_buffer(self.buffer.as_ref(), 0, bytemuck::cast_slice(&inputs));

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

impl FeedForwardLayerConnection for InputLayer {
    fn get_output_buffer(&self) -> Rc<Buffer> {
        self.buffer.clone()
    }
}
