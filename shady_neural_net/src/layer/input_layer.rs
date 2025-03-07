use std::rc::Rc;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderStages,
    include_wgsl,
};

use super::{Layer, WORK_GROUP_SIZE, compute_workgroup_size, errors::InputLengthMismatchError};

/// Input Layer struct used in neural net layer
#[derive(Debug)]
pub struct InputLayer {
    pub num_inputs: u64,
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
