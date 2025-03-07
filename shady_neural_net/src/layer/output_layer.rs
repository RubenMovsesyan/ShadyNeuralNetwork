use std::rc::Rc;

use super::{ConnectingBindGroup, WORK_GROUP_SIZE, bias::Bias, compute_workgroup_size};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Maintain, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue,
    ShaderModule, ShaderStages, include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

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
                device.create_shader_module(include_wgsl!("../shaders/output_layer.wgsl"));

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
