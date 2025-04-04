use std::error::Error;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, ShaderStages,
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    errors::GpuMathNotInitializedError,
    get_device, get_pipelines, get_queue,
    gpu_utils::{WORK_GROUP_SIZE_2D, compute_workgroup_size_2d, get_buffer, read_buffer},
    test_init,
};

use super::math_errors::MatrixDotError;

const DATA_SIZE: u64 = std::mem::size_of::<f32>() as u64;

#[allow(dead_code)]
#[derive(Debug)]
pub struct MatrixPipelines {
    // Bind Group Layouts
    readable_bind_group_layout: BindGroupLayout,
    writable_bind_group_layout: BindGroupLayout,

    // Pipeline Layouts
    matrix_matrix_pipeline_layout: PipelineLayout,

    // Pipelines
    dot_pipeline: ComputePipeline,
}

impl MatrixPipelines {
    pub fn init(device: &Device) -> Result<Self, GpuMathNotInitializedError> {
        // Create the readable bind group layout for the pipelines
        let readable_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Readable Bind Group Layout"),
                entries: &[
                    // Matrix Buffer
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Dimensions
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Transpose
                    BindGroupLayoutEntry {
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

        // Create the writable bind group layout for the pipelines
        let writable_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Readable Bind Group Layout"),
                entries: &[
                    // Matrix Buffer
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
                    // Dimensions
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Transpose
                    BindGroupLayoutEntry {
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

        // This is the pipeline layout for a Matrix Matrix operation
        let matrix_matrix_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Matrix Matrix Pipeline Layout"),
                bind_group_layouts: &[
                    &readable_bind_group_layout,
                    &readable_bind_group_layout,
                    &writable_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let dot_shader = device.create_shader_module(include_wgsl!("shaders/dotting.wgsl"));

        let dot_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Matrix Dot Pipeline"),
            module: &dot_shader,
            layout: Some(&matrix_matrix_pipeline_layout),
            cache: None,
            compilation_options: PipelineCompilationOptions::default(),
            entry_point: Some("dot_main"),
        });

        Ok(Self {
            readable_bind_group_layout,
            writable_bind_group_layout,
            matrix_matrix_pipeline_layout,
            dot_pipeline,
        })
    }
}

#[derive(Debug)]
pub struct Matrix {
    rows: u32,
    cols: u32,
    // GPU vals
    dimensions: Buffer,
    data: Buffer,
    transpose: Buffer,

    // Bind Groups
    readable_bind_group: BindGroup,
    writable_bind_group: BindGroup,
}

impl Matrix {
    /// Creates a new matrix with a specified `shape` and fills it with `data` if provided
    /// The Matrix will not be transposed by default
    pub fn new(shape: (u32, u32), data: Option<Vec<f32>>) -> Result<Self, Box<dyn Error>> {
        test_init("Matrix::new")?;

        let device = unsafe { get_device() };
        let pipeline_information = unsafe { get_pipelines() };

        let buffer = match data {
            Some(data) => device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Matrix Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }),
            None => device.create_buffer(&BufferDescriptor {
                label: Some("Matrix Buffer"),
                mapped_at_creation: false,
                size: shape.0 as u64 * shape.1 as u64 * DATA_SIZE,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }),
        };

        let dimensions = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Dimensions Buffer"),
            contents: bytemuck::cast_slice(&vec![shape.0 as u32, shape.1 as u32]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let transpose = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Transpose Buffer"),
            contents: bytemuck::cast_slice(&[false as u32]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let readable_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Matrix Readable Bind Group"),
            layout: &pipeline_information.readable_bind_group_layout,
            entries: &[
                // Data Buffer
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                // Dimensions
                BindGroupEntry {
                    binding: 1,
                    resource: dimensions.as_entire_binding(),
                },
                // Transpose
                BindGroupEntry {
                    binding: 2,
                    resource: transpose.as_entire_binding(),
                },
            ],
        });

        let writable_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Matrix Writable Bind Group"),
            layout: &pipeline_information.writable_bind_group_layout,
            entries: &[
                // Data Buffer
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                // Dimensions
                BindGroupEntry {
                    binding: 1,
                    resource: dimensions.as_entire_binding(),
                },
                // Transpose
                BindGroupEntry {
                    binding: 2,
                    resource: transpose.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            rows: shape.0,
            cols: shape.1,
            dimensions,
            data: buffer,
            transpose,
            readable_bind_group,
            writable_bind_group,
        })
    }

    /// Dots the `source1` and `source2` matrix together and stores the output in `destination`
    pub fn dot(
        source1: &Matrix,
        source2: &Matrix,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Make sure that the gpu is initialized
        test_init("Matrix::dot")?;
        // Check to see if the matrix rows and columns line up
        if source1.cols != source2.rows {
            return Err(Box::new(MatrixDotError(
                "Source 1 cols do not match Source 2 rows in Matrix::dot".to_string(),
            )));
        }

        // Run the dot pipeline
        let device = unsafe { get_device() };
        let queue = unsafe { get_queue() };
        let pipeline_info = unsafe { get_pipelines() };

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Matrix Dot Command Encoder"),
        });

        {
            // Get the workgroup size
            let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                (destination.rows, destination.cols),
                (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
            );

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Matrix Dot Compute Pass"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline(&pipeline_info.dot_pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &source1.readable_bind_group, &[]);
            compute_pass.set_bind_group(1, &source2.readable_bind_group, &[]);
            compute_pass.set_bind_group(2, &destination.writable_bind_group, &[]);

            // Dispatch the workgroups
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        queue.submit(Some(encoder.finish()));

        Ok(())
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        match test_init("Matrix::eq as ParialEq") {
            Ok(_) => {}
            Err(err) => {
                panic!("{:#?}", err);
            }
        }

        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        let device = unsafe { get_device() };
        let queue = unsafe { get_queue() };

        let (mat1, mat2) = {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Matrix Reading Command Encoder"),
            });

            let buffer_1 = read_buffer(
                &self.data,
                self.rows as u64 * self.cols as u64 * DATA_SIZE,
                device,
                &mut encoder,
            );

            let buffer_2 = read_buffer(
                &other.data,
                other.rows as u64 * other.cols as u64 * DATA_SIZE,
                device,
                &mut encoder,
            );

            queue.submit(Some(encoder.finish()));

            (get_buffer(&buffer_1, device), get_buffer(&buffer_2, device))
        };

        // TODO: Check for transpose
        for (val1, val2) in mat1.iter().zip(mat2.iter()) {
            if val1 != val2 {
                return false;
            }
        }

        true
    }
}
