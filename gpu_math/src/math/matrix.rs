use std::error::Error;

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferUsages, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, ShaderStages,
    include_wgsl,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{errors::GpuMathNotInitializedError, get_device, get_queue, test_init};

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
    pub fn init() -> Result<Self, GpuMathNotInitializedError> {
        test_init("MatrixPipelines::init")?;

        let device = unsafe { get_device() };

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
    shape: (u64, u64),
    dimensions: Buffer,
    data: Buffer,
    transpose: Buffer,
}

impl Matrix {
    /// Creates a new matrix with a specified `shape` and fills it with `data` if provided
    /// The Matrix will not be transposed by default
    pub fn new(shape: (u64, u64), data: Option<Vec<f32>>) -> Result<Self, Box<dyn Error>> {
        test_init("Matrix::new")?;

        let device = unsafe { get_device() };

        let buffer = match data {
            Some(data) => device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Matrix Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }),
            None => device.create_buffer(&BufferDescriptor {
                label: Some("Matrix Buffer"),
                mapped_at_creation: false,
                size: shape.0 * shape.1 * DATA_SIZE,
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

        Ok(Self {
            shape,
            dimensions,
            data: buffer,
            transpose,
        })
    }
}
