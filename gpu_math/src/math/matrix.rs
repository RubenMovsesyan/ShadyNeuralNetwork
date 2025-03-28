use std::env::consts::OS;
use std::fmt::Display;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
// WGPU imports
use wgpu::{
    BindGroup, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, Maintain,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, Queue,
    ShaderModuleDescriptor, include_wgsl,
};

use crate::create_buffer_bind_group;
use crate::gpu_utils::{
    WORK_GROUP_SIZE, WORK_GROUP_SIZE_2D, compute_workgroup_size, compute_workgroup_size_2d,
    get_buffer, read_buffer,
};

use super::math_errors::{
    MatrixAddError, MatrixCustomError, MatrixDotError, MatrixExpError, MatrixMultError,
    MatrixSubError, MatrixSumError, MatrixVariantError,
};

#[derive(Debug, Clone)]
struct CPUMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    transpose: bool,
}

#[derive(Debug)]
struct GPUMatrix {
    rows: u64,
    cols: u64,
    data: Buffer,
    transpose: bool,

    // Uniform to keep track of transpose
    transpose_buffer: Buffer,

    // Uniform for scalar multiplications and can be used for sending data to the gpu
    scalar_buffer: Buffer,

    // Buffer for summing elements
    sum_buffer: Buffer,

    // Bind Group Information for matrix operations
    bind_group: BindGroup,
    writable_bind_group: BindGroup,

    // Dotting
    dot_pipeline: ComputePipeline,

    // Adding
    add_pipeline: ComputePipeline,
    add_in_place_pipeline: ComputePipeline,
    add_scalar_in_place_pipeline: ComputePipeline,

    // Subtracting
    sub_pipeline: ComputePipeline,
    sub_in_place_pipeline: ComputePipeline,
    sub_scalar_in_place_pipeline: ComputePipeline,

    // Multiplying
    mult_pipeline: ComputePipeline,
    mult_in_place_pipeline: ComputePipeline,

    // Element-wise multiplication
    elem_mult_pipeline: ComputePipeline,
    elem_mult_in_place_pipeline: ComputePipeline,

    // Exponential
    exp_pipeline: ComputePipeline,
    exp_in_place_pipeline: ComputePipeline,

    // Summing all elements
    sum_pipeline: ComputePipeline,

    // Vectored Adding
    vectored_add_pipeline: ComputePipeline,
    vectored_add_in_place_pipeline: ComputePipeline,

    // Vectored Subtracting
    vectored_sub_pipeline: ComputePipeline,
    vectored_sub_in_place_pipeline: ComputePipeline,

    // Custom Pipelines
    custom_pipelines: Vec<ComputePipeline>,

    // Layouts for adding custom pipelines
    multi_op_pipeline_layout: PipelineLayout,
    multi_op_in_place_pipeline_layout: PipelineLayout,
    single_op_pipeline_layout: PipelineLayout,
    single_op_in_place_pipeline_layout: PipelineLayout,

    // WGPU variables
    device: Rc<Device>,
    queue: Rc<Queue>,
}

impl GPUMatrix {
    // Function to create the GPU Matrix witha defined shape
    fn with_shape(
        capacity: (u64, u64),
        data: Option<Vec<f32>>,
        transposed: bool,
        device: Rc<Device>,
        queue: Rc<Queue>,
    ) -> Self {
        let new_rows = capacity.0;
        let new_cols = capacity.1;

        // Create a buffer with the current data
        let buffer = match data {
            Some(data) => {
                // Create a buffer with the current data
                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Matrix Buffer"),
                    contents: bytemuck::cast_slice(&data),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                })
            }
            None => device.create_buffer(&BufferDescriptor {
                label: Some("Matrix Buffer"),
                mapped_at_creation: false,
                size: new_rows * new_cols * std::mem::size_of::<f32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }),
        };

        let dims = vec![new_rows as u32, new_cols as u32];

        // Create a buffer with the current dimensions
        let dimensions = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dims),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Create a buffer to keep track of the transpose status
        let transpose = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Transpose Buffer"),
            contents: bytemuck::cast_slice(&[transposed as u32]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Create a buffer for multiplying scalars with
        let scalar_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Matrix Scalar Buffer"),
            mapped_at_creation: false,
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Create a buffer for summing all the elements
        let sum_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Matrix Sum Buffer"),
            mapped_at_creation: false,
            size: ((new_rows * new_cols) as f32 / 256.0).ceil() as u64
                * std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Create bind groups
        let (bind_group_layout, bind_group) = create_buffer_bind_group!(
            &device,
            "Bind Group",
            (0, &buffer, Bbt::Storage { read_only: true }),
            (1, &dimensions, Bbt::Uniform),
            (2, &transpose, Bbt::Uniform),
            (3, &scalar_buffer, Bbt::Uniform),
            (4, &sum_buffer, Bbt::Storage { read_only: false })
        );

        // Create a writaboe bind group layout for matrix operations
        let (writable_bind_group_layout, writable_bind_group) = create_buffer_bind_group!(
            &device,
            "Writable Bind Group",
            (0, &buffer, Bbt::Storage { read_only: false }),
            (1, &dimensions, Bbt::Uniform),
            (2, &transpose, Bbt::Uniform),
            (3, &scalar_buffer, Bbt::Uniform)
        );

        // Create the pipeline layout for each of the operation pipelines
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Matrix Operations Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layout,
                &bind_group_layout,
                &writable_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Create a pipeline layout for in place operations
        let in_place_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Matrix In Place Operations Pipeline Layout"),
            bind_group_layouts: &[&writable_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the pipeline layout for a single operation
        let single_op_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Single Op Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &writable_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the pipeline layout for a single operation in place
        let single_op_in_place_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Single Op In Place Pipeline Layout"),
                bind_group_layouts: &[&writable_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create the pipeline layout for summing
        let sum_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Sum Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the compute pipeline for dotting
        let dot_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/dotting.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Dot Compute Pipeline"),
                module: &shader,
                layout: Some(&pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("dot_main"),
            })
        };

        // Create the compute pipeline for adding
        let add_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/adding.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Add Compute Pipeline"),
                module: &shader,
                layout: Some(&pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("add_main"),
            })
        };

        // Create the pipeline for adding in place
        let add_in_place_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/add_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Add In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("add_in_place_main"),
            })
        };

        // Create the pipeline for adding a scalar in place
        let add_scalar_in_place_pipeline = {
            let shader =
                device.create_shader_module(include_wgsl!("shaders/add_scalar_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Add Scalar In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&single_op_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("add_scalar_in_place_main"),
            })
        };

        // Create the compute pipeline for vectored adding
        let vectored_add_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/vectored_add.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Vectored Add Compute Pipeline"),
                module: &shader,
                layout: Some(&pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("vectored_add_main"),
            })
        };

        // Create the compute pipeline for vectored adding in place
        let vectored_add_in_place_pipeline = {
            let shader =
                device.create_shader_module(include_wgsl!("shaders/vectored_add_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Vectored Add In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("vectored_add_in_place_main"),
            })
        };

        // Create the compute pipeline for subtracting
        let sub_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/subtracting.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Sub Compute Pipeline"),
                module: &shader,
                layout: Some(&pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("sub_main"),
            })
        };

        // Create the compute pipeline for subtracting in place
        let sub_in_place_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/sub_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Sub In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("sub_in_place_main"),
            })
        };

        // Create the pipeline for adding a scalar in place
        let sub_scalar_in_place_pipeline = {
            let shader =
                device.create_shader_module(include_wgsl!("shaders/sub_scalar_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Sub Scalar In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&single_op_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("sub_scalar_in_place_main"),
            })
        };

        // Create the compute pipeline for vectored subtracting
        let vectored_sub_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/vectored_sub.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Vectored Sub Compute Pipeline"),
                module: &shader,
                layout: Some(&pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("vectored_sub_main"),
            })
        };

        // Create the compute pipeline for the vectored subtracing in place
        let vectored_sub_in_place_pipeline = {
            let shader =
                device.create_shader_module(include_wgsl!("shaders/vectored_sub_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Vectored Sub In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("vectored_sub_in_place_main"),
            })
        };

        // Create the compute pipeline for multiplying by scalar
        let mult_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/mult.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Mult Compute Pipeline"),
                module: &shader,
                layout: Some(&single_op_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("mult_main"),
            })
        };

        // Create the compute pipeline for multiplying in place by a scalar
        let mult_in_place_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/mult_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Mult In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&single_op_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("mult_in_place_main"),
            })
        };

        // Create the compute pipeline for element-wise multiplication
        let elem_mult_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/elem_mult.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Elem Mult Compute Pipeline"),
                module: &shader,
                layout: Some(&pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("elem_mult_main"),
            })
        };

        // Create the compute pipeline for element-wise multiplication in place
        let elem_mult_in_place_pipeline = {
            let shader =
                device.create_shader_module(include_wgsl!("shaders/elem_mult_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Elem Mult In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("elem_mult_in_place_main"),
            })
        };

        // Create the compute pipeline for exponenting the matrix
        let exp_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/exp.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Exp Compute Pipeline"),
                module: &shader,
                layout: Some(&single_op_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("exp_main"),
            })
        };

        // Create the compute pipeline for exponenting the matrix in place
        let exp_in_place_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/exp_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Exp In Place Compute Pipeline"),
                module: &shader,
                layout: Some(&single_op_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("exp_in_place_main"),
            })
        };

        // Create the compute pipeline for summing the matrix
        let sum_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/sum.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Sum Compute Pipeline"),
                module: &shader,
                layout: Some(&sum_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("sum_main"),
            })
        };

        GPUMatrix {
            rows: new_rows,
            cols: new_cols,
            data: buffer,
            scalar_buffer,
            sum_buffer,
            transpose: transposed,
            transpose_buffer: transpose,
            device,
            queue,
            bind_group,
            writable_bind_group,
            dot_pipeline,
            add_pipeline,
            add_in_place_pipeline,
            add_scalar_in_place_pipeline,
            sub_pipeline,
            sub_in_place_pipeline,
            sub_scalar_in_place_pipeline,
            mult_pipeline,
            mult_in_place_pipeline,
            elem_mult_pipeline,
            elem_mult_in_place_pipeline,
            exp_pipeline,
            exp_in_place_pipeline,
            sum_pipeline,
            vectored_add_pipeline,
            vectored_add_in_place_pipeline,
            vectored_sub_pipeline,
            vectored_sub_in_place_pipeline,
            custom_pipelines: Vec::new(),
            multi_op_pipeline_layout: pipeline_layout,
            multi_op_in_place_pipeline_layout: in_place_pipeline_layout,
            single_op_pipeline_layout,
            single_op_in_place_pipeline_layout,
        }
    }
}

/// Matrix that can have a defined shape on the gpu or the cpu
///
/// # Variants
///
/// * `CPU` - CPU stored and computed matrix
/// * `GPU` - GPU stored and computed matrix
#[allow(private_interfaces)]
#[derive(Debug)]
pub enum Matrix {
    CPU(CPUMatrix),
    GPU(GPUMatrix),
}

impl Matrix {
    /// Creates a matrix filled with zeros with a defined shape
    ///
    /// # Arguments
    ///
    /// * `capacity` - tuple defining the shape of the matrix in terms of rows and columns
    ///
    /// # Returns
    ///
    /// `Matrix::CPU` of shape `capacity` filled with zeros
    pub fn with_shape(capacity: (usize, usize)) -> Self {
        let rows = capacity.0;
        let cols = capacity.1;

        let data = vec![0.0; rows * cols];

        Matrix::CPU(CPUMatrix {
            rows,
            cols,
            data,
            transpose: false,
        })
    }

    /// Creates a matrix filled with random numbers from 0 to 1 with a defined shape
    ///
    /// # Arguments
    ///
    /// * `capacity` - tuple defining the shape of the matrix in terms of rows and columns
    ///
    /// # Returns
    ///
    /// `Matrix::CPU` of shape `capacity` filled with random numbers
    pub fn rand_with_shape(capacity: (usize, usize)) -> Self {
        let rows = capacity.0;
        let cols = capacity.1;

        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            data.push(rand::random_range(0.0..=1.0));
        }

        Matrix::CPU(CPUMatrix {
            rows,
            cols,
            data,
            transpose: false,
        })
    }

    /// Gets the number of rows in the matrix
    ///
    /// # Returns
    ///
    /// The number of rows in the matrix
    pub fn rows(&self) -> usize {
        match self {
            Matrix::CPU(cpu_matrix) => cpu_matrix.rows,
            Matrix::GPU(gpu_matrix) => gpu_matrix.rows as usize,
        }
    }

    /// Gets the number of columns in the matrix
    ///
    /// # Returns
    ///
    /// The number of columns in the matrix
    pub fn cols(&self) -> usize {
        match self {
            Matrix::CPU(cpu_matrix) => cpu_matrix.cols,
            Matrix::GPU(gpu_matrix) => gpu_matrix.cols as usize,
        }
    }

    /// Checks if the current matrix is a vector
    ///
    /// # Returns
    ///
    /// `true` if the matrix has either 1 row or 1 column, but not both
    pub fn is_vector(&self) -> bool {
        let (rows, cols) = match self {
            Matrix::CPU(cpu_matrix) => (cpu_matrix.rows, cpu_matrix.cols),
            Matrix::GPU(gpu_matrix) => (gpu_matrix.rows as usize, gpu_matrix.cols as usize),
        };

        (rows == 1 || cols == 1) && !(rows == cols)
    }

    /// Writes a scalar to the gpu variants scalar buffer
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if it was successful and `MatrixVariantError` if it failed
    pub fn write_to_scalar(&self, value: f32) -> Result<(), MatrixVariantError> {
        match self {
            Matrix::CPU(_) => Err(MatrixVariantError(String::from(
                "Matrix is not of the GPU variant",
            ))),
            Matrix::GPU(gpu_matrix) => {
                gpu_matrix.queue.write_buffer(
                    &gpu_matrix.scalar_buffer,
                    0,
                    bytemuck::cast_slice(&[value]),
                );

                Ok(())
            }
        }
    }

    /// Gets a reference to the device being used for this matrix
    ///
    /// # Returns
    ///
    /// `Result` with a refernce of the device if successfull or `MatrixVariantError` if not
    pub fn device(&self) -> Result<&Rc<Device>, MatrixVariantError> {
        match self {
            Matrix::CPU(_) => Err(MatrixVariantError(String::from(
                "Matrix CPU does not have a device",
            ))),
            Matrix::GPU(gpu_matrix) => Ok(&gpu_matrix.device),
        }
    }

    /// Gets a refernce to the queue being used for this matrix
    ///
    /// # Returns
    ///
    /// `Result` with a reference of the queue if successfull or `MatrixVariantError` if not
    pub fn queue(&self) -> Result<&Rc<Queue>, MatrixVariantError> {
        match self {
            Matrix::CPU(_) => Err(MatrixVariantError(String::from(
                "Matrix CPU does not have a queue",
            ))),
            Matrix::GPU(gpu_matrix) => Ok(&gpu_matrix.queue),
        }
    }

    /// Consumes the `Matrix::CPU` and converts it into a `Matrix::GPU`
    ///
    /// # Arguments
    ///
    /// * `device` - WGPU device to use for matrix operations
    /// * `queue` - WGPU queue to use for matrix operations
    ///
    /// # Returns
    ///
    /// `Matrix::GPU` with the data moved from self
    pub fn buf(self, device: Rc<Device>, queue: Rc<Queue>) -> Self {
        match self {
            Matrix::CPU(CPUMatrix {
                rows,
                cols,
                data,
                transpose,
            }) => Matrix::GPU(GPUMatrix::with_shape(
                (rows as u64, cols as u64),
                Some(data),
                transpose,
                device,
                queue,
            )),
            Matrix::GPU(_) => self,
        }
    }

    /// Consumes the `Matrix::GPU` and converts it to a `Matrix::CPU`
    ///
    /// # Returns
    ///
    /// `Matrix::CPU` with the data from self
    pub fn debuf(self) -> Self {
        match self {
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                data,
                transpose,
                device,
                queue,
                ..
            }) => {
                let values = {
                    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("Matrix Debuf encoder"),
                    });

                    let values_buffer = read_buffer(
                        &data,
                        rows * cols * std::mem::size_of::<f32>() as u64,
                        &device,
                        &mut encoder,
                    );

                    queue.submit(Some(encoder.finish()));

                    get_buffer(&values_buffer, &device)
                };

                Matrix::CPU(CPUMatrix {
                    rows: rows as usize,
                    cols: cols as usize,
                    data: values,
                    transpose,
                })
            }
            Matrix::CPU(_) => self,
        }
    }

    /// Gets a copy of the matrix as a CPU matrix
    ///
    /// # Returns
    ///
    /// Matrix copied as a CPU Matrix
    pub fn get_copy(&self) -> Self {
        match self {
            Matrix::CPU(cpu_matrix) => Matrix::CPU(cpu_matrix.clone()),
            Matrix::GPU(gpu_matrix) => {
                let values = {
                    let mut encoder =
                        gpu_matrix
                            .device
                            .create_command_encoder(&CommandEncoderDescriptor {
                                label: Some("Matrix Debuf encoder"),
                            });

                    let values_buffer = read_buffer(
                        &gpu_matrix.data,
                        gpu_matrix.rows * gpu_matrix.cols * std::mem::size_of::<f32>() as u64,
                        &gpu_matrix.device,
                        &mut encoder,
                    );

                    gpu_matrix.queue.submit(Some(encoder.finish()));

                    get_buffer(&values_buffer, &gpu_matrix.device)
                };

                Matrix::CPU(CPUMatrix {
                    rows: gpu_matrix.rows as usize,
                    cols: gpu_matrix.cols as usize,
                    data: values,
                    transpose: gpu_matrix.transpose,
                })
            }
        }
    }

    /// Performs the dot product with the matrix described in `other`
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the dot product with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the dot product was successful and `Err` if the dot product failed
    pub fn dot(&self, other: &Matrix) -> Result<Matrix, MatrixDotError> {
        match self {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                let (b_rows, b_cols) = match other {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => {
                        return Err(MatrixDotError(String::from("Matrix Variants do not match")));
                    }
                };

                // before getting the data make sure to check if the dot product is possible
                if *cols != *b_rows {
                    return Err(MatrixDotError(String::from(
                        "Columns of matrix 1 do not match rows of matrix 2",
                    )));
                }

                let (result_rows, result_cols) = (*rows, *b_cols);
                let mut output_mat = Matrix::with_shape((result_rows, result_cols));
                for i in 0..result_rows {
                    for j in 0..result_cols {
                        for k in 0..*cols {
                            output_mat[(i, j)] += self[(i, k)] * other[(k, j)];
                        }
                    }
                }

                Ok(output_mat)
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                queue,
                bind_group,
                dot_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixDotError(String::from("Matrix Variants do not match"))),
                };

                // before getting the data make sure to check if the dot product is possible
                if *cols != *b_rows {
                    return Err(MatrixDotError(String::from(
                        "Columns of matrix 1 do not match rows of matrix 2",
                    )));
                }

                // Create the output matrix to use as the return matrix
                let output = GPUMatrix::with_shape(
                    (*rows, *b_cols),
                    None,
                    false,
                    device.clone(),
                    queue.clone(),
                );

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Dot Product Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *b_cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Dot Product Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&dot_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);
                    compute_pass.set_bind_group(2, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Performs the dot product with the matrices descibed in `source1` and `source2`
    /// and stores the result in the `destination` matrix
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `source1` - first matrix in the dot product
    /// * `source2` - second matrix in the dot product
    /// * `destination` - destination matrix in the dot product
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the dot product was successful and `Err` if the dot product failed
    pub fn dot_into(
        source1: &Matrix,
        source2: &Matrix,
        destination: &mut Matrix,
    ) -> Result<(), MatrixDotError> {
        match source1 {
            Matrix::CPU(source1_cpu) => {
                let (source2_rows, source2_cols) = match source2 {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (*rows, *cols),
                    _ => {
                        return Err(MatrixDotError(String::from("Matrix Variants do not match")));
                    }
                };

                // before getting the data make sure to check if the dot product is possible
                if source1_cpu.cols != source2_rows {
                    return Err(MatrixDotError(String::from(
                        "Columns of matrix 1 do not match rows of matrix 2",
                    )));
                }

                // Make sure the destination matrix rows and columns are correct
                let (destination_rows, destination_cols) = match destination {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                        if *rows == source1_cpu.rows && *cols == source2_cols {
                            (*rows, *cols)
                        } else {
                            return Err(MatrixDotError(String::from(
                                "Destination dimensions don't match required dimensions",
                            )));
                        }
                    }
                    Matrix::GPU(_) => {
                        return Err(MatrixDotError(String::from(
                            "Destincation Matrix Variant does not match",
                        )));
                    }
                };

                // let mut output_mat = Matrix::with_shape((result_rows, result_cols));
                for i in 0..destination_rows {
                    for j in 0..destination_cols {
                        for k in 0..source1_cpu.cols {
                            destination[(i, j)] += source1[(i, k)] * source2[(k, j)];
                        }
                    }
                }

                Ok(())
            }
            Matrix::GPU(source1_gpu) => {
                let (source2_rows, source2_cols, source2_bind_group) = match source2 {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (*rows, *cols, bind_group),
                    _ => {
                        return Err(MatrixDotError(String::from("Matrix Variants do not match")));
                    }
                };

                // before getting the data make sure to check if the dot product is possible
                if source1_gpu.cols != source2_rows {
                    return Err(MatrixDotError(String::from(
                        "Columns of matrix 1 do not match rows of matrix 2",
                    )));
                }

                // Make sure the destination matrix rows and columns are correct
                let destination_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        writable_bind_group,
                        ..
                    }) => {
                        if *rows == source1_gpu.rows && *cols == source2_cols {
                            &*writable_bind_group
                        } else {
                            return Err(MatrixDotError(String::from(
                                "Destination dimensions don't match required dimensions",
                            )));
                        }
                    }
                    Matrix::CPU(_) => {
                        return Err(MatrixDotError(String::from(
                            "Destincation Matrix Variant does not match",
                        )));
                    }
                };

                let mut encoder =
                    source1_gpu
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Dot Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (source1_gpu.rows as u32, source2_cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Dot Product Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&source1_gpu.dot_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &source1_gpu.bind_group, &[]);
                    compute_pass.set_bind_group(1, source2_bind_group, &[]);
                    compute_pass.set_bind_group(2, destination_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                source1_gpu.device.poll(Maintain::Wait);
                source1_gpu.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs an addition with the matrix described in `other`
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the addition with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the addition was successful and `Err` if the addition failed
    pub fn add(&self, other: &Matrix) -> Result<Matrix, MatrixAddError> {
        match self {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                let (b_rows, b_cols) = match other {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => {
                        return Err(MatrixAddError(String::from("Matrix Variants do not match")));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixAddError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                let mut output_mat = Matrix::with_shape((*rows, *cols));

                for i in 0..*rows {
                    for j in 0..*cols {
                        output_mat[(i, j)] = self[(i, j)] + other[(i, j)];
                    }
                }

                Ok(output_mat)
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                transpose,
                queue,
                bind_group,
                add_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixAddError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                // Create the output matrix to store add into
                let output = GPUMatrix::with_shape(
                    (*rows, *cols),
                    None,
                    *transpose,
                    device.clone(),
                    queue.clone(),
                );

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Add Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Add Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&add_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);
                    compute_pass.set_bind_group(2, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Performs an addition in place with the scalar value
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the addition with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the addition was successful and `Err` if the addition failed
    pub fn add_scalar_in_place(&mut self, scalar: f32) -> Result<(), MatrixAddError> {
        match self {
            Matrix::CPU(_) => {}
            Matrix::GPU(gpu_matrix) => {
                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Add Scalar In Place Command Encoder"),
                        });

                gpu_matrix.queue.write_buffer(
                    &gpu_matrix.scalar_buffer,
                    0,
                    bytemuck::cast_slice(&[scalar]),
                );

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Add Scalar In Place Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.add_scalar_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                return Ok(());
            }
        }

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self[(i, j)] += scalar;
            }
        }

        Ok(())
    }

    /// Performs an addition in place with the matrix described in `other`
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the addition with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the addition was successful and `Err` if the addition failed
    pub fn add_in_place(&mut self, other: &Matrix) -> Result<(), MatrixAddError> {
        match self {
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                queue,
                writable_bind_group,
                add_in_place_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixAddError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Add In Place Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Add In Place Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&add_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &*writable_bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                return Ok(());
            }
            Matrix::CPU(_) => {}
        }

        let (b_rows, b_cols) = match other {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
            _ => {
                return Err(MatrixAddError(String::from("Matrix Variants do not match")));
            }
        };

        if self.rows() != *b_rows || self.cols() != *b_cols {
            return Err(MatrixAddError(String::from(
                "Matrix Rows and Colums do not match",
            )));
        }

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self[(i, j)] += other[(i, j)];
            }
        }

        Ok(())
    }

    /// Performs the addition with the matrices descibed in `source1` and `source2`
    /// and stores the result in the `destination` matrix
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `source1` - first matrix in the addition
    /// * `source2` - second matrix in the addition
    /// * `destination` - destination matrix in the addition
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the addition was successful and `Err` if the addition failed
    pub fn add_into(
        source1: &Matrix,
        source2: &Matrix,
        destination: &mut Matrix,
    ) -> Result<(), MatrixAddError> {
        match source1 {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                let (b_rows, b_cols) = match source2 {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => {
                        return Err(MatrixAddError(String::from("Matrix Variants do not match")));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixAddError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                // let mut output_mat = Matrix::with_shape((*rows, *cols));
                match destination {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                        if *rows != *b_rows || *cols != *b_cols {
                            return Err(MatrixAddError(String::from(
                                "Destination Matrix Dimensions do not match",
                            )));
                        }
                    }
                    Matrix::GPU(_) => {
                        return Err(MatrixAddError(String::from(
                            "Destination Matrix Variant does not match",
                        )));
                    }
                }

                for i in 0..*rows {
                    for j in 0..*cols {
                        destination[(i, j)] = source1[(i, j)] + source2[(i, j)];
                    }
                }

                Ok(())
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                queue,
                bind_group,
                add_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match source2 {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixAddError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                // Create the output matrix to store add into
                let output_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        writable_bind_group,
                        ..
                    }) => {
                        if *rows != *b_rows || *cols != *b_cols {
                            return Err(MatrixAddError(String::from(
                                "Destination Dimensions do not match",
                            )));
                        } else {
                            &*writable_bind_group
                        }
                    }
                    Matrix::CPU(_) => {
                        return Err(MatrixAddError(String::from(
                            "Destination Variants does not match",
                        )));
                    }
                };

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Add Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Add Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&add_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);
                    compute_pass.set_bind_group(2, output_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs a vectored addition on the current matrix given the `other` matrix
    /// The other matrix must be a vector with either the same number of rows or columns as the
    /// current matrix. If the rows and columns are the same it will default to the orientation that
    /// the `other` matrix is in.
    ///
    /// # Arguments
    ///
    /// * `other` - vector to use to add to this matrix
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the addition was successful and `Err` if the addition failed
    pub fn vectored_add(&self, other: &Matrix) -> Result<Matrix, MatrixAddError> {
        match self {
            Matrix::CPU(cpu_matrix) => {
                let (other_rows, other_cols) = match other {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !other.is_vector() {
                    return Err(MatrixAddError(String::from("Matrix 2 is not a Vector")));
                }

                let mut output = Matrix::with_shape((cpu_matrix.rows, cpu_matrix.cols));

                // Check the rows first
                if *other_rows == cpu_matrix.rows {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            output[(i, j)] = self[(i, j)] + other[(i, 0)];
                        }
                    }
                // Check the columns before checking the transposes
                } else if *other_cols == cpu_matrix.cols {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            output[(i, j)] = self[(i, j)] + other[(0, j)];
                        }
                    }
                } else if *other_rows == cpu_matrix.cols {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            output[(i, j)] = self[(i, j)] + other[(j, 0)];
                        }
                    }
                } else if *other_cols == cpu_matrix.rows {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            output[(i, j)] = self[(i, j)] + other[(0, i)];
                        }
                    }
                } else {
                    return Err(MatrixAddError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                Ok(output)
            }
            Matrix::GPU(gpu_matrix) => {
                let (other_rows, other_cols, other_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !other.is_vector() {
                    return Err(MatrixAddError(String::from("Matrix 2 is not a Vector")));
                }

                if !(*other_rows == gpu_matrix.rows
                    || *other_cols == gpu_matrix.cols
                    || *other_rows == gpu_matrix.cols
                    || *other_cols == gpu_matrix.rows)
                {
                    return Err(MatrixAddError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                // Create the output matrix to store the add into
                let output = GPUMatrix::with_shape(
                    (gpu_matrix.rows, gpu_matrix.cols),
                    None,
                    gpu_matrix.transpose,
                    gpu_matrix.device.clone(),
                    gpu_matrix.queue.clone(),
                );

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Vectored Add Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Vectored Add Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.vectored_add_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, other_bind_group, &[]);
                    compute_pass.set_bind_group(2, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Performs a vectored addition on the current matrix given the `other` matrix in place
    /// The other matrix must be a vector with either the same number of rows or columns as the
    /// current matrix. If the rows and columns are the same it will default to the orientation that
    /// the `other` matrix is in.
    ///
    /// # Arguments
    ///
    /// * `other` - vector to use to add to this matrix
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the addition was successful and `Err` if the addition failed
    pub fn vectored_add_in_place(&mut self, other: &Matrix) -> Result<(), MatrixAddError> {
        match self {
            Matrix::CPU(_) => {}
            Matrix::GPU(gpu_matrix) => {
                let (other_rows, other_cols, other_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !other.is_vector() {
                    return Err(MatrixAddError(String::from("Matrix 2 is not a Vector")));
                }

                if !(*other_rows == gpu_matrix.rows
                    || *other_cols == gpu_matrix.cols
                    || *other_rows == gpu_matrix.cols
                    || *other_cols == gpu_matrix.rows)
                {
                    return Err(MatrixAddError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Vectored Add In Place Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Vectored Add In Place Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.vectored_add_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.writable_bind_group, &[]);
                    compute_pass.set_bind_group(1, other_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                return Ok(());
            }
        }

        let (other_rows, other_cols) = match other {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
            _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
        };

        // Check if the other matrix is a vector with the correct size
        // Check if the other is a vector in the first place
        if !other.is_vector() {
            return Err(MatrixAddError(String::from("Matrix 2 is not a Vector")));
        }

        // Check the rows first
        if *other_rows == self.rows() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    self[(i, j)] += other[(i, 0)];
                }
            }
        // Check the columns before checking the transposes
        } else if *other_cols == self.cols() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    self[(i, j)] += other[(0, j)];
                }
            }
        } else if *other_rows == self.cols() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    self[(i, j)] += other[(j, 0)];
                }
            }
        } else if *other_cols == self.rows() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    self[(i, j)] += other[(0, i)];
                }
            }
        } else {
            return Err(MatrixAddError(String::from(
                "Vector Dimensions do no match matrix",
            )));
        }

        Ok(())
    }

    /// Performs a vector addition with the matrix described in `source` and the vector described in `vector`
    /// and stored the output into `destination`
    ///
    /// # Arguments
    ///
    /// * `source` - source matrix to do the vector addition on
    /// * `vector` - vector to add to the source matrix
    /// * `destination` - matrix where the output of the computation is stored
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the addition was successful and `Err` if the addition failed
    pub fn vectored_add_into(
        source: &Matrix,
        vector: &Matrix,
        destination: &mut Matrix,
    ) -> Result<(), MatrixAddError> {
        match source {
            Matrix::CPU(cpu_matrix) => {
                let (other_rows, other_cols) = match vector {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !vector.is_vector() {
                    return Err(MatrixAddError(String::from("Matrix 2 is not a Vector")));
                }

                match destination {
                    Matrix::CPU(_) => {}
                    Matrix::GPU(_) => {
                        return Err(MatrixAddError(String::from(
                            "Vector Variant does not match",
                        )));
                    }
                }

                // Check the rows first
                if *other_rows == cpu_matrix.rows {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            destination[(i, j)] = source[(i, j)] + vector[(i, 0)];
                        }
                    }
                // Check the columns before checking the transposes
                } else if *other_cols == cpu_matrix.cols {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            destination[(i, j)] = source[(i, j)] + vector[(0, j)];
                        }
                    }
                } else if *other_rows == cpu_matrix.cols {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            destination[(i, j)] = source[(i, j)] + vector[(j, 0)];
                        }
                    }
                } else if *other_cols == cpu_matrix.rows {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            destination[(i, j)] = source[(i, j)] + vector[(0, i)];
                        }
                    }
                } else {
                    return Err(MatrixAddError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                Ok(())
            }
            Matrix::GPU(gpu_matrix) => {
                let (other_rows, other_cols, other_bind_group) = match vector {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !vector.is_vector() {
                    return Err(MatrixAddError(String::from("Matrix 2 is not a Vector")));
                }

                let output_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        writable_bind_group,
                        ..
                    }) => &*writable_bind_group,
                    Matrix::CPU(_) => {
                        return Err(MatrixAddError(String::from(
                            "Vector Variant does not match",
                        )));
                    }
                };

                if !(*other_rows == gpu_matrix.rows
                    || *other_cols == gpu_matrix.cols
                    || *other_rows == gpu_matrix.cols
                    || *other_cols == gpu_matrix.rows)
                {
                    return Err(MatrixAddError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                // Create the output matrix to store the add into

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Vectored Add Into Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Vectored Add Into Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.vectored_add_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, other_bind_group, &[]);
                    compute_pass.set_bind_group(2, output_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs a subtraction with the matrix described in `other`
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the subtraction with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the subtraction was successful and `Err` if the subtraction failed
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, MatrixSubError> {
        match self {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                let (b_rows, b_cols) = match other {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => {
                        return Err(MatrixSubError(String::from("Matrix Variants do not match")));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixSubError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                let mut output_mat = Matrix::with_shape((*rows, *cols));

                for i in 0..*rows {
                    for j in 0..*cols {
                        output_mat[(i, j)] = self[(i, j)] - other[(i, j)];
                    }
                }

                Ok(output_mat)
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                transpose,
                queue,
                bind_group,
                sub_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixSubError(String::from("Matrix Variants do not match"))),
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixSubError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                // Create the output matrix to store add into
                let output = GPUMatrix::with_shape(
                    (*rows, *cols),
                    None,
                    *transpose,
                    device.clone(),
                    queue.clone(),
                );

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Sub Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Sub Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&sub_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);
                    compute_pass.set_bind_group(2, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Performs an subtraction in place with the matrix described in `other`
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the subtraction with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the subtraction was successful and `Err` if the subtraction failed
    pub fn sub_in_place(&mut self, other: &Matrix) -> Result<(), MatrixAddError> {
        match self {
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                queue,
                writable_bind_group,
                sub_in_place_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixAddError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Sub In Place Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Sub In Place Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&sub_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &*writable_bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                return Ok(());
            }
            Matrix::CPU(_) => {}
        }

        let (b_rows, b_cols) = match other {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
            _ => {
                return Err(MatrixAddError(String::from("Matrix Variants do not match")));
            }
        };

        if self.rows() != *b_rows || self.cols() != *b_cols {
            return Err(MatrixAddError(String::from(
                "Matrix Rows and Colums do not match",
            )));
        }

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self[(i, j)] -= other[(i, j)];
            }
        }

        Ok(())
    }

    /// Performs an subtraction in place with the scalar value
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the subtraction with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the subtraction was successful and `Err` if the subtraction failed
    pub fn sub_scalar_in_place(&mut self, scalar: f32) -> Result<(), MatrixAddError> {
        match self {
            Matrix::CPU(_) => {}
            Matrix::GPU(gpu_matrix) => {
                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Sub Scalar In Place Command Encoder"),
                        });

                gpu_matrix.queue.write_buffer(
                    &gpu_matrix.scalar_buffer,
                    0,
                    bytemuck::cast_slice(&[scalar]),
                );

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Sub Scalar In Place Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.sub_scalar_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                return Ok(());
            }
        }

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self[(i, j)] -= scalar;
            }
        }

        Ok(())
    }

    /// Performs the subtracion with the matrices descibed in `source1` and `source2`
    /// and stores the result in the `destination` matrix
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `source1` - first matrix in the subtracion
    /// * `source2` - second matrix in the subtracion
    /// * `destination` - destination matrix in the subtracion
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the subtracion was successful and `Err` if the subtracion failed
    pub fn sub_into(
        source1: &Matrix,
        source2: &Matrix,
        destination: &mut Matrix,
    ) -> Result<(), MatrixSubError> {
        match source1 {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                let (b_rows, b_cols) = match source2 {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => {
                        return Err(MatrixSubError(String::from("Matrix Variants do not match")));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixSubError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                // let mut output_mat = Matrix::with_shape((*rows, *cols));
                match destination {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                        if *rows != *b_rows || *cols != *b_cols {
                            return Err(MatrixSubError(String::from(
                                "Destination Matrix Dimensions do not match",
                            )));
                        }
                    }
                    Matrix::GPU(_) => {
                        return Err(MatrixSubError(String::from(
                            "Destination Matrix Variant does not match",
                        )));
                    }
                }

                for i in 0..*rows {
                    for j in 0..*cols {
                        destination[(i, j)] = source1[(i, j)] - source2[(i, j)];
                    }
                }

                Ok(())
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                queue,
                bind_group,
                sub_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match source2 {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixSubError(String::from("Matrix Variants do not match"))),
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixSubError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                // Create the output matrix to store add into
                let output_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        writable_bind_group,
                        ..
                    }) => {
                        if *rows != *b_rows || *cols != *b_cols {
                            return Err(MatrixSubError(String::from(
                                "Destination Dimensions do not match",
                            )));
                        } else {
                            &*writable_bind_group
                        }
                    }
                    Matrix::CPU(_) => {
                        return Err(MatrixSubError(String::from(
                            "Destination Variants does not match",
                        )));
                    }
                };

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Add Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Add Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&sub_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);
                    compute_pass.set_bind_group(2, output_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs a vectored subtraction on the current matrix given the `other` matrix
    /// The other matrix must be a vector with either the same number of rows or columns as the
    /// current matrix. If the rows and columns are the same it will default to the orientation that
    /// the `other` matrix is in.
    ///
    /// # Arguments
    ///
    /// * `other` - vector to use to subtract from this matrix
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the subtraction was successful and `Err` if the subtraction failed
    pub fn vectored_sub(&self, other: &Matrix) -> Result<Matrix, MatrixSubError> {
        match self {
            Matrix::CPU(cpu_matrix) => {
                let (other_rows, other_cols) = match other {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => return Err(MatrixSubError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !other.is_vector() {
                    return Err(MatrixSubError(String::from("Matrix 2 is not a Vector")));
                }

                let mut output = Matrix::with_shape((cpu_matrix.rows, cpu_matrix.cols));

                // Check the rows first
                if *other_rows == cpu_matrix.rows {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            output[(i, j)] = self[(i, j)] - other[(i, 0)];
                        }
                    }
                // Check the columns before checking the transposes
                } else if *other_cols == cpu_matrix.cols {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            output[(i, j)] = self[(i, j)] - other[(0, j)];
                        }
                    }
                } else if *other_rows == cpu_matrix.cols {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            output[(i, j)] = self[(i, j)] - other[(j, 0)];
                        }
                    }
                } else if *other_cols == cpu_matrix.rows {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            output[(i, j)] = self[(i, j)] - other[(0, i)];
                        }
                    }
                } else {
                    return Err(MatrixSubError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                Ok(output)
            }
            Matrix::GPU(gpu_matrix) => {
                let (other_rows, other_cols, other_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixSubError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !other.is_vector() {
                    return Err(MatrixSubError(String::from("Matrix 2 is not a Vector")));
                }

                if !(*other_rows == gpu_matrix.rows
                    || *other_cols == gpu_matrix.cols
                    || *other_rows == gpu_matrix.cols
                    || *other_cols == gpu_matrix.rows)
                {
                    return Err(MatrixSubError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                // Create the output matrix to store the add into
                let output = GPUMatrix::with_shape(
                    (gpu_matrix.rows, gpu_matrix.cols),
                    None,
                    gpu_matrix.transpose,
                    gpu_matrix.device.clone(),
                    gpu_matrix.queue.clone(),
                );

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Vectored Sub Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Vectored Sub Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.vectored_sub_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, other_bind_group, &[]);
                    compute_pass.set_bind_group(2, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Performs a vectored subtraction on the current matrix given the `other` matrix in place
    /// The other matrix must be a vector with either the same number of rows or columns as the
    /// current matrix. If the rows and columns are the same it will default to the orientation that
    /// the `other` matrix is in.
    ///
    /// # Arguments
    ///
    /// * `other` - vector to use to add to this matrix
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the subtraction was successful and `Err` if the subtraction failed
    pub fn vectored_sub_in_place(&mut self, other: &Matrix) -> Result<(), MatrixAddError> {
        match self {
            Matrix::CPU(_) => {}
            Matrix::GPU(gpu_matrix) => {
                let (other_rows, other_cols, other_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !other.is_vector() {
                    return Err(MatrixAddError(String::from("Matrix 2 is not a Vector")));
                }

                if !(*other_rows == gpu_matrix.rows
                    || *other_cols == gpu_matrix.cols
                    || *other_rows == gpu_matrix.cols
                    || *other_cols == gpu_matrix.rows)
                {
                    return Err(MatrixAddError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Vectored Sub In Place Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Vectored Sub In Place Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.vectored_sub_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.writable_bind_group, &[]);
                    compute_pass.set_bind_group(1, other_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                return Ok(());
            }
        }

        let (other_rows, other_cols) = match other {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
            _ => return Err(MatrixAddError(String::from("Matrix Variants do not match"))),
        };

        // Check if the other matrix is a vector with the correct size
        // Check if the other is a vector in the first place
        if !other.is_vector() {
            return Err(MatrixAddError(String::from("Matrix 2 is not a Vector")));
        }

        // Check the rows first
        if *other_rows == self.rows() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    self[(i, j)] -= other[(i, 0)];
                }
            }
        // Check the columns before checking the transposes
        } else if *other_cols == self.cols() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    self[(i, j)] -= other[(0, j)];
                }
            }
        } else if *other_rows == self.cols() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    self[(i, j)] -= other[(j, 0)];
                }
            }
        } else if *other_cols == self.rows() {
            for i in 0..self.rows() {
                for j in 0..self.cols() {
                    self[(i, j)] -= other[(0, i)];
                }
            }
        } else {
            return Err(MatrixAddError(String::from(
                "Vector Dimensions do no match matrix",
            )));
        }

        Ok(())
    }

    /// Performs a vector subtraction with the matrix described in `source` and the vector described in `vector`
    /// and stored the output into `destination`
    ///
    /// # Arguments
    ///
    /// * `source` - source matrix to do the vector subtraction on
    /// * `vector` - vector to add to the source matrix
    /// * `destination` - matrix where the output of the computation is stored
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the subtraction was successful and `Err` if the subtraction failed
    pub fn vectored_sub_into(
        source: &Matrix,
        vector: &Matrix,
        destination: &mut Matrix,
    ) -> Result<(), MatrixSubError> {
        match source {
            Matrix::CPU(cpu_matrix) => {
                let (other_rows, other_cols) = match vector {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => return Err(MatrixSubError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !vector.is_vector() {
                    return Err(MatrixSubError(String::from("Matrix 2 is not a Vector")));
                }

                match destination {
                    Matrix::CPU(_) => {}
                    Matrix::GPU(_) => {
                        return Err(MatrixSubError(String::from(
                            "Vector Variant does not match",
                        )));
                    }
                }

                // Check the rows first
                if *other_rows == cpu_matrix.rows {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            destination[(i, j)] = source[(i, j)] - vector[(i, 0)];
                        }
                    }
                // Check the columns before checking the transposes
                } else if *other_cols == cpu_matrix.cols {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            destination[(i, j)] = source[(i, j)] - vector[(0, j)];
                        }
                    }
                } else if *other_rows == cpu_matrix.cols {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            destination[(i, j)] = source[(i, j)] - vector[(j, 0)];
                        }
                    }
                } else if *other_cols == cpu_matrix.rows {
                    for i in 0..cpu_matrix.rows {
                        for j in 0..cpu_matrix.cols {
                            destination[(i, j)] = source[(i, j)] - vector[(0, i)];
                        }
                    }
                } else {
                    return Err(MatrixSubError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                Ok(())
            }
            Matrix::GPU(gpu_matrix) => {
                let (other_rows, other_cols, other_bind_group) = match vector {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => return Err(MatrixSubError(String::from("Matrix Variants do not match"))),
                };

                // Check if the other matrix is a vector with the correct size
                // Check if the other is a vector in the first place
                if !vector.is_vector() {
                    return Err(MatrixSubError(String::from("Matrix 2 is not a Vector")));
                }

                let output_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        writable_bind_group,
                        ..
                    }) => &*writable_bind_group,
                    Matrix::CPU(_) => {
                        return Err(MatrixSubError(String::from(
                            "Vector Variant does not match",
                        )));
                    }
                };

                if !(*other_rows == gpu_matrix.rows
                    || *other_cols == gpu_matrix.cols
                    || *other_rows == gpu_matrix.cols
                    || *other_cols == gpu_matrix.rows)
                {
                    return Err(MatrixSubError(String::from(
                        "Vector Dimensions do no match matrix",
                    )));
                }

                // Create the output matrix to store the add into

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Vectored Sub Into Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Vectored Sub Into Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.vectored_sub_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, other_bind_group, &[]);
                    compute_pass.set_bind_group(2, output_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs a scalar multiplicaiton on the matrix and returns a new matrix as the result
    ///
    /// # Arguments
    ///
    /// * `scalar` - scalar value to multiply matrix by
    ///
    /// # Returns
    ///
    /// `Matrix` that has been multiplied by the value that has been specified
    pub fn mult(&self, scalar: f32) -> Result<Matrix, MatrixMultError> {
        match self {
            Matrix::CPU(cpu_matrix) => {
                let mut output = cpu_matrix.clone();
                output.data.iter_mut().for_each(|value| *value *= scalar);

                Ok(Matrix::CPU(output))
            }
            Matrix::GPU(gpu_matrix) => {
                let output = GPUMatrix::with_shape(
                    (gpu_matrix.rows, gpu_matrix.cols),
                    None,
                    gpu_matrix.transpose,
                    gpu_matrix.device.clone(),
                    gpu_matrix.queue.clone(),
                );

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Mult Command Encoder"),
                        });

                gpu_matrix.queue.write_buffer(
                    &gpu_matrix.scalar_buffer,
                    0,
                    bytemuck::cast_slice(&[scalar]),
                );

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Mult Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.mult_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Performs a scalar multiplicaiton on the matrix in place
    ///
    /// # Arguments
    ///
    /// * `scalar` - scalar value to multiply matrix by
    ///
    /// # Returns
    ///
    /// `Result` wiht `Ok` if successful
    pub fn mult_in_place(&mut self, scalar: f32) -> Result<(), MatrixMultError> {
        match self {
            Matrix::CPU(cpu_matrix) => {
                cpu_matrix
                    .data
                    .iter_mut()
                    .for_each(|value| *value *= scalar);

                Ok(())
            }
            Matrix::GPU(gpu_matrix) => {
                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Mult In Place Command Encoder"),
                        });

                gpu_matrix.queue.write_buffer(
                    &gpu_matrix.scalar_buffer,
                    0,
                    bytemuck::cast_slice(&[scalar]),
                );

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Mult In Place Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.mult_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs a scalar multiplicaiton on the matrix and stores the result in `destination`
    ///
    /// # Arguments
    ///
    /// * `source` - the source matrix to perform the multiplication on
    /// * `scalar` - scalar value to multiply matrix by
    /// * `destination` - the destination to write the new values to
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the multiplication was successful and `Err` if it failed
    pub fn mult_into(
        source: &Matrix,
        scalar: f32,
        destination: &mut Matrix,
    ) -> Result<(), MatrixMultError> {
        match source {
            Matrix::CPU(cpu_matrix) => {
                match destination {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                        if cpu_matrix.rows != *rows || cpu_matrix.cols != *cols {
                            return Err(MatrixMultError(String::from(
                                "Destination Dimensions do not match",
                            )));
                        }
                    }
                    Matrix::GPU(_) => {
                        return Err(MatrixMultError(String::from(
                            "Destination Variant does not match",
                        )));
                    }
                }

                for i in 0..cpu_matrix.rows {
                    for j in 0..cpu_matrix.cols {
                        destination[(i, j)] = source[(i, j)] * scalar;
                    }
                }

                Ok(())
            }
            Matrix::GPU(gpu_matrix) => {
                let destination_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        writable_bind_group,
                        ..
                    }) => {
                        if gpu_matrix.rows != *rows || gpu_matrix.cols != *cols {
                            return Err(MatrixMultError(String::from(
                                "Destination Dimensions do not match",
                            )));
                        } else {
                            &*writable_bind_group
                        }
                    }
                    Matrix::CPU(_) => {
                        return Err(MatrixMultError(String::from(
                            "Destination Variant does not match",
                        )));
                    }
                };

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Mult Command Encoder"),
                        });

                gpu_matrix.queue.write_buffer(
                    &gpu_matrix.scalar_buffer,
                    0,
                    bytemuck::cast_slice(&[scalar]),
                );

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Mult Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.mult_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, destination_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs an element-wise multiplication with the matrix described in `other`
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the element-wise multiplication with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the element-wise multiplication was successful and `Err` if the element-wise multiplication failed
    pub fn elem_mult(&self, other: &Matrix) -> Result<Matrix, MatrixMultError> {
        match self {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                let (b_rows, b_cols) = match other {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => {
                        return Err(MatrixMultError(String::from(
                            "Matrix Variants do not match",
                        )));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixMultError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                let mut output_mat = Matrix::with_shape((*rows, *cols));

                for i in 0..*rows {
                    for j in 0..*cols {
                        output_mat[(i, j)] = self[(i, j)] * other[(i, j)];
                    }
                }

                Ok(output_mat)
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                transpose,
                queue,
                bind_group,
                elem_mult_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => {
                        return Err(MatrixMultError(String::from(
                            "Matrix Variants do not match",
                        )));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixMultError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                // Create the output matrix to store add into
                let output = GPUMatrix::with_shape(
                    (*rows, *cols),
                    None,
                    *transpose,
                    device.clone(),
                    queue.clone(),
                );

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Elem Mult Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Elem Mult Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&elem_mult_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);
                    compute_pass.set_bind_group(2, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Performs an element-wise multiplication with the matrix described in `other` in place
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `other` - reference to another matrix to do the element-wise multiplication with
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the element-wise multiplication was successful and `Err` if the element-wise multiplication failed
    pub fn elem_mult_in_place(&mut self, other: &Matrix) -> Result<(), MatrixMultError> {
        match self {
            Matrix::CPU(_) => {}
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                queue,
                writable_bind_group,
                elem_mult_in_place_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => {
                        return Err(MatrixMultError(String::from(
                            "Matrix Variants do not match",
                        )));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixMultError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Elem Mult In Place Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Elem Mult In Place Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&elem_mult_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &*writable_bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                return Ok(());
            }
        }

        let (b_rows, b_cols) = match other {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
            _ => {
                return Err(MatrixMultError(String::from(
                    "Matrix Variants do not match",
                )));
            }
        };

        if self.rows() != *b_rows || self.cols() != *b_cols {
            return Err(MatrixMultError(String::from(
                "Matrix Rows and Colums do not match",
            )));
        }

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self[(i, j)] *= other[(i, j)];
            }
        }

        Ok(())
    }

    /// Performs an element-wise multiplication with the matrices described in `source1`
    /// and `source2` and stores it in `destination`
    /// If the matrix is a `Matrix::CPU` it will do a sequential computation
    /// If the matrix is a `Matrix::GPU` it will do a parallel computation
    ///
    /// # Arguments
    ///
    /// * `source1` - first matrix to do the element-wise multiplication with
    /// * `source2` - second matrix to do the element-wise multiplication with
    /// * `destination` - destination matrix to store the result of the element-wise multiplication in
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the element-wise multiplication was successful and `Err` if the element-wise multiplication failed
    pub fn elem_mult_into(
        source1: &Matrix,
        source2: &Matrix,
        destination: &mut Matrix,
    ) -> Result<(), MatrixMultError> {
        match source1 {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                let (b_rows, b_cols) = match source2 {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => (rows, cols),
                    _ => {
                        return Err(MatrixMultError(String::from(
                            "Matrix Variants do not match",
                        )));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixMultError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                match destination {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                        if rows != b_rows || cols != b_cols {
                            return Err(MatrixMultError(String::from(
                                "Destination Matrix dimensions do not match",
                            )));
                        }
                    }
                    Matrix::GPU(_) => {
                        return Err(MatrixMultError(String::from(
                            "Destination Matrix Variant does not match",
                        )));
                    }
                }

                for i in 0..*rows {
                    for j in 0..*cols {
                        destination[(i, j)] = source1[(i, j)] * source2[(i, j)];
                    }
                }

                Ok(())
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                device,
                queue,
                bind_group,
                elem_mult_pipeline,
                ..
            }) => {
                let (b_rows, b_cols, b_bind_group) = match source2 {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        bind_group,
                        ..
                    }) => (rows, cols, bind_group),
                    _ => {
                        return Err(MatrixMultError(String::from(
                            "Matrix Variants do not match",
                        )));
                    }
                };

                if *rows != *b_rows || *cols != *b_cols {
                    return Err(MatrixMultError(String::from(
                        "Matrix Rows and Colums do not match",
                    )));
                }

                // Create the output matrix to store add into
                let output_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        writable_bind_group,
                        ..
                    }) => {
                        if rows != b_rows || cols != b_cols {
                            return Err(MatrixMultError(String::from(
                                "Destination Matrix dimensions do not match",
                            )));
                        } else {
                            &*writable_bind_group
                        }
                    }
                    Matrix::CPU(_) => {
                        return Err(MatrixMultError(String::from(
                            "Destination Matrix Variant does not match",
                        )));
                    }
                };

                let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Matrix Elem Mult Into Command Encoder"),
                });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (*rows as u32, *cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Elem Mult Into Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&elem_mult_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);
                    compute_pass.set_bind_group(2, output_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);
                queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs the exponential operation on every element of the matrix
    /// returning a matrix where every element is now e^element
    ///
    /// # Returns
    ///
    /// `Result` with the new exponented matrix if success or `MatrixExpError` if failed
    pub fn exp(&self) -> Result<Matrix, MatrixExpError> {
        match self {
            Matrix::CPU(cpu_matrix) => {
                let mut output = cpu_matrix.clone();
                output
                    .data
                    .iter_mut()
                    .for_each(|value| *value = f32::exp(*value));

                Ok(Matrix::CPU(output))
            }
            Matrix::GPU(gpu_matrix) => {
                let output = GPUMatrix::with_shape(
                    (gpu_matrix.rows, gpu_matrix.cols),
                    None,
                    false,
                    gpu_matrix.device.clone(),
                    gpu_matrix.queue.clone(),
                );

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Exp Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Exp Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.exp_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Performs the exponential operation on every element of the matrix in place
    /// returning a matrix where every element is now e^element
    ///
    /// # Returns
    ///
    /// `Result` witha`Ok` if success or `MatrixExpError` if failed
    pub fn exp_in_place(&mut self) -> Result<(), MatrixExpError> {
        match self {
            Matrix::CPU(cpu_matrix) => {
                cpu_matrix
                    .data
                    .iter_mut()
                    .for_each(|value| *value = f32::exp(*value));

                Ok(())
            }
            Matrix::GPU(gpu_matrix) => {
                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Exp Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Exp Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.exp_in_place_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Performs the exponential operation on every element of the matrix
    /// Storing the result in `destination`
    ///
    /// # Arguments
    ///
    /// * `source` - source matrix to use for the exp procedure
    /// * `destination` - matrix to stoer the result into
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if success or `MatrixExpError` if failed
    pub fn exp_into(source: &Matrix, destination: &mut Matrix) -> Result<(), MatrixExpError> {
        match source {
            Matrix::CPU(cpu_matrix) => {
                match destination {
                    Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                        if cpu_matrix.rows != *rows || cpu_matrix.cols != *cols {
                            return Err(MatrixExpError(String::from(
                                "Destination Matrix dimensoins do not match",
                            )));
                        }
                    }
                    Matrix::GPU(_) => {
                        return Err(MatrixExpError(String::from(
                            "Destination Matrix Variant does not match",
                        )));
                    }
                }

                for i in 0..destination.rows() {
                    for j in 0..destination.cols() {
                        destination[(i, j)] = f32::exp(source[(i, j)]);
                    }
                }

                Ok(())
            }
            Matrix::GPU(gpu_matrix) => {
                let output_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        writable_bind_group,
                        ..
                    }) => {
                        if gpu_matrix.rows != *rows || gpu_matrix.cols != *cols {
                            return Err(MatrixExpError(String::from(
                                "Destination dimensions do not match",
                            )));
                        } else {
                            &*writable_bind_group
                        }
                    }
                    Matrix::CPU(_) => {
                        return Err(MatrixExpError(String::from(
                            "Destination Variant does not match",
                        )));
                    }
                };

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Exp Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Exp Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.exp_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, output_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Computes the sum of all the elements in a matrix and returns the result
    ///
    /// # Returns
    ///
    /// `f32` of the sum of all the elements
    pub fn sum(&self) -> Result<f32, MatrixSumError> {
        match self {
            Matrix::CPU(cpu_matrix) => {
                let output = cpu_matrix.data.iter().sum();
                Ok(output)
            }
            Matrix::GPU(gpu_matrix) => {
                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Sum Command Encoder"),
                        });

                {
                    let dispatch_size = compute_workgroup_size(
                        (gpu_matrix.rows * gpu_matrix.cols) as u32,
                        WORK_GROUP_SIZE,
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Sum Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.sum_pipeline);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);

                    // Dispatch Work Groups
                    compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
                }

                let output_buf = read_buffer(
                    &gpu_matrix.sum_buffer,
                    ((gpu_matrix.rows * gpu_matrix.cols) as f32 / 256.0).ceil() as u64
                        * std::mem::size_of::<f32>() as u64,
                    &gpu_matrix.device,
                    &mut encoder,
                );

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                // Get the sum of all the elements in the ouptut buffer since it has been reduced already
                Ok(get_buffer(&output_buf, &gpu_matrix.device).iter().sum())
            }
        }
    }

    /// Returns a transposed version of the current matrix
    ///
    /// # Returns
    ///
    /// `Matrix` with tranposed dimensions
    pub fn transposed(&self) -> Matrix {
        match self {
            Matrix::CPU(cpu_matrix) => {
                let (new_rows, new_cols) = (cpu_matrix.cols, cpu_matrix.rows);

                Matrix::CPU(CPUMatrix {
                    rows: new_rows,
                    cols: new_cols,
                    data: cpu_matrix.data.clone(),
                    transpose: !cpu_matrix.transpose,
                })
            }
            Matrix::GPU(gpu_matrix) => {
                let (new_rows, new_cols) = (gpu_matrix.cols, gpu_matrix.rows);

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Transpose Command Encoder"),
                        });

                let buf = read_buffer(
                    &gpu_matrix.data,
                    new_rows * new_cols * std::mem::size_of::<f32>() as u64,
                    &gpu_matrix.device,
                    &mut encoder,
                );

                gpu_matrix.queue.submit(Some(encoder.finish()));

                let data = get_buffer(&buf, &gpu_matrix.device);

                let output = GPUMatrix::with_shape(
                    (new_rows, new_cols),
                    Some(data),
                    !gpu_matrix.transpose,
                    gpu_matrix.device.clone(),
                    gpu_matrix.queue.clone(),
                );

                Matrix::GPU(output)
            }
        }
    }

    /// Consumes the matrix and returns a transposed version
    ///
    /// # Returns
    ///
    /// A transposed version of the current matrix
    pub fn transpose(self) -> Self {
        match self {
            Matrix::CPU(CPUMatrix {
                rows,
                cols,
                transpose,
                data,
            }) => {
                let (new_rows, new_cols) = (cols, rows);

                Matrix::CPU(CPUMatrix {
                    rows: new_rows,
                    cols: new_cols,
                    data,
                    transpose: !transpose,
                })
            }
            Matrix::GPU(mut gpu_matrix) => {
                let (new_rows, new_cols) = (gpu_matrix.cols, gpu_matrix.rows);

                gpu_matrix.queue.write_buffer(
                    &gpu_matrix.transpose_buffer,
                    0,
                    bytemuck::cast_slice(&[!gpu_matrix.transpose as u32]),
                );

                gpu_matrix.transpose = !gpu_matrix.transpose;
                gpu_matrix.rows = new_rows;
                gpu_matrix.cols = new_cols;

                Matrix::GPU(gpu_matrix)
            }
        }
    }

    /// Transposes the matrix in place
    pub fn transpose_in_place(&mut self) {
        match self {
            Matrix::CPU(CPUMatrix {
                rows,
                cols,
                transpose,
                ..
            }) => {
                let (new_rows, new_cols) = (*cols, *rows);
                *rows = new_rows;
                *cols = new_cols;
                *transpose = !*transpose;
            }
            Matrix::GPU(gpu_matrix) => {
                let (new_rows, new_cols) = (gpu_matrix.cols, gpu_matrix.rows);

                gpu_matrix.queue.write_buffer(
                    &gpu_matrix.transpose_buffer,
                    0,
                    bytemuck::cast_slice(&[!gpu_matrix.transpose as u32]),
                );

                gpu_matrix.transpose = !gpu_matrix.transpose;
                gpu_matrix.rows = new_rows;
                gpu_matrix.cols = new_cols;
            }
        }
    }

    /// Adds a custom single op shader described in `shader_module_descriptor`
    /// The entry point for this pipeline will always be "op_main"
    /// The bind groups consist of
    /// (0, 0, this matrix buffer, readable array<f32>)
    /// (0, 1, this matrix dimensions, uniform vec2<u32>)
    /// (0, 2, this matrix transpose, uniform u32)
    /// (1, 0, output matrix buffer, writable array<f32>)
    /// (1, 1, output matrix dimensions, uniform vec2<u32>)
    /// (1, 2, output matrix transpose, uniform u32)
    ///
    /// # Arguments
    ///
    /// * `shader_module_descriptor` - custom shader module to add
    ///
    /// # Returns
    ///
    /// `Option<usize>` of the index of the operation to be called when needed, None if it is a CPU matrix
    pub fn add_custom_single_op_pipeline(
        &mut self,
        shader_module_descriptor: ShaderModuleDescriptor,
    ) -> Option<usize> {
        match self {
            Matrix::CPU(_) => None,
            Matrix::GPU(gpu_matrix) => {
                let shader = gpu_matrix
                    .device
                    .create_shader_module(shader_module_descriptor);

                gpu_matrix
                    .custom_pipelines
                    .push(
                        gpu_matrix
                            .device
                            .create_compute_pipeline(&ComputePipelineDescriptor {
                                label: Some("Custom Pipeline"),
                                module: &shader,
                                layout: Some(&gpu_matrix.single_op_pipeline_layout),
                                entry_point: Some("op_main"),
                                cache: None,
                                compilation_options: PipelineCompilationOptions::default(),
                            }),
                    );

                Some(gpu_matrix.custom_pipelines.len() - 1)
            }
        }
    }

    /// Adds a custom single op shader described in `shader_module_descriptor`
    /// The entry point for this pipeline will always be "op_main"
    /// The bind groups consist of
    /// (0, 0, this matrix buffer, writable array<f32>)
    /// (0, 1, this matrix dimensions, uniform vec2<u32>)
    /// (0, 2, this matrix transpose, uniform u32)
    /// (0, 3, this matrix scalar buffer, uniform f32)
    ///
    /// # Arguments
    ///
    /// * `shader_module_descriptor` - custom shader module to add
    ///
    /// # Returns
    ///
    /// `Option<usize>` of the index of the operation to be called when needed, None if it is a CPU matrix
    pub fn add_custom_single_op_in_place_pipeline(
        &mut self,
        shader_module_descriptor: ShaderModuleDescriptor,
    ) -> Option<usize> {
        match self {
            Matrix::CPU(_) => None,
            Matrix::GPU(gpu_matrix) => {
                let shader = gpu_matrix
                    .device
                    .create_shader_module(shader_module_descriptor);

                gpu_matrix
                    .custom_pipelines
                    .push(
                        gpu_matrix
                            .device
                            .create_compute_pipeline(&ComputePipelineDescriptor {
                                label: Some("Custom Pipeline"),
                                module: &shader,
                                layout: Some(&gpu_matrix.single_op_in_place_pipeline_layout),
                                entry_point: Some("op_main"),
                                cache: None,
                                compilation_options: PipelineCompilationOptions::default(),
                            }),
                    );

                Some(gpu_matrix.custom_pipelines.len() - 1)
            }
        }
    }

    /// Runs the custom single op pipeline at the index described by `index`
    ///
    /// # Arguments
    ///
    /// `index` - index of the custome pipeline that was added
    ///
    /// # Returns
    ///
    /// `Result` with a `Matrix` if the operation was successful or `MatrixCustomError` if not
    pub fn run_custom_single_op_pipeline(&self, index: usize) -> Result<Matrix, MatrixCustomError> {
        match self {
            Matrix::CPU(_) => Err(MatrixCustomError(String::from(
                "Matrix is not a GPU Matrix",
            ))),
            Matrix::GPU(gpu_matrix) => {
                if index >= gpu_matrix.custom_pipelines.len() {
                    return Err(MatrixCustomError(String::from(
                        "Pipeline Index Out of Range",
                    )));
                }

                let output = GPUMatrix::with_shape(
                    (gpu_matrix.rows, gpu_matrix.cols),
                    None,
                    false,
                    gpu_matrix.device.clone(),
                    gpu_matrix.queue.clone(),
                );

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Custom Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Custom Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.custom_pipelines[index]);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Runs the custom single op pipeline at the index described by `index`
    /// And stores the result into `destination`
    ///
    /// # Arguments
    ///
    /// * `source` - source matrix to use
    /// * `destination` - matrix to store the result in
    /// * `index` - index of the custome pipeline that was added
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the operation was successful or `MatrixCustomError` if not
    pub fn run_custom_single_op_pipeline_into(
        source: &Matrix,
        index: usize,
        destination: &mut Matrix,
    ) -> Result<(), MatrixCustomError> {
        match source {
            Matrix::CPU(_) => Err(MatrixCustomError(String::from(
                "Matrix is not a GPU Matrix",
            ))),
            Matrix::GPU(gpu_matrix) => {
                if index >= gpu_matrix.custom_pipelines.len() {
                    return Err(MatrixCustomError(String::from(
                        "Pipeline Index Out of Range",
                    )));
                }

                let output_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        writable_bind_group,
                        ..
                    }) => &*writable_bind_group,
                    Matrix::CPU(_) => {
                        return Err(MatrixCustomError(String::from(
                            "Destination Variant does not match",
                        )));
                    }
                };

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Custom Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Custom Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.custom_pipelines[index]);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, output_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Runs the custom single op pipeline at the index described by `index`
    /// in place
    ///
    /// # Arguments
    ///
    /// * `index` - index of the custome pipeline that was added
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the operation was successful or `MatrixCustomError` if not
    pub fn run_custom_single_op_pipeline_in_place(
        &mut self,
        index: usize,
    ) -> Result<(), MatrixCustomError> {
        match self {
            Matrix::CPU(_) => Err(MatrixCustomError(String::from(
                "Matrix is not a GPU Matrix",
            ))),
            Matrix::GPU(gpu_matrix) => {
                if index >= gpu_matrix.custom_pipelines.len() {
                    return Err(MatrixCustomError(String::from(
                        "Pipeline Index Out of Range",
                    )));
                }

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Custom Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Custom Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.custom_pipelines[index]);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Adds a custom multi op shader described in `shader_module_descriptor`
    /// The entry point for this pipeline will always be "op_main"
    /// The bind groups consist of
    /// (0, 0, this matrix buffer, readable array<f32>)
    /// (0, 1, this matrix dimensions, uniform vec2<u32>)
    /// (0, 2, this matrix transpose, uniform u32)
    /// (0, 3, this matrix scalar, uniform f32)
    /// (0, 4, this matrix sum buffer, writable array<f32>)
    /// (1, 0, other matrix buffer, readable array<f32>)
    /// (1, 1, other matrix dimensions, uniform vec2<u32>)
    /// (1, 2, other matrix transpose, uniform u32)
    /// (1, 4, other matrix sum buffer, writable array<f32>)
    /// (2, 0, output matrix buffer, writable array<f32>)
    /// (2, 1, output matrix dimensions, uniform vec2<u32>)
    /// (2, 2, output matrix transpose, uniform u32)
    /// (2, 3, output matrix scalar, uniform f32)
    ///
    /// # Arguments
    ///
    /// * `shader_module_descriptor` - custom shader module to add
    ///
    /// # Returns
    ///
    /// `Option<usize>` of the index of the operation to be called when needed, None if it is a CPU matrix
    pub fn add_custom_multi_op_pipeline(
        &mut self,
        shader_module_descriptor: ShaderModuleDescriptor,
    ) -> Option<usize> {
        match self {
            Matrix::CPU(_) => None,
            Matrix::GPU(gpu_matrix) => {
                let shader = gpu_matrix
                    .device
                    .create_shader_module(shader_module_descriptor);

                gpu_matrix
                    .custom_pipelines
                    .push(
                        gpu_matrix
                            .device
                            .create_compute_pipeline(&ComputePipelineDescriptor {
                                label: Some("Custom Pipeline"),
                                module: &shader,
                                layout: Some(&gpu_matrix.multi_op_pipeline_layout),
                                entry_point: Some("op_main"),
                                cache: None,
                                compilation_options: PipelineCompilationOptions::default(),
                            }),
                    );

                Some(gpu_matrix.custom_pipelines.len() - 1)
            }
        }
    }

    /// Adds a custom multi op shader described in `shader_module_descriptor`
    /// The entry point for this pipeline will always be "op_main"
    /// The bind groups consist of
    /// (0, 0, this matrix buffer, readable array<f32>)
    /// (0, 1, this matrix dimensions, uniform vec2<u32>)
    /// (0, 2, this matrix transpose, uniform u32)
    /// (0, 3, this matrix scalar, uniform f32)
    /// (0, 4, this matrix sum buffer, writable array<f32>)
    /// (1, 0, other matrix buffer, readable array<f32>)
    /// (1, 1, other matrix dimensions, uniform vec2<u32>)
    /// (1, 2, other matrix transpose, uniform u32)
    /// (1, 4, other matrix sum buffer, writable array<f32>)
    /// (2, 0, output matrix buffer, writable array<f32>)
    /// (2, 1, output matrix dimensions, uniform vec2<u32>)
    /// (2, 2, output matrix transpose, uniform u32)
    /// (2, 3, output matrix scalar, uniform f32)
    ///
    /// # Arguments
    ///
    /// * `shader_module_descriptor` - custom shader module to add
    ///
    /// # Returns
    ///
    /// `Option<usize>` of the index of the operation to be called when needed, None if it is a CPU matrix
    pub fn add_custom_multi_op_in_place_pipeline(
        &mut self,
        shader_module_descriptor: ShaderModuleDescriptor,
    ) -> Option<usize> {
        match self {
            Matrix::CPU(_) => None,
            Matrix::GPU(gpu_matrix) => {
                let shader = gpu_matrix
                    .device
                    .create_shader_module(shader_module_descriptor);

                gpu_matrix
                    .custom_pipelines
                    .push(
                        gpu_matrix
                            .device
                            .create_compute_pipeline(&ComputePipelineDescriptor {
                                label: Some("Custom Pipeline"),
                                module: &shader,
                                layout: Some(&gpu_matrix.multi_op_in_place_pipeline_layout),
                                entry_point: Some("op_main"),
                                cache: None,
                                compilation_options: PipelineCompilationOptions::default(),
                            }),
                    );

                Some(gpu_matrix.custom_pipelines.len() - 1)
            }
        }
    }

    /// Runs the custom multi op pipeline at the index described by `index`
    ///
    /// # Arguments
    ///
    /// * `other` - other matrix to use in the multi op pipeline
    /// * `index` - index of the custom pipeline that was added
    ///
    /// # Returns
    ///
    /// `Result` with a `Matrix` if the operation was successful or `MatrixCustomError` if not
    pub fn run_custom_multi_op_pipeline(
        &self,
        other: &Matrix,
        index: usize,
    ) -> Result<Matrix, MatrixCustomError> {
        match self {
            Matrix::CPU(_) => Err(MatrixCustomError(String::from(
                "Matrix is not a GPU Matrix",
            ))),
            Matrix::GPU(gpu_matrix) => {
                if index >= gpu_matrix.custom_pipelines.len() {
                    return Err(MatrixCustomError(String::from(
                        "Pipeline Index Out of Range",
                    )));
                }

                let output = GPUMatrix::with_shape(
                    (gpu_matrix.rows, gpu_matrix.cols),
                    None,
                    false,
                    gpu_matrix.device.clone(),
                    gpu_matrix.queue.clone(),
                );

                let other_bind_group = match other {
                    Matrix::CPU(_) => {
                        return Err(MatrixCustomError(String::from(
                            "Other Matrix is not GPU matrix",
                        )));
                    }
                    Matrix::GPU(gpu_matrix) => &gpu_matrix.bind_group,
                };

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Custom Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Custom Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.custom_pipelines[index]);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, other_bind_group, &[]);
                    compute_pass.set_bind_group(2, &output.writable_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(Matrix::GPU(output))
            }
        }
    }

    /// Runs the custom multi op pipeline at the index described by `index`
    /// And stores the result into `destination`
    ///
    /// # Arguments
    ///
    /// * `source1` - source matrix to use
    /// * `source2` - second source matrix to use
    /// * `destination` - matrix to store the result in
    /// * `index` - index of the custome pipeline that was added
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the operation was successful or `MatrixCustomError` if not
    pub fn run_custom_multi_op_pipeline_into(
        source1: &Matrix,
        source2: &Matrix,
        index: usize,
        destination: &mut Matrix,
    ) -> Result<(), MatrixCustomError> {
        match source1 {
            Matrix::CPU(_) => Err(MatrixCustomError(String::from(
                "Matrix is not a GPU Matrix",
            ))),
            Matrix::GPU(gpu_matrix) => {
                if index >= gpu_matrix.custom_pipelines.len() {
                    return Err(MatrixCustomError(String::from(
                        "Pipeline Index Out of Range",
                    )));
                }

                let b_bind_group = match source2 {
                    Matrix::CPU(_) => {
                        return Err(MatrixCustomError(String::from(
                            "Source 2 Matrix is not GPU matrix",
                        )));
                    }
                    Matrix::GPU(gpu_matrix) => &gpu_matrix.bind_group,
                };

                let output_bind_group = match destination {
                    Matrix::GPU(GPUMatrix {
                        writable_bind_group,
                        ..
                    }) => &*writable_bind_group,
                    Matrix::CPU(_) => {
                        return Err(MatrixCustomError(String::from(
                            "Destination Variant does not match",
                        )));
                    }
                };

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Custom Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Custom Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.custom_pipelines[index]);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.bind_group, &[]);
                    compute_pass.set_bind_group(1, b_bind_group, &[]);
                    compute_pass.set_bind_group(2, output_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }

    /// Runs the custom multi op pipeline at the index described by `index`
    /// in place
    ///
    /// # Arguments
    ///
    /// * `index` - index of the custom pipeline that was added
    /// * `other` - other matrix to usin the multi op pipeline
    ///
    /// # Returns
    ///
    /// `Result` with `Ok` if the operation was successful or `MatrixCustomError` if not
    pub fn run_custom_multi_op_pipeline_in_place(
        &mut self,
        other: &Matrix,
        index: usize,
    ) -> Result<(), MatrixCustomError> {
        match self {
            Matrix::CPU(_) => Err(MatrixCustomError(String::from(
                "Matrix is not a GPU Matrix",
            ))),
            Matrix::GPU(gpu_matrix) => {
                if index >= gpu_matrix.custom_pipelines.len() {
                    return Err(MatrixCustomError(String::from(
                        "Pipeline Index Out of Range",
                    )));
                }

                let other_bind_group = match other {
                    Matrix::CPU(_) => {
                        return Err(MatrixCustomError(String::from(
                            "Other Matrix Variant is not GPU matrix",
                        )));
                    }
                    Matrix::GPU(gpu_matrix) => &gpu_matrix.bind_group,
                };

                let mut encoder =
                    gpu_matrix
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Matrix Custom Command Encoder"),
                        });

                {
                    let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                        (gpu_matrix.rows as u32, gpu_matrix.cols as u32),
                        (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                    );

                    // Begin the compute pass
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Matrix Custom Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Set the pipeline
                    compute_pass.set_pipeline(&gpu_matrix.custom_pipelines[index]);

                    // Set the bind groups
                    compute_pass.set_bind_group(0, &gpu_matrix.writable_bind_group, &[]);
                    compute_pass.set_bind_group(1, other_bind_group, &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(())
            }
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        match self {
            Matrix::CPU(CPUMatrix {
                rows,
                cols,
                data,
                transpose,
                ..
            }) => {
                let inner_index = if *transpose {
                    index.0 + *rows * index.1
                } else {
                    index.0 * *cols + index.1
                };
                &data[inner_index]
            }
            Matrix::GPU(_gpu_matrix) => {
                todo!()
            }
        }
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        match self {
            Matrix::CPU(CPUMatrix {
                rows,
                cols,
                data,
                transpose,
            }) => {
                let inner_index = if *transpose {
                    index.0 + *rows * index.1
                } else {
                    index.0 * *cols + index.1
                };
                &mut data[inner_index]
            }
            _ => {
                todo!()
            }
        }
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        match self {
            Matrix::CPU(CPUMatrix { rows, cols, .. }) => {
                for i in 0..*rows {
                    write!(f, "| ")?;
                    for j in 0..*cols {
                        write!(f, "{}, ", self[(i, j)])?;
                    }
                    writeln!(f, "|")?;
                }
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                data,
                transpose,
                device,
                queue,
                ..
            }) => {
                let values = {
                    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("Matrix Print Command Encoder"),
                    });

                    let val_buf = read_buffer(
                        &data,
                        *rows * *cols * std::mem::size_of::<f32>() as u64,
                        &device,
                        &mut encoder,
                    );

                    queue.submit(Some(encoder.finish()));

                    get_buffer(&val_buf, &device)
                };

                for i in 0..*rows {
                    write!(f, "| ")?;
                    for j in 0..*cols {
                        let index = if *transpose {
                            i + *rows * j
                        } else {
                            i * *cols + j
                        };

                        write!(f, "{}, ", values[index as usize])?;
                    }
                    writeln!(f, "|")?;
                }
            }
        }

        Ok(())
    }
}
