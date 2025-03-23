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

    // Uniform for scalar multiplications
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

    // Subtracting
    sub_pipeline: ComputePipeline,

    // Multiplying
    mult_pipeline: ComputePipeline,

    // Exponential
    exp_pipeline: ComputePipeline,

    // Summing all elements
    sum_pipeline: ComputePipeline,

    // Custom Pipelines
    custom_pipelines: Vec<ComputePipeline>,

    // Layouts for adding custom pipelines
    multi_op_pipeline_layout: PipelineLayout,
    single_op_pipeline_layout: PipelineLayout,

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
            label: Some("Matirx Scalar Buffer"),
            mapped_at_creation: false,
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Create a buffer for summing all the elements
        let sum_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Matrix Sum Buffer"),
            mapped_at_creation: false,
            size: {
                // Computing half the size based on the closest power of 2
                let half = (new_rows * new_cols) as f32 / 2.0;
                let next_pow_2 = f32::powf(2.0, (half.log(2.0)).ceil()) as u64;

                next_pow_2 * std::mem::size_of::<f32>() as u64
            },
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
            (2, &transpose, Bbt::Uniform)
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

        // Create the pipeline for a single operation
        let single_op_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Single Op Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &writable_bind_group_layout],
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
            sub_pipeline,
            mult_pipeline,
            exp_pipeline,
            sum_pipeline,
            custom_pipelines: Vec::new(),
            multi_op_pipeline_layout: pipeline_layout,
            single_op_pipeline_layout,
        }
    }
}

/// Matrix that can have a defined shape on the gpu or the cpu
///
/// # Variants
///
/// * `CPU` - CPU stored and computed matrix
/// * `GPU` - GPU stored and computed matrix
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
                        (gpu_matrix.rows * gpu_matrix.cols / 2) as u32,
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
                    (gpu_matrix.rows * gpu_matrix.cols) / 2 * std::mem::size_of::<f32>() as u64,
                    &gpu_matrix.device,
                    &mut encoder,
                );

                gpu_matrix.device.poll(Maintain::Wait);
                gpu_matrix.queue.submit(Some(encoder.finish()));

                Ok(get_buffer(&output_buf, &gpu_matrix.device)[0])
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
                                label: Some(""),
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
            _ => {
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

#[cfg(test)]
mod tests {
    use pollster::FutureExt;
    use wgpu::{
        Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        PowerPreference, RequestAdapterOptions,
    };

    use super::*;

    #[test]
    fn test_rand_with_shape() {
        let mat = Matrix::rand_with_shape((10, 5));

        println!("{}", mat);
        assert!(true);
    }

    #[test]
    fn test_setting_values() {
        let mut mat = Matrix::with_shape((10, 10));

        for i in 0..10 {
            mat[(i, i)] = 1.0;
        }

        println!("{}", mat);
        assert!(true);

        let mut mat = Matrix::with_shape((10, 5));

        for i in 0..10 {
            mat[(i, 0)] = 1.0;
        }

        println!("{}", mat);
        assert!(true);
    }

    #[test]
    fn test_cpu_dot() {
        let mut mat1 = Matrix::with_shape((3, 4));
        let mut mat2 = Matrix::with_shape((4, 2));

        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                let index = i * mat1.cols() + j;

                mat1[(i, j)] = (index + 1) as f32;
            }
        }

        for i in 0..mat2.cols() {
            for j in 0..mat2.rows() {
                let index = i * mat2.rows() + j;

                mat2[(j, i)] = (index + 1) as f32;
            }
        }

        println!("Matrix 1: {}", mat1);
        println!("Matrix 2: {}", mat2);

        assert!(true);

        println!("Mat 1: {}x{}", mat1.rows(), mat1.cols());
        println!("Mat 2: {}x{}", mat2.rows(), mat2.cols());

        let result = match mat1.dot(&mat2) {
            Ok(res) => res,
            Err(err) => panic!("Error: {}", err),
        };

        println!("Result: {}", result);

        assert!(true);
    }

    #[test]
    fn test_gpu_dot() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat1 = Matrix::with_shape((3, 4));
        let mut mat2 = Matrix::with_shape((4, 2));

        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                let index = i * mat1.cols() + j;

                mat1[(i, j)] = (index + 1) as f32;
            }
        }

        for i in 0..mat2.cols() {
            for j in 0..mat2.rows() {
                let index = i * mat2.rows() + j;

                mat2[(j, i)] = (index + 1) as f32;
            }
        }

        mat1 = mat1.buf(device.clone(), queue.clone());
        mat2 = mat2.buf(device.clone(), queue.clone());

        let mut output = mat1.dot(&mat2).expect("Failed to compute dot product");

        println!("A: {}", mat1);
        println!("B: {}", mat2);
        println!("Result: {}", output);

        output = output.debuf();

        println!("Result Debuf: {}", output);

        assert!(true);
    }

    #[test]
    fn test_cpu_add() {
        let mut mat1 = Matrix::with_shape((5, 6));
        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                mat1[(i, j)] = (i * mat1.cols() + j) as f32;
            }
        }

        let mut mat2 = Matrix::with_shape((5, 6));
        for i in 0..mat2.rows() {
            for j in 0..mat2.cols() {
                mat2[(i, j)] = (i * mat2.cols() + j) as f32;
            }
        }

        let output_mat = mat1.add(&mat2).expect("Could not add matrices");

        println!("Add A: {}", mat1);
        println!("Add B: {}", mat2);
        println!("Add Result: {}", output_mat);
        assert!(true);
    }

    #[test]
    fn test_gpu_add() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat1 = Matrix::with_shape((5, 6));
        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                mat1[(i, j)] = (i * mat1.cols() + j) as f32;
            }
        }
        mat1 = mat1.buf(device.clone(), queue.clone());

        let mut mat2 = Matrix::with_shape((5, 6));
        for i in 0..mat2.rows() {
            for j in 0..mat2.cols() {
                mat2[(i, j)] = (i * mat2.cols() + j) as f32;
            }
        }
        mat2 = mat2.buf(device.clone(), queue.clone());

        let output_mat = mat1.add(&mat2).expect("Could not add matrices");

        println!("Add A: {}", mat1);
        println!("Add B: {}", mat2);
        println!("Add Result: {}", output_mat);
        assert!(true);
    }

    #[test]
    fn test_cpu_sub() {
        let mut mat1 = Matrix::with_shape((5, 6));
        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                mat1[(i, j)] = (i * mat1.cols() + j) as f32;
            }
        }

        let mut mat2 = Matrix::with_shape((5, 6));
        for i in 0..mat2.rows() {
            for j in 0..mat2.cols() {
                mat2[(i, j)] = (i * mat2.cols() + j) as f32;
            }
        }

        let output_mat = mat1.sub(&mat2).expect("Could not add matrices");

        println!("Sub A: {}", mat1);
        println!("Sub B: {}", mat2);
        println!("Sub Result: {}", output_mat);
        assert!(true);
    }

    #[test]
    fn test_gpu_sub() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat1 = Matrix::with_shape((5, 6));
        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                mat1[(i, j)] = (i * mat1.cols() + j) as f32;
            }
        }
        mat1 = mat1.buf(device.clone(), queue.clone());

        let mut mat2 = Matrix::with_shape((5, 6));
        for i in 0..mat2.rows() {
            for j in 0..mat2.cols() {
                mat2[(i, j)] = (i * mat2.cols() + j) as f32;
            }
        }
        mat2 = mat2.buf(device.clone(), queue.clone());

        let output_mat = mat1.sub(&mat2).expect("Could not add matrices");

        println!("Sub A: {}", mat1);
        println!("Sub B: {}", mat2);
        println!("Sub Result: {}", output_mat);
        assert!(true);
    }

    #[test]
    fn test_cpu_trasnpose() {
        let mut mat1 = Matrix::with_shape((5, 6));
        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                mat1[(i, j)] = (i * mat1.cols() + j) as f32;
            }
        }
        println!("Before Transpose: {}", mat1);
        println!("After Trasnpose: {}", mat1.transpose());

        assert!(true);
    }

    #[test]
    fn test_gpu_transpose() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat1 = Matrix::with_shape((5, 6));
        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                mat1[(i, j)] = (i * mat1.cols() + j) as f32;
            }
        }

        mat1 = mat1.buf(device.clone(), queue.clone());
        println!("Before Transpose: {}", mat1);
        println!("After Trasnpose: {}", mat1.transpose());

        assert!(true);
    }

    #[test]
    fn test_cpu_transpose_add() {
        let mut mat1 = Matrix::with_shape((5, 6));
        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                mat1[(i, j)] = (i * mat1.cols() + j) as f32;
            }
        }

        let mut mat2 = Matrix::with_shape((6, 5));
        for i in 0..mat2.rows() {
            for j in 0..mat2.cols() {
                mat2[(i, j)] = (i * mat2.cols() + j) as f32;
            }
        }

        mat1 = mat1.transpose();
        println!("A^T: {}", mat1);
        println!("B: {}", mat2);
        println!(
            "Result: {}",
            mat1.add(&mat2).expect("Adding matrices failed")
        );

        assert!(true);
    }

    #[test]
    fn test_gpu_transpose_add() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat1 = Matrix::with_shape((5, 6));
        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                mat1[(i, j)] = (i * mat1.cols() + j) as f32;
            }
        }
        mat1 = mat1.buf(device.clone(), queue.clone());

        let mut mat2 = Matrix::with_shape((6, 5));
        for i in 0..mat2.rows() {
            for j in 0..mat2.cols() {
                mat2[(i, j)] = (i * mat2.cols() + j) as f32;
            }
        }
        mat2 = mat2.buf(device.clone(), queue.clone());
        mat1 = mat1.transpose();

        let output_mat = mat1.add(&mat2).expect("Could not add matrices");

        println!("Add A: {}", mat1);
        println!("Add B: {}", mat2);
        println!("Add Result: {}", output_mat);
        assert!(true);
    }

    #[test]
    fn test_cpu_transpose_dot() {
        let mut mat1 = Matrix::with_shape((3, 4));
        let mut mat2 = Matrix::with_shape((3, 5));

        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                let index = i * mat1.cols() + j;

                mat1[(i, j)] = (index + 1) as f32;
            }
        }

        for i in 0..mat2.cols() {
            for j in 0..mat2.rows() {
                let index = i * mat2.rows() + j;

                mat2[(j, i)] = (index + 1) as f32;
            }
        }

        mat1 = mat1.transpose();

        println!("Matrix 1: {}", mat1);
        println!("Matrix 2: {}", mat2);

        assert!(true);

        println!("Mat 1: {}x{}", mat1.rows(), mat1.cols());
        println!("Mat 2: {}x{}", mat2.rows(), mat2.cols());

        let result = match mat1.dot(&mat2) {
            Ok(res) => res,
            Err(err) => panic!("Error: {}", err),
        };

        println!("Result: {}", result);

        assert!(true);
    }

    #[test]
    fn test_gpu_transpose_dot() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat1 = Matrix::with_shape((3, 4));
        let mut mat2 = Matrix::with_shape((3, 5));

        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                let index = i * mat1.cols() + j;

                mat1[(i, j)] = (index + 1) as f32;
            }
        }

        for i in 0..mat2.cols() {
            for j in 0..mat2.rows() {
                let index = i * mat2.rows() + j;

                mat2[(j, i)] = (index + 1) as f32;
            }
        }

        mat1 = mat1.transpose();

        mat1 = mat1.buf(device.clone(), queue.clone());
        mat2 = mat2.buf(device.clone(), queue.clone());

        let mut output = mat1.dot(&mat2).expect("Failed to compute dot product");

        println!("A: {}", mat1);
        println!("B: {}", mat2);
        println!("Result: {}", output);

        output = output.debuf();

        println!("Result Debuf: {}", output);

        assert!(true);
    }

    #[test]
    fn test_cpu_double_transpose() {
        let mut mat1 = Matrix::with_shape((3, 4));

        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                let index = i * mat1.cols() + j;

                mat1[(i, j)] = (index + 1) as f32;
            }
        }

        println!("Before: {}", mat1);
        mat1 = mat1.transpose();
        println!("After First Tranpose: {}", mat1);
        mat1 = mat1.transpose();
        println!("After Second Transpose: {}", mat1);

        assert!(true);
    }

    #[test]
    fn test_gpu_double_transpose() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat1 = Matrix::with_shape((3, 4));

        for i in 0..mat1.rows() {
            for j in 0..mat1.cols() {
                let index = i * mat1.cols() + j;

                mat1[(i, j)] = (index + 1) as f32;
            }
        }

        mat1 = mat1.buf(device.clone(), queue.clone());

        println!("Before: {}", mat1);
        mat1 = mat1.transpose();
        println!("After First Tranpose: {}", mat1);
        mat1 = mat1.transpose();
        println!("After Second Transpose: {}", mat1);

        assert!(true);
    }

    #[test]
    fn test_cpu_scalar_mult() {
        let mut mat = Matrix::with_shape((5, 6));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let index = i * mat.cols() + j;
                mat[(i, j)] = (index + 1) as f32;
            }
        }

        println!("Before Mult: {}", mat);
        println!(
            "After Mult: {}",
            mat.mult(12.0).expect("Could Not Multiply Matrix")
        );

        assert!(true)
    }

    #[test]
    fn test_gpu_scalar_mult() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat = Matrix::with_shape((5, 6));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let index = i * mat.cols() + j;
                mat[(i, j)] = (index + 1) as f32;
            }
        }

        mat = mat.buf(device.clone(), queue.clone());

        println!("Before Mult: {}", mat);
        println!(
            "After Mult: {}",
            mat.mult(12.0).expect("Could not multiply matrix")
        );

        assert!(true);
    }

    #[test]
    fn test_cpu_exp() {
        let mut mat = Matrix::with_shape((5, 6));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let index = i * mat.cols() + j;
                mat[(i, j)] = (index + 1) as f32;
            }
        }

        println!("Before Exp: {}", mat);
        println!(
            "After Exp: {}",
            mat.exp().expect("Could Not Multiply Matrix")
        );

        assert!(true)
    }

    #[test]
    fn test_gpu_exp() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat = Matrix::with_shape((5, 6));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let index = i * mat.cols() + j;
                mat[(i, j)] = (index + 1) as f32;
            }
        }

        mat = mat.buf(device.clone(), queue.clone());

        println!("Before Exp: {}", mat);
        println!("After Exp: {}", mat.exp().expect("Could Not do Matrix Exp"));

        assert!(true);
    }

    #[test]
    fn test_cpu_sum() {
        let mut mat = Matrix::with_shape((50, 1));

        for i in 0..mat.rows() {
            mat[(i, 0)] = i as f32;
        }

        println!(
            "Sum of: {} is {}",
            mat,
            mat.sum().expect("Could Not compute Sum")
        );

        assert!(true);
    }

    #[test]
    fn test_gpu_sum() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat = Matrix::with_shape((50, 1));

        for i in 0..mat.rows() {
            mat[(i, 0)] = i as f32;
        }

        mat = mat.buf(device.clone(), queue.clone());

        println!(
            "Sum of: {} is {}",
            mat,
            mat.sum().expect("Could Not compute Sum")
        );

        assert!(true);
    }

    #[test]
    fn test_custom_pipeline() {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        let mut mat = Matrix::with_shape((10, 10));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let value = (i * mat.cols() + j) as f32 - ((mat.rows() * mat.cols()) as f32 / 2.0);
                mat[(i, j)] = value;
            }
        }

        mat = mat.buf(device.clone(), queue.clone());
        let index = mat
            .add_custom_single_op_pipeline(include_wgsl!("shaders/relu.wgsl"))
            .expect("Failed to Add Pipeline");

        println!("Before Compute: {}", mat);
        println!(
            "After Compute: {}",
            mat.run_custom_single_op_pipeline(index)
                .expect("Failed to Run Custom Compute")
        );
    }
}
