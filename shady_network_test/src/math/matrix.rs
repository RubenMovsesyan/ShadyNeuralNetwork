use std::fmt::Display;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
// WGPU imports
use wgpu::{
    BindGroup, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, Maintain,
    PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderStages, include_wgsl,
};

use crate::create_buffer_bind_group;
use crate::gpu_utils::{
    WORK_GROUP_SIZE_2D, compute_workgroup_size_2d, get_buffer, print_buffer, read_buffer,
};

use super::math_errors::MatrixDotError;

#[derive(Debug)]
struct CPUMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

#[derive(Debug)]
struct GPUMatrix {
    rows: u64,
    cols: u64,
    data: Buffer,
    dimensions: Buffer,
    device: Rc<Device>,
    queue: Rc<Queue>,

    // Dotting
    dot_bind_group_layout: BindGroupLayout,
    dot_bind_group: BindGroup,
    dot_pipeline: ComputePipeline,
}

impl GPUMatrix {
    fn with_shape(capacity: (u64, u64), device: Rc<Device>, queue: Rc<Queue>) -> Self {
        let new_rows = capacity.0;
        let new_cols = capacity.1;

        // Create a buffer with the current data
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Matrix Buffer"),
            mapped_at_creation: false,
            size: new_rows * new_cols * std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let dims = vec![new_rows as u32, new_cols as u32];

        // Create a buffer with the current dimensions
        let dimensions_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dims),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let (dot_bind_group_layout, dot_bind_group) = create_buffer_bind_group!(
            &device,
            "Dot Bind Group",
            (0, &buffer, Bbt::Storage { read_only: true }),
            (1, &dimensions_buffer, Bbt::Uniform)
        );

        let (dot_writable_bind_group_layout, _) = create_buffer_bind_group!(
            &device,
            "Dot Writable Bind Group",
            (0, &buffer, Bbt::Storage { read_only: false }),
            (1, &dimensions_buffer, Bbt::Uniform)
        );

        // Create the compute pipeline for dotting
        let dot_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/dotting.wgsl"));

            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Dot Compute Pipeline Descriptor"),
                bind_group_layouts: &[
                    &dot_bind_group_layout,
                    &dot_bind_group_layout, // The input bind group layouts should be the same
                    &dot_writable_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Dot Compute Pipeline"),
                module: &shader,
                layout: Some(&layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("dot_main"),
            })
        };

        GPUMatrix {
            rows: new_rows,
            cols: new_cols,
            data: buffer,
            dimensions: dimensions_buffer,
            device,
            queue,
            dot_bind_group_layout,
            dot_bind_group,
            dot_pipeline,
        }
    }

    fn from_data(
        rows: usize,
        cols: usize,
        data: Vec<f32>,
        device: Rc<Device>,
        queue: Rc<Queue>,
    ) -> Self {
        let new_rows = rows as u64;
        let new_cols = cols as u64;

        // Create a buffer with the current data
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let dims = vec![new_rows as u32, new_cols as u32];

        // Create a buffer with the current dimensions
        let dimensions_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dims),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let (dot_bind_group_layout, dot_bind_group) = create_buffer_bind_group!(
            &device,
            "Dot Bind Group",
            (0, &buffer, Bbt::Storage { read_only: true }),
            (1, &dimensions_buffer, Bbt::Uniform)
        );

        let (dot_writable_bind_group_layout, _) = create_buffer_bind_group!(
            &device,
            "Dot Writable Bind Group",
            (0, &buffer, Bbt::Storage { read_only: false }),
            (1, &dimensions_buffer, Bbt::Uniform)
        );

        // Create the compute pipeline for dotting
        let dot_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/dotting.wgsl"));

            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Dot Compute Pipeline Descriptor"),
                bind_group_layouts: &[
                    &dot_bind_group_layout,
                    &dot_bind_group_layout, // The input bind group layouts should be the same
                    &dot_writable_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Dot Compute Pipeline"),
                module: &shader,
                layout: Some(&layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("dot_main"),
            })
        };

        GPUMatrix {
            rows: new_rows,
            cols: new_cols,
            data: buffer,
            dimensions: dimensions_buffer,
            device,
            queue,
            dot_bind_group_layout,
            dot_bind_group,
            dot_pipeline,
        }
    }

    fn get_writable_bind_group(&self, device: &Device) -> BindGroup {
        let (_, dot_bind_group) = create_buffer_bind_group!(
            &device,
            "Dot Bind Group",
            (0, &self.data, Bbt::Storage { read_only: false }),
            (1, &self.dimensions, Bbt::Uniform)
        );

        dot_bind_group
    }
}

#[derive(Debug)]
pub enum Matrix {
    CPU(CPUMatrix),
    GPU(GPUMatrix),
}

impl Matrix {
    pub fn with_shape(capacity: (usize, usize)) -> Self {
        let rows = capacity.0;
        let cols = capacity.1;

        let data = vec![0.0; rows * cols];

        Matrix::CPU(CPUMatrix { rows, cols, data })
    }

    pub fn rand_with_shape(capacity: (usize, usize)) -> Self {
        let rows = capacity.0;
        let cols = capacity.1;

        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            data.push(rand::random_range(0.0..=1.0));
        }

        Matrix::CPU(CPUMatrix { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        match self {
            Matrix::CPU(cpu_matrix) => cpu_matrix.rows,
            Matrix::GPU(gpu_matrix) => gpu_matrix.rows as usize,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Matrix::CPU(cpu_matrix) => cpu_matrix.cols,
            Matrix::GPU(gpu_matrix) => gpu_matrix.cols as usize,
        }
    }

    pub fn buf(self, device: Rc<Device>, queue: Rc<Queue>) -> Self {
        match self {
            Matrix::CPU(CPUMatrix { rows, cols, data }) => {
                Matrix::GPU(GPUMatrix::from_data(rows, cols, data, device, queue))
            }
            Matrix::GPU(_) => self,
        }
    }

    pub fn dot(&self, other: &Matrix) -> Result<Matrix, MatrixDotError> {
        match self {
            Matrix::CPU(CPUMatrix { rows, cols, data }) => {
                // Get the pointers to the other matrix's data elements
                // Because we have already asserted that they are the same
                let (other_rows, other_cols, other_data) = match other {
                    Matrix::CPU(CPUMatrix { rows, cols, data }) => (rows, cols, data),
                    _ => {
                        return Err(MatrixDotError(String::from("Matrix Variants do not match")));
                    }
                };

                // before getting the data make sure to check if the dot product is possible
                if *cols != *other_rows {
                    return Err(MatrixDotError(String::from(
                        "Columns of matrix 1 do not match rows of matrix 2",
                    )));
                }

                let (result_rows, result_cols) = (*rows, *other_cols);
                let mut output_mat = Matrix::with_shape((result_rows, result_cols));
                for i in 0..result_rows {
                    for j in 0..result_cols {
                        for k in 0..*cols {
                            let self_data_index = i * *cols + k;
                            let other_data_index = k * *other_cols + j;
                            output_mat[(i, j)] +=
                                data[self_data_index] * other_data[other_data_index];
                        }
                    }
                }

                Ok(output_mat)
            }
            Matrix::GPU(GPUMatrix {
                rows,
                cols,
                data,
                dimensions,
                device,
                queue,
                dot_bind_group_layout: _,
                dot_bind_group,
                dot_pipeline,
            }) => {
                let (b_rows, b_cols, b_data, b_dimensions, b_dot_bind_group) = match other {
                    Matrix::GPU(GPUMatrix {
                        rows,
                        cols,
                        data,
                        dimensions,
                        dot_bind_group,
                        ..
                    }) => (rows, cols, data, dimensions, dot_bind_group),
                    _ => return Err(MatrixDotError(String::from("Matrix Variants do not match"))),
                };

                // Create the output matrix to use as the return matrix
                let output = GPUMatrix::with_shape((*rows, *b_cols), device.clone(), queue.clone());

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
                    compute_pass.set_bind_group(0, dot_bind_group, &[]);
                    compute_pass.set_bind_group(1, b_dot_bind_group, &[]);
                    compute_pass.set_bind_group(2, &output.get_writable_bind_group(&device), &[]);

                    // Dispatch the workgroups
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                device.poll(Maintain::Wait);

                queue.submit(Some(encoder.finish()));

                // print_buffer(&output_buf, &device, "Output");

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
                rows: _,
                cols,
                data,
            }) => {
                let inner_index = index.0 * *cols + index.1;
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
                rows: _,
                cols,
                data,
            }) => {
                let inner_index = index.0 * *cols + index.1;
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
            Matrix::CPU(CPUMatrix {
                rows,
                cols,
                data: _,
            }) => {
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
                        let index = i * *cols + j;
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

    // #[test]
    fn test_rand_with_shape() {
        let mat = Matrix::rand_with_shape((10, 5));

        println!("{}", mat);
        assert!(true);
    }

    // #[test]
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

    // #[test]
    fn test_matrix_dot() {
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

        println!("A: {}", mat1);
        println!("B: {}", mat2);
        println!(
            "Result: {}",
            mat1.dot(&mat2).expect("Failed to compute dot product")
        );

        assert!(true);
    }
}
