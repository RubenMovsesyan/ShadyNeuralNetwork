#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use pollster::FutureExt;
    use wgpu::{
        Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        PowerPreference, RequestAdapterOptions,
    };

    use gpu_math::math::matrix::*;

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
    fn test_cpu_dot_into() {
        let mut mat1 = Matrix::with_shape((3, 4));
        let mut mat2 = Matrix::with_shape((4, 2));
        let mut output = Matrix::with_shape((3, 2));

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

        println!("Output Before: {}", output);

        _ = Matrix::dot_into(&mat1, &mat2, &mut output);

        println!("Output After: {}", output);

        assert!(true);
    }

    #[test]
    fn test_gpu_dot_into() {
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
        let mut output = Matrix::with_shape((3, 2));

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
        output = output.buf(device.clone(), queue.clone());

        println!("Output Before: {}", output);

        _ = Matrix::dot_into(&mat1, &mat2, &mut output);

        println!("Output After: {}", output);

        assert!(true);
    }
}
