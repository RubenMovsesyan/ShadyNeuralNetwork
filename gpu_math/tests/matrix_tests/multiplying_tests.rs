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
    fn test_cpu_scalar_mult_in_place() {
        let mut mat = Matrix::with_shape((5, 6));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let index = i * mat.cols() + j;
                mat[(i, j)] = (index + 1) as f32;
            }
        }

        println!("Before Mult: {}", mat);
        mat.mult_in_place(12.0).expect("Could Not Multiply Matrix");
        println!("After Mult: {}", mat);

        assert!(true)
    }

    #[test]
    fn test_gpu_scalar_mult_in_place() {
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
        mat.mult_in_place(12.0).expect("Could Not Multiply Matrix");
        println!("After Mult: {}", mat);

        assert!(true);
    }

    #[test]
    fn test_cpu_scalar_mult_into() {
        let mut mat = Matrix::with_shape((5, 6));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let index = i * mat.cols() + j;
                mat[(i, j)] = (index + 1) as f32;
            }
        }

        let mut output = Matrix::with_shape((5, 6));

        println!("Before Mult: {}", output);
        _ = Matrix::mult_into(&mat, 12.0, &mut output);
        println!("After Mult: {}", output);

        assert!(true)
    }

    #[test]
    fn test_gpu_scalar_mult_into() {
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

        let mut output = Matrix::with_shape((5, 6));

        mat = mat.buf(device.clone(), queue.clone());
        output = output.buf(device.clone(), queue.clone());

        println!("Before Mult: {}", output);
        _ = Matrix::mult_into(&mat, 12.0, &mut output);
        println!("After Mult: {}", output);

        assert!(true);
    }

    #[test]
    fn test_cpu_elem_mult_in_place() {
        let mut mat = Matrix::with_shape((5, 6));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let index = i * mat.cols() + j;
                mat[(i, j)] = (index + 1) as f32;
            }
        }

        let mut other = Matrix::with_shape((5, 6));
        for i in 0..other.rows() {
            for j in 0..other.cols() {
                let index = i * other.cols() + j;
                other[(i, j)] = (index + 1) as f32;
            }
        }

        println!("Mat before: {}", mat);
        mat.elem_mult_in_place(&other).expect("Failed");
        println!("Mat after: {}", mat);
    }

    #[test]
    fn test_gpu_elem_mult_in_place() {
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

        let mut other = Matrix::with_shape((5, 6));
        for i in 0..other.rows() {
            for j in 0..other.cols() {
                let index = i * other.cols() + j;
                other[(i, j)] = (index + 1) as f32;
            }
        }
        mat = mat.buf(device.clone(), queue.clone());
        other = other.buf(device.clone(), queue.clone());

        println!("Mat before: {}", mat);
        mat.elem_mult_in_place(&other).expect("Failed");
        println!("Mat after: {}", mat);
    }
}
