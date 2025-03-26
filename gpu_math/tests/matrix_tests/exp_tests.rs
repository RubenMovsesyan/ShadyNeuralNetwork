#[cfg(test)]
mod test {
    use std::rc::Rc;

    use pollster::FutureExt;
    use wgpu::{
        Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        PowerPreference, RequestAdapterOptions,
    };

    use gpu_math::math::matrix::*;

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
    fn test_cpu_exp_in_place() {
        let mut mat = Matrix::with_shape((5, 6));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let index = i * mat.cols() + j;
                mat[(i, j)] = (index + 1) as f32;
            }
        }

        println!("Before Exp: {}", mat);
        mat.exp_in_place().expect("Could Not Multiply Matrix");
        println!("After Exp: {}", mat);

        assert!(true)
    }

    #[test]
    fn test_gpu_exp_in_place() {
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
        mat.exp_in_place().expect("Could Not Multiply Matrix");
        println!("After Exp: {}", mat);

        assert!(true);
    }
}
