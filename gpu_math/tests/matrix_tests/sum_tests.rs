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
}
