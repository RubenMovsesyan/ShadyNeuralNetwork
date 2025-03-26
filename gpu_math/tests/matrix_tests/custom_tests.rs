#[cfg(test)]
mod test {
    use std::rc::Rc;

    use pollster::FutureExt;
    use wgpu::{
        Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        PowerPreference, RequestAdapterOptions, include_wgsl,
    };

    use gpu_math::math::matrix::*;

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
            .add_custom_single_op_pipeline(include_wgsl!("../test_shaders/relu.wgsl"))
            .expect("Failed to Add Pipeline");

        println!("Before Compute: {}", mat);
        println!(
            "After Compute: {}",
            mat.run_custom_single_op_pipeline(index)
                .expect("Failed to Run Custom Compute")
        );
    }

    #[test]
    fn test_custom_pipeline_into() {
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
        let mut output = Matrix::with_shape((10, 10));

        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                let value = (i * mat.cols() + j) as f32 - ((mat.rows() * mat.cols()) as f32 / 2.0);
                mat[(i, j)] = value;
            }
        }

        mat = mat.buf(device.clone(), queue.clone());
        output = output.buf(device.clone(), queue.clone());
        let index = mat
            .add_custom_single_op_pipeline(include_wgsl!("../test_shaders/relu.wgsl"))
            .expect("Failed to Add Pipeline");

        println!("Before Compute: {}", mat);
        _ = Matrix::run_custom_single_op_pipeline_into(&mat, index, &mut output);
        println!("After Compute: {}", output);
    }
}
