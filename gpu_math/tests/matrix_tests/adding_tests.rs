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
    fn test_cpu_add_in_place() {
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

        println!("Add A: {}", mat1);
        mat1.add_in_place(&mat2).expect("Could not add matrices");
        println!("Add B: {}", mat2);
        println!("Add Result: {}", mat1);
        assert!(true);
    }

    #[test]
    fn test_gpu_add_in_place() {
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

        println!("Add A: {}", mat1);
        mat1.add_in_place(&mat2).expect("Could not add matrices");
        println!("Add B: {}", mat2);
        println!("Add Result: {}", mat1);
        assert!(true);
    }

    #[test]
    fn test_cpu_add_into() {
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

        let mut output_mat = Matrix::with_shape((5, 6));

        println!("Output Before: {}", output_mat);

        _ = Matrix::add_into(&mat1, &mat2, &mut output_mat);

        println!("Output After: {}", output_mat);
        assert!(true);
    }

    #[test]
    fn test_gpu_add_into() {
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

        let mut output_mat = Matrix::with_shape((5, 6));
        output_mat = output_mat.buf(device.clone(), queue.clone());

        println!("Output Before: {}", output_mat);

        _ = Matrix::add_into(&mat1, &mat2, &mut output_mat);

        println!("Output After: {}", output_mat);
    }

    #[test]
    fn test_cpu_vectored_add() {
        let mut mat = Matrix::with_shape((5, 6));
        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                mat[(i, j)] = (i * mat.cols() + j) as f32;
            }
        }

        let mut vec = Matrix::with_shape((5, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_add(&vec).expect("Failed"));

        vec = Matrix::with_shape((1, 5));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_add(&vec).expect("Failed"));

        vec = Matrix::with_shape((6, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_add(&vec).expect("Failed"));

        vec = Matrix::with_shape((1, 6));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_add(&vec).expect("Failed"));

        vec = Matrix::with_shape((2, 6));
        for i in 0..vec.rows() {
            for j in 0..vec.cols() {
                vec[(0, i)] = (i * mat.cols() + j) as f32;
            }
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);

        assert!(mat.vectored_add(&vec).is_err())
    }

    #[test]
    fn test_gpu_vectored_add() {
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
                mat[(i, j)] = (i * mat.cols() + j) as f32;
            }
        }

        mat = mat.buf(device.clone(), queue.clone());

        let mut vec = Matrix::with_shape((5, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_add(&vec).expect("Failed"));

        vec = Matrix::with_shape((1, 5));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_add(&vec).expect("Failed"));

        vec = Matrix::with_shape((6, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_add(&vec).expect("Failed"));

        vec = Matrix::with_shape((1, 6));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_add(&vec).expect("Failed"));

        vec = Matrix::with_shape((2, 6));
        for i in 0..vec.rows() {
            for j in 0..vec.cols() {
                vec[(0, i)] = (i * mat.cols() + j) as f32;
            }
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);

        assert!(mat.vectored_add(&vec).is_err())
    }

    #[test]
    fn test_cpu_vectored_add_in_place() {
        let mut mat = Matrix::with_shape((5, 6));
        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                mat[(i, j)] = (i * mat.cols() + j) as f32;
            }
        }

        let mut vec = Matrix::with_shape((5, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        println!("Mat: {}", mat);
        mat.vectored_add_in_place(&vec).expect("Failed");
        println!("Vec: {}", vec);
        println!("Result: {}", mat);

        vec = Matrix::with_shape((1, 5));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        println!("Mat: {}", mat);
        mat.vectored_add_in_place(&vec).expect("Failed");
        println!("Vec: {}", vec);
        println!("Result: {}", mat);

        vec = Matrix::with_shape((6, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        println!("Mat: {}", mat);
        mat.vectored_add_in_place(&vec).expect("Failed");
        println!("Vec: {}", vec);
        println!("Result: {}", mat);

        vec = Matrix::with_shape((1, 6));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        println!("Mat: {}", mat);
        mat.vectored_add_in_place(&vec).expect("Failed");
        println!("Vec: {}", vec);
        println!("Result: {}", mat);

        vec = Matrix::with_shape((2, 6));
        for i in 0..vec.rows() {
            for j in 0..vec.cols() {
                vec[(0, i)] = (i * mat.cols() + j) as f32;
            }
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);

        assert!(mat.vectored_add_in_place(&vec).is_err())
    }

    #[test]
    fn test_gpu_vectored_add_in_place() {
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
                mat[(i, j)] = (i * mat.cols() + j) as f32;
            }
        }

        mat = mat.buf(device.clone(), queue.clone());

        let mut vec = Matrix::with_shape((5, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        mat.vectored_add_in_place(&vec).expect("Failed");
        println!("Vec: {}", vec);
        println!("Result: {}", mat);

        vec = Matrix::with_shape((1, 5));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        mat.vectored_add_in_place(&vec).expect("Failed");
        println!("Vec: {}", vec);
        println!("Result: {}", mat);

        vec = Matrix::with_shape((6, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        mat.vectored_add_in_place(&vec).expect("Failed");
        println!("Vec: {}", vec);
        println!("Result: {}", mat);

        vec = Matrix::with_shape((1, 6));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        mat.vectored_add_in_place(&vec).expect("Failed");
        println!("Vec: {}", vec);
        println!("Result: {}", mat);

        vec = Matrix::with_shape((2, 6));
        for i in 0..vec.rows() {
            for j in 0..vec.cols() {
                vec[(0, i)] = (i * mat.cols() + j) as f32;
            }
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);

        assert!(mat.vectored_add_in_place(&vec).is_err())
    }

    #[test]
    fn test_cpu_vectored_add_into() {
        let mut mat = Matrix::with_shape((5, 6));
        let mut output = Matrix::with_shape((5, 6));
        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                mat[(i, j)] = (i * mat.cols() + j) as f32;
            }
        }

        let mut vec = Matrix::with_shape((5, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", output);

        vec = Matrix::with_shape((1, 5));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", output);

        vec = Matrix::with_shape((6, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", output);

        vec = Matrix::with_shape((1, 6));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", output);

        vec = Matrix::with_shape((2, 6));
        for i in 0..vec.rows() {
            for j in 0..vec.cols() {
                vec[(0, i)] = (i * mat.cols() + j) as f32;
            }
        }
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);

        assert!(Matrix::vectored_add_into(&mat, &vec, &mut output).is_err());
    }

    #[test]
    fn test_gpu_vectored_add_into() {
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
        let mut output = Matrix::with_shape((5, 6));
        output = output.buf(device.clone(), queue.clone());
        for i in 0..mat.rows() {
            for j in 0..mat.cols() {
                mat[(i, j)] = (i * mat.cols() + j) as f32;
            }
        }

        mat = mat.buf(device.clone(), queue.clone());

        let mut vec = Matrix::with_shape((5, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", output);

        vec = Matrix::with_shape((1, 5));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", output);

        vec = Matrix::with_shape((6, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", output);

        vec = Matrix::with_shape((1, 6));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());
        _ = Matrix::vectored_add_into(&mat, &vec, &mut output);

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", output);

        vec = Matrix::with_shape((2, 6));
        for i in 0..vec.rows() {
            for j in 0..vec.cols() {
                vec[(0, i)] = (i * mat.cols() + j) as f32;
            }
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);

        assert!(Matrix::vectored_add_into(&mat, &vec, &mut output).is_err());
    }
}
