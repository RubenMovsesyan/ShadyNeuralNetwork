#[cfg(test)]
mod tests {

    use std::rc::Rc;

    use pollster::FutureExt;
    use wgpu::{
        Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        PowerPreference, RequestAdapterOptions, include_wgsl,
    };

    use gpu_math::math::matrix::*;

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

    #[test]
    fn test_cpu_vectored_sub() {
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
        println!("Result: {}", mat.vectored_sub(&vec).expect("Failed"));

        vec = Matrix::with_shape((1, 5));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_sub(&vec).expect("Failed"));

        vec = Matrix::with_shape((6, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_sub(&vec).expect("Failed"));

        vec = Matrix::with_shape((1, 6));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_sub(&vec).expect("Failed"));

        vec = Matrix::with_shape((2, 6));
        for i in 0..vec.rows() {
            for j in 0..vec.cols() {
                vec[(0, i)] = (i * mat.cols() + j) as f32;
            }
        }

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);

        assert!(mat.vectored_sub(&vec).is_err())
    }

    #[test]
    fn test_gpu_vectored_sub() {
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
        println!("Result: {}", mat.vectored_sub(&vec).expect("Failed"));

        vec = Matrix::with_shape((1, 5));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_sub(&vec).expect("Failed"));

        vec = Matrix::with_shape((6, 1));
        for i in 0..vec.rows() {
            vec[(i, 0)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_sub(&vec).expect("Failed"));

        vec = Matrix::with_shape((1, 6));
        for i in 0..vec.cols() {
            vec[(0, i)] = i as f32;
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);
        println!("Result: {}", mat.vectored_sub(&vec).expect("Failed"));

        vec = Matrix::with_shape((2, 6));
        for i in 0..vec.rows() {
            for j in 0..vec.cols() {
                vec[(0, i)] = (i * mat.cols() + j) as f32;
            }
        }

        vec = vec.buf(device.clone(), queue.clone());

        println!("Mat: {}", mat);
        println!("Vec: {}", vec);

        assert!(mat.vectored_sub(&vec).is_err())
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
    fn test_cpu_sub_into() {
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

        _ = Matrix::sub_into(&mat1, &mat2, &mut output_mat);

        println!("Output After: {}", output_mat);
        assert!(true);
    }

    #[test]
    fn test_gpu_sub_into() {
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

        _ = Matrix::sub_into(&mat1, &mat2, &mut output_mat);

        println!("Output After: {}", output_mat);
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
            .add_custom_single_op_pipeline(include_wgsl!("test_shaders/relu.wgsl"))
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
            .add_custom_single_op_pipeline(include_wgsl!("test_shaders/relu.wgsl"))
            .expect("Failed to Add Pipeline");

        println!("Before Compute: {}", mat);
        _ = Matrix::run_custom_single_op_pipeline_into(&mat, index, &mut output);
        println!("After Compute: {}", output);
    }
}
