use std::{error::Error, rc::Rc};

use damage::csv::*;
use gpu_math::math::matrix::Matrix;
use pollster::FutureExt;
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions, include_wgsl,
};

fn init_params() -> (Matrix, Matrix, Matrix, Matrix) {
    let mut weights_1 = Matrix::with_shape((784, 10));
    for i in 0..weights_1.rows() {
        for j in 0..weights_1.cols() {
            weights_1[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    let mut biases_1 = Matrix::with_shape((10, 1000));
    for i in 0..biases_1.rows() {
        let r = rand::random_range(-0.5..=0.5);
        for j in 0..biases_1.cols() {
            biases_1[(i, j)] = r;
        }
    }

    let mut weights_2 = Matrix::with_shape((10, 10));
    for i in 0..weights_2.rows() {
        for j in 0..weights_2.cols() {
            weights_2[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    let mut biases_2 = Matrix::with_shape((10, 1000));
    for i in 0..biases_2.rows() {
        let r = rand::random_range(-0.5..=0.5);
        for j in 0..biases_2.cols() {
            biases_2[(i, j)] = r;
        }
    }

    (weights_1, biases_1, weights_2, biases_2)
}

fn softmax(z: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let top = z.exp()?;
    let bot = top.sum()?;

    Ok(top.mult(1.0 / bot)?)
}

fn forward_prop(
    w1: &Matrix,
    b1: &Matrix,
    w2: &Matrix,
    b2: &Matrix,
    x: &Matrix,
) -> Result<(Matrix, Matrix, Matrix, Matrix), Box<dyn Error>> {
    let mut z1 = x.dot(&w1)?.transpose().add(&b1)?;
    let relu = z1
        .add_custom_single_op_pipeline(include_wgsl!("shaders/relu.wgsl"))
        .unwrap();
    let a1 = z1.run_custom_single_op_pipeline(relu)?;

    let z2 = w2.dot(&a1)?.add(&b2)?;
    let a2 = softmax(&z2)?;

    Ok((z1, a1, z2, a2))
}

fn one_hot(y: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let mut one_hot_y = Matrix::with_shape((y.rows(), 10));

    for i in 0..y.rows() {
        let num = y[(i, 0)] as usize;

        one_hot_y[(i, num)] = 1.0;
    }

    // Ok(one_hot_y.buf(y.device()?.clone(), y.queue()?.clone()))
    Ok(one_hot_y)
}

fn back_prop(
    z1: &mut Matrix,
    a1: &Matrix,
    z2: &Matrix,
    a2: &Matrix,
    w2: &Matrix,
    x: &Matrix,
    y: &Matrix,
    m: f32,
) -> Result<(Matrix, f32, Matrix, f32), Box<dyn Error>> {
    let mut one_hot_y = one_hot(y)?;
    one_hot_y = one_hot_y
        .buf(z1.device()?.clone(), z1.queue()?.clone())
        .transpose();
    println!("ohy: {} {}", one_hot_y.rows(), one_hot_y.cols());
    let dz2 = a2.sub(&one_hot_y)?;
    let dw2 = dz2.dot(&a1.transposed())?.mult(1.0 / m)?;
    let db2 = dz2.sum()? / m;

    let index = z1
        .add_custom_single_op_pipeline(include_wgsl!("shaders/d_relu.wgsl"))
        .unwrap();

    let dz1 = w2
        .transposed()
        .dot(&dz2)?
        .dot(&z1.run_custom_single_op_pipeline(index)?.transposed())?;

    println!("dz1: {} {}", dz1.rows(), dz1.cols());
    println!("x: {} {}", x.cols(), x.rows());
    let dw1 = dz1.dot(&x)?.mult(1.0 / m)?;
    let db1 = dz1.sum()? / m;

    Ok((dw1, db1, dw2, db2))
}

fn update_params(
    w1: &mut Matrix,
    b1: &mut Matrix,
    w2: &mut Matrix,
    b2: &mut Matrix,
    dw1: &Matrix,
    db1: f32,
    dw2: &Matrix,
    db2: f32,
    alpha: f32,
) -> Result<(), Box<dyn Error>> {
    *w1 = w1.sub(&dw1.mult(alpha)?)?;
    let mut m1 = Matrix::with_shape((b1.rows(), b1.cols()));
    m1 = m1.buf(b1.device()?.clone(), b1.queue()?.clone());
    *b1 = b1.sub(&m1.mult(alpha * db1)?)?;

    *w2 = w2.sub(&dw2.mult(alpha)?)?;
    let mut m2 = Matrix::with_shape((b2.rows(), b2.cols()));
    m2 = m2.buf(b2.device()?.clone(), b2.queue()?.clone());
    *b2 = b2.sub(&m2.mult(alpha * db2)?)?;

    Ok(())
}

fn gradient_descent(
    x: &Matrix,
    y: &Matrix,
    iterations: usize,
    alpha: f32,
    device: Rc<Device>,
    queue: Rc<Queue>,
) -> Result<(Matrix, Matrix, Matrix, Matrix), Box<dyn Error>> {
    let (mut w1, mut b1, mut w2, mut b2) = init_params();

    w1 = w1.buf(device.clone(), queue.clone());
    b1 = b1.buf(device.clone(), queue.clone());
    w2 = w2.buf(device.clone(), queue.clone());
    b2 = b2.buf(device.clone(), queue.clone());

    for i in 0..iterations {
        let (mut z1, a1, z2, a2) = forward_prop(&w1, &b1, &w2, &b2, x)?;
        println!("z1: {} {}", z1.rows(), z1.cols());
        println!("a1: {} {}", a1.rows(), a1.cols());
        println!("z2: {} {}", z2.rows(), z2.cols());
        println!("a2: {} {}", a2.rows(), a2.cols());
        let (dw1, db1, dw2, db2) = back_prop(&mut z1, &a1, &z2, &a2, &w2, x, y, 1000.0)?;

        update_params(
            &mut w1, &mut b1, &mut w2, &mut b2, &dw1, db1, &dw2, db2, alpha,
        )?;

        if i % 10 == 0 {
            println!("Iteration: {}", i);
        }
    }

    Ok((w1, b1, w2, b2))
}

fn main() {
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

    let data = parse_csv("../../test_files/mnist_train.csv").expect("Failed");

    let label_train = data
        .column_slice("label", 0..1000)
        .expect("Failed")
        .iter()
        .map(|&label_data| label_data.as_float().expect("Failed"))
        .collect::<Vec<f32>>();

    let mut label_inputs = Matrix::with_shape((1000, 1));

    for i in 0..label_inputs.rows() {
        label_inputs[(i, 0)] = label_train[i];
    }

    // label_inputs = label_inputs.buf(device.clone(), queue.clone());

    let image_train = data
        .columns_slice("1x1"..="28x28", 0..1000)
        .expect("Failed")
        .iter()
        .map(|&data_array| {
            data_array
                .iter()
                .map(|data| data.as_float().expect("Failed") / 255.0)
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();

    let mut training_inputs = Matrix::with_shape((1000, 784));

    for i in 0..training_inputs.rows() {
        for j in 0..training_inputs.cols() {
            training_inputs[(i, j)] = image_train[i][j];
        }
    }

    training_inputs = training_inputs.buf(device.clone(), queue.clone());
    // training_inputs = training_inputs.transpose();

    let (w1, b1, w2, b2) =
        gradient_descent(&training_inputs, &label_inputs, 500, 0.1, device, queue).expect("Failed");

    println!("B2: {}", b2);
}
