use std::{error::Error, rc::Rc};

use damage::csv::*;
use gpu_math::math::matrix::Matrix;
use pollster::FutureExt;
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions, include_wgsl,
};

const NUM_INPUTS: usize = 10;

fn one_hot_y(y: &Matrix) -> Matrix {
    let mut output = Matrix::with_shape((y.rows(), 10));

    for i in 0..y.rows() {
        for j in 0..output.cols() {
            output[(i, j)] = if j as f32 == y[(i, 0)] { 1.0 } else { 0.0 };
        }
    }

    output
}

fn init_params(device: Rc<Device>, queue: Rc<Queue>) -> (Matrix, Matrix, Matrix, Matrix) {
    let mut w1 = Matrix::with_shape((10, 784));

    for i in 0..w1.rows() {
        for j in 0..w1.cols() {
            w1[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    let mut b1 = Matrix::with_shape((10, 1));

    for i in 0..b1.rows() {
        for j in 0..b1.cols() {
            b1[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    let mut w2 = Matrix::with_shape((10, 10));

    for i in 0..w2.rows() {
        for j in 0..w2.cols() {
            w2[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    let mut b2 = Matrix::with_shape((10, 1));

    for i in 0..b2.rows() {
        for j in 0..b2.cols() {
            b2[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    w1 = w1.buf(device.clone(), queue.clone());
    b1 = b1.buf(device.clone(), queue.clone());
    w2 = w2.buf(device.clone(), queue.clone());
    b2 = b2.buf(device.clone(), queue.clone());

    (w1, b1, w2, b2)
}

fn feed_forward(
    w1: &Matrix,
    b1: &Matrix,
    w2: &Matrix,
    b2: &Matrix,
    x: &Matrix,
) -> Result<(Matrix, Matrix, Matrix, Matrix), Box<dyn Error>> {
    let mut z1 = w1.dot(x)?.vectored_add(b1)?;
    let relu = z1
        .add_custom_single_op_pipeline(include_wgsl!("shaders/relu.wgsl"))
        .expect("Failed");
    let a1 = z1.run_custom_single_op_pipeline(relu)?;

    let mut z2 = w2.dot(&a1)?.vectored_add(b2)?;
    let softmax = z2
        .add_custom_single_op_pipeline(include_wgsl!("shaders/softmax.wgsl"))
        .expect("Failed");
    let a2 = z2.run_custom_single_op_pipeline(softmax)?;

    Ok((z1, a1, z2, a2))
}

fn back_prop(
    z1: &mut Matrix,
    a1: &Matrix,
    z2: &Matrix,
    a2: &Matrix,
    w2: &Matrix,
    x: &Matrix,
    y: &Matrix,
) -> Result<(Matrix, f32, Matrix, f32), Box<dyn Error>> {
    let dz2 = a2.sub(y)?;
    let dw2 = dz2.dot(&a1.transposed())?.mult(1.0 / NUM_INPUTS as f32)?;
    let db2 = dz2.sum()? / NUM_INPUTS as f32;

    let d_relu = z1
        .add_custom_single_op_pipeline(include_wgsl!("shaders/d_relu.wgsl"))
        .expect("Failed");
    let dz1 = w2
        .transposed()
        .dot(&dz2)?
        .elem_mult(&z1.run_custom_single_op_pipeline(d_relu)?)?;
    let dw1 = dz1.dot(&x.transposed())?.mult(1.0 / NUM_INPUTS as f32)?;
    let db1 = dz1.sum()? / NUM_INPUTS as f32;

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
    let mut b1_sub = Matrix::with_shape((b1.rows(), b1.cols()));
    for i in 0..b1_sub.rows() {
        for j in 0..b1_sub.cols() {
            b1_sub[(i, j)] = 1.0;
        }
    }
    b1_sub = b1_sub.buf(b1.device()?.clone(), b1.queue()?.clone());
    *b1 = b1.sub(&b1_sub.mult(alpha * db1)?)?;

    *w2 = w2.sub(&dw2.mult(alpha)?)?;
    let mut b2_sub = Matrix::with_shape((b2.rows(), b2.cols()));
    for i in 0..b2_sub.rows() {
        for j in 0..b2_sub.cols() {
            b2_sub[(i, j)] = 1.0;
        }
    }
    b2_sub = b2_sub.buf(b2.device()?.clone(), b2.queue()?.clone());
    *b2 = b2.sub(&b2_sub.mult(alpha * db2)?)?;

    Ok(())
}

fn gradient_descent(
    x: &Matrix,
    y: &Matrix,
    iterations: usize,
    alpha: f32,
    device: Rc<Device>,
    queue: Rc<Queue>,
) -> Result<f32, Box<dyn Error>> {
    let (mut w1, mut b1, mut w2, mut b2) = init_params(device.clone(), queue.clone());

    for i in 0..iterations {
        let (mut z1, a1, z2, a2) = feed_forward(&w1, &b1, &w2, &b2, x)?;
        let (dw1, db1, dw2, db2) = back_prop(&mut z1, &a1, &z2, &a2, &w2, x, y)?;
        update_params(
            &mut w1, &mut b1, &mut w2, &mut b2, &dw1, db1, &dw2, db2, alpha,
        )?;

        if i % 10 == 0 {
            println!("Iteration: {i}");
            // println!("W1: {}", w1);
            // println!("B1: {}", b1);
            // println!("W2: {}", w2);
            // println!("B2: {}", b2);
            println!("Diff: {}", a2.sub(y)?.sum()?);
        }
    }

    todo!()
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
        .column_slice("label", 0..NUM_INPUTS)
        .expect("Failed")
        .iter()
        .map(|&label_data| label_data.as_float().expect("Failed"))
        .collect::<Vec<f32>>();

    let mut label_inputs = Matrix::with_shape((NUM_INPUTS, 1));

    for i in 0..label_inputs.rows() {
        label_inputs[(i, 0)] = label_train[i];
    }

    label_inputs = one_hot_y(&label_inputs).transposed();
    label_inputs = label_inputs.buf(device.clone(), queue.clone());

    let image_train = data
        .columns_slice("1x1"..="28x28", 0..NUM_INPUTS)
        .expect("Failed")
        .iter()
        .map(|&data_array| {
            data_array
                .iter()
                .map(|data| data.as_float().expect("Failed") / 255.0)
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();

    let mut training_inputs = Matrix::with_shape((NUM_INPUTS, 784));

    for i in 0..training_inputs.rows() {
        for j in 0..training_inputs.cols() {
            training_inputs[(i, j)] = image_train[i][j];
        }
    }

    training_inputs = training_inputs
        .buf(device.clone(), queue.clone())
        .transposed();

    gradient_descent(&training_inputs, &label_inputs, 500, 0.1, device, queue).expect("Failed");
    // let (w1, b1, w2, b2) = init_params(device.clone(), queue.clone());
    // let (mut z1, a1, z2, a2) = feed_forward(&w1, &b1, &w2, &b2, &training_inputs).expect("Failed");

    // let (dw1, db1, dw2, db2) = back_prop(
    //     &mut z1,
    //     &a1,
    //     &z2,
    //     &a2,
    //     &w2,
    //     &training_inputs,
    //     &label_inputs,
    //     NUM_INPUTS as f32,
    // )
    // .expect("Failed");
}
