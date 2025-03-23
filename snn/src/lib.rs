use std::error::Error;

use gpu_math::math::matrix::Matrix;
use wgpu::include_wgsl;

fn init_params() -> (Matrix, Matrix, Matrix, Matrix) {
    let mut weights_1 = Matrix::with_shape((10, 784));
    for i in 0..weights_1.rows() {
        for j in 0..weights_1.cols() {
            weights_1[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    let mut biases_1 = Matrix::with_shape((10, 1));
    for i in 0..biases_1.rows() {
        for j in 0..biases_1.cols() {
            biases_1[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    let mut weights_2 = Matrix::with_shape((10, 10));
    for i in 0..weights_2.rows() {
        for j in 0..weights_2.cols() {
            weights_2[(i, j)] = rand::random_range(-0.5..=0.5);
        }
    }

    let mut biases_2 = Matrix::with_shape((10, 1));
    for i in 0..biases_2.rows() {
        for j in 0..biases_2.cols() {
            biases_2[(i, j)] = rand::random_range(-0.5..=0.5);
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
    let mut z1 = w1.dot(&x)?.add(&b1)?;
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

    Ok(one_hot_y.buf(y.device()?.clone(), y.queue()?.clone()))
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
    let one_hot_y = one_hot(y)?;
    let dz2 = a2.sub(&one_hot_y)?;
    let dw2 = dz2.dot(&a1.transposed())?.mult(1.0 / m)?;
    let db2 = dz2.sum()? / m;

    let index = z1
        .add_custom_single_op_pipeline(include_wgsl!("shaders/d_relu.wgsl"))
        .unwrap();

    let dz1 = w2
        .transposed()
        .dot(&dz2)?
        .dot(&z1.run_custom_single_op_pipeline(index)?)?;
    let dw1 = dz1.dot(&x.transposed())?.mult(1.0 / m)?;
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
) -> Result<(), Box<dyn Error>> {
    let (mut w1, mut b1, mut w2, mut b2) = init_params();

    for i in 0..iterations {
        let (mut z1, a1, z2, a2) = forward_prop(&w1, &b1, &w2, &b2, x)?;
        let (dw1, db1, dw2, db2) = back_prop(&mut z1, &a1, &z2, &a2, &w2, x, y, 60000.0)?;

        update_params(
            &mut w1, &mut b1, &mut w2, &mut b2, &dw1, db1, &dw2, db2, alpha,
        )?;

        if i % 10 == 0 {
            println!("Iteration: {}", i);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_params() {
        let (w1, b1, w2, b2) = init_params();

        println!("W1: {}", w1);
        println!("b1: {}", b1);
        println!("W2: {}", w2);
        println!("b2: {}", b2);

        assert!(true);
    }
}
