use std::{error::Error, rc::Rc};

use damage::csv::*;
use gpu_math::math::matrix::Matrix;
use pollster::FutureExt;
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions, include_wgsl,
};

const NUM_INPUTS: usize = 10;

struct NeuralNet {
    // Weights and biases
    w1: Matrix,
    b1: Matrix,
    w2: Matrix,
    b2: Matrix,

    // Gradients
    dw1: Matrix,
    db1: f32,
    dw2: Matrix,
    db2: f32,

    // Intermediary matrices
    z1: Matrix,
    z1_dotted: Matrix,
    z1_d_relu: Matrix,
    relu: usize,
    d_relu: usize,
    argmax: usize,
    abs: usize,
    a1: Matrix,
    z2: Matrix,
    z2_dotted: Matrix,
    softmax: usize,
    a2: Matrix,
    w2t_dotted: Matrix,

    // Intermediary Gradients
    dz2: Matrix,
    dz2_dotted: Matrix,
    dz1: Matrix,
    dz1_dotted: Matrix,

    // label matrix
    one_hot_y: Matrix,
    // data matrix
    x: Matrix,

    // Updating paramters
    b1_sub: Matrix,
    b2_sub: Matrix,

    diff_into: Matrix,

    // wgpu
    device: Rc<Device>,
    queue: Rc<Queue>,
}

impl NeuralNet {
    fn new() -> Self {
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

        // Create the weights and biases matrices
        let (w1, b1, w2, b2) = NeuralNet::init_params(&device, &queue);

        // Create the gradient matrices for the weights and biases
        let mut dw1 = Matrix::with_shape((w1.rows(), w1.cols()));
        let mut dw2 = Matrix::with_shape((w2.rows(), w2.cols()));

        dw1 = dw1.buf(device.clone(), queue.clone());
        dw2 = dw2.buf(device.clone(), queue.clone());

        let mut z1 = Matrix::with_shape((w1.rows(), NUM_INPUTS));
        let mut z1_dotted = Matrix::with_shape((z1.rows(), z1.cols()));
        let mut z1_d_relu = Matrix::with_shape((z1.rows(), z1.cols()));
        let mut a1 = Matrix::with_shape((z1.rows(), z1.cols()));
        let mut dz1 = Matrix::with_shape((z1.rows(), z1.cols()));

        z1 = z1.buf(device.clone(), queue.clone());
        let relu = z1
            .add_custom_single_op_pipeline(include_wgsl!("shaders/relu.wgsl"))
            .expect("Failed");
        let d_relu = z1
            .add_custom_single_op_pipeline(include_wgsl!("shaders/d_relu.wgsl"))
            .expect("Failed");
        z1_dotted = z1_dotted.buf(device.clone(), queue.clone());
        z1_d_relu = z1_d_relu.buf(device.clone(), queue.clone());
        a1 = a1.buf(device.clone(), queue.clone());
        dz1 = dz1.buf(device.clone(), queue.clone());

        let mut z2 = Matrix::with_shape((w2.rows(), a1.cols()));
        let mut z2_dotted = Matrix::with_shape((z2.rows(), z2.cols()));
        let mut a2 = Matrix::with_shape((z2.rows(), z2.cols()));
        let mut dz2 = Matrix::with_shape((z2.rows(), z2.cols()));

        z2 = z2.buf(device.clone(), queue.clone());
        let softmax = z2
            .add_custom_single_op_pipeline(include_wgsl!("shaders/softmax.wgsl"))
            .expect("Failed");
        z2_dotted = z2_dotted.buf(device.clone(), queue.clone());
        a2 = a2.buf(device.clone(), queue.clone());
        dz2 = dz2.buf(device.clone(), queue.clone());

        let argmax = a2
            .add_custom_single_op_pipeline(include_wgsl!("shaders/argmax_y.wgsl"))
            .expect("Failed");

        let mut w2t_dotted = Matrix::with_shape((w2.cols(), dz2.cols()));
        w2t_dotted = w2t_dotted.buf(device.clone(), queue.clone());

        let mut x = Matrix::with_shape((784, NUM_INPUTS));
        x = x.buf(device.clone(), queue.clone());

        let mut one_hot_y = Matrix::with_shape((10, NUM_INPUTS));
        one_hot_y = one_hot_y.buf(device.clone(), queue.clone());

        let mut dz1_dotted = Matrix::with_shape((dz1.rows(), x.rows()));
        let mut dz2_dotted = Matrix::with_shape((dz2.rows(), a1.rows()));
        dz2_dotted = dz2_dotted.buf(device.clone(), queue.clone());
        dz1_dotted = dz1_dotted.buf(device.clone(), queue.clone());

        let mut b1_sub = Matrix::with_shape((b1.rows(), b1.cols()));
        for i in 0..b1_sub.rows() {
            for j in 0..b1_sub.cols() {
                b1_sub[(i, j)] = 1.0;
            }
        }
        b1_sub = b1_sub.buf(device.clone(), queue.clone());

        let mut b2_sub = Matrix::with_shape((b2.rows(), b2.cols()));
        for i in 0..b2_sub.rows() {
            for j in 0..b2_sub.cols() {
                b2_sub[(i, j)] = 1.0;
            }
        }
        b2_sub = b2_sub.buf(device.clone(), queue.clone());

        let mut diff_into = Matrix::with_shape((one_hot_y.rows(), one_hot_y.cols()));
        diff_into = diff_into.buf(device.clone(), queue.clone());
        let abs = diff_into
            .add_custom_single_op_pipeline(include_wgsl!("shaders/abs.wgsl"))
            .expect("Failed");

        Self {
            w1,
            b1,
            w2,
            b2,
            dw1,
            db1: 0.0,
            dw2,
            db2: 0.0,
            z1,
            relu,
            d_relu,
            argmax,
            abs,
            z1_dotted,
            z1_d_relu,
            a1,
            dz1,
            z2,
            z2_dotted,
            softmax,
            a2,
            dz2,
            dz1_dotted,
            dz2_dotted,
            w2t_dotted,
            one_hot_y,
            x,
            b1_sub,
            b2_sub,
            diff_into,
            device,
            queue,
        }
    }

    fn init_params(device: &Rc<Device>, queue: &Rc<Queue>) -> (Matrix, Matrix, Matrix, Matrix) {
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

    fn set_labels(&mut self, one_hot_y: Matrix) {
        self.one_hot_y = one_hot_y.buf(self.device.clone(), self.queue.clone());
    }

    fn set_inputs(&mut self, x: Matrix) {
        self.x = x.buf(self.device.clone(), self.queue.clone());
    }

    fn feed_forward(&mut self) -> Result<(), Box<dyn Error>> {
        // Feed through the first layer
        Matrix::dot_into(&self.w1, &self.x, &mut self.z1_dotted)?;
        Matrix::vectored_add_into(&self.z1_dotted, &self.b1, &mut self.z1)?;
        Matrix::run_custom_single_op_pipeline_into(&self.z1, self.relu, &mut self.a1)?;

        // Feed through the second layer
        Matrix::dot_into(&self.w2, &self.a1, &mut self.z2_dotted)?;
        Matrix::vectored_add_into(&self.z2_dotted, &self.b2, &mut self.z2)?;
        Matrix::run_custom_single_op_pipeline_into(&self.z2, self.softmax, &mut self.a2)?;

        Ok(())
    }

    fn back_prop(&mut self) -> Result<(), Box<dyn Error>> {
        Matrix::sub_into(&self.a2, &self.one_hot_y, &mut self.dz2)?;
        Matrix::dot_into(&self.dz2, &self.a1.transposed(), &mut self.dz2_dotted)?;
        Matrix::mult_into(&self.dz2_dotted, 1.0 / NUM_INPUTS as f32, &mut self.dw2)?;
        self.db2 = self.dz2.sum()? / NUM_INPUTS as f32;

        Matrix::dot_into(&self.w2.transposed(), &self.dz2, &mut self.w2t_dotted)?;
        Matrix::run_custom_single_op_pipeline_into(&self.z1, self.d_relu, &mut self.z1_d_relu)?;
        Matrix::elem_mult_into(&self.w2t_dotted, &self.z1_d_relu, &mut self.dz1)?;
        Matrix::dot_into(&self.dz1, &self.x.transposed(), &mut self.dz1_dotted)?;
        Matrix::mult_into(&self.dz1_dotted, 1.0 / NUM_INPUTS as f32, &mut self.dw1)?;
        self.db1 = self.dz1.sum()? / NUM_INPUTS as f32;

        Ok(())
    }

    fn update_params(&mut self, alpha: f32) -> Result<(), Box<dyn Error>> {
        self.w1 = self.w1.sub(&self.dw1.mult(alpha)?)?;
        self.b1 = self.b1.sub(&self.b1_sub.mult(alpha * self.db1)?)?;

        self.w2 = self.w2.sub(&self.dw2.mult(alpha)?)?;
        self.b2 = self.b2.sub(&self.b2_sub.mult(alpha * self.db2)?)?;
        Ok(())
    }

    fn get_accuracy(&mut self) -> Result<f32, Box<dyn Error>> {
        let argmax = self.a2.run_custom_single_op_pipeline(self.argmax)?;
        Matrix::sub_into(&self.a2, &self.one_hot_y, &mut self.diff_into)?;
        let diff = self
            .diff_into
            .run_custom_single_op_pipeline(self.abs)?
            .sum()?;

        Ok((NUM_INPUTS as f32 * 2.0 - diff) / (NUM_INPUTS as f32 * 2.0))
    }
}

fn one_hot_y(y: &Matrix) -> Matrix {
    let mut output = Matrix::with_shape((y.rows(), 10));

    for i in 0..y.rows() {
        for j in 0..output.cols() {
            output[(i, j)] = if j as f32 == y[(i, 0)] { 1.0 } else { 0.0 };
        }
    }

    output
}

fn gradient_descent(label_inputs: Matrix, training_inputs: Matrix) -> Result<(), Box<dyn Error>> {
    let mut neural_network = NeuralNet::new();
    println!("Created Neural Network");
    neural_network.set_labels(label_inputs);
    println!("Set Labels");
    neural_network.set_inputs(training_inputs);
    println!("Set Inputs");

    for i in 0..500 {
        neural_network.feed_forward()?;
        neural_network.back_prop()?;
        neural_network.update_params(0.1)?;

        if i % 10 == 0 {
            println!("Iteration: {i}");
            println!("Accuracy: {}", neural_network.get_accuracy()?);
        }
    }
    Ok(())
}

fn main() {
    let data = parse_csv("../../test_files/mnist_train.csv").expect("Failed");

    let label_train = data
        .column_slice(&String::from("label"), 0..NUM_INPUTS)
        .expect("Failed")
        .iter()
        .map(|&label_data| label_data.as_float().expect("Failed"))
        .collect::<Vec<f32>>();

    let mut label_inputs = Matrix::with_shape((NUM_INPUTS, 1));

    for i in 0..label_inputs.rows() {
        label_inputs[(i, 0)] = label_train[i];
    }

    label_inputs = one_hot_y(&label_inputs).transposed();
    // label_inputs = label_inputs.buf(device.clone(), queue.clone());

    let image_train = data
        .columns_slice("1x1".to_string()..="28x28".to_string(), 0..NUM_INPUTS)
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

    match gradient_descent(label_inputs, training_inputs.transposed()) {
        Ok(_) => println!("Finished!"),
        Err(err) => println!("Error: {err}"),
    };
}
