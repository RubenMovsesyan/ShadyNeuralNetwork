use damage::csv::parse_csv;
use gpu_math::math::matrix::Matrix;
use shady_neural_net::{
    layers::{activation_function::ActivationFunction, loss_function::LossFunction},
    neural_network::NeuralNetwork,
};

const BATCH_SIZE: usize = 0;
const BATCHES: usize = 10;

fn one_hot_y(y: &Matrix) -> Matrix {
    let mut output = Matrix::with_shape((10, y.rows()));

    for i in 0..output.rows() {
        for j in 0..output.cols() {
            output[(i, j)] = if i as f32 == y[(j, 0)] { 1.0 } else { 0.0 };
        }
    }

    output
}

fn main() {
    let mut shady_neural_net = NeuralNetwork::new(784, BATCH_SIZE, LossFunction::LogLoss, 0.1);

    let data = parse_csv("../../test_files/mnist_train.csv").expect("Failed");

    let label_train = data
        .column_slice("label", 0..(BATCH_SIZE * BATCHES))
        .expect("Failed")
        .iter()
        .map(|&label_data| label_data.as_float().expect("Failed"))
        .collect::<Vec<f32>>();

    let mut label_inputs = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        label_inputs.push(Matrix::with_shape((BATCH_SIZE, 1)));
    }

    // for i in 0..label_inputs.rows() {
    //     label_inputs[(i, 0)] = label_train[i];
    // }

    // label_inputs = one_hot_y(&label_inputs);

    for (batch_index, label_batch) in label_inputs.iter_mut().enumerate() {
        for i in 0..label_batch.rows() {
            label_batch[(i, 0)] = label_train[BATCH_SIZE * batch_index + i];
        }

        *label_batch = one_hot_y(&label_batch);
    }

    let image_train = data
        .columns_slice("1x1"..="28x28", 0..(BATCH_SIZE * BATCHES))
        .expect("Failed")
        .iter()
        .map(|&data_array| {
            data_array
                .iter()
                .map(|data| data.as_float().expect("Failed") / 255.0)
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();

    // let mut training_inputs = Matrix::with_shape((784, BATCH_SIZE));
    let mut training_inputs = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        training_inputs.push(Matrix::with_shape((784, BATCH_SIZE)));
    }

    for (batch_index, training_batch) in training_inputs.iter_mut().enumerate() {
        for i in 0..training_batch.rows() {
            for j in 0..training_batch.cols() {
                training_batch[(i, j)] = image_train[BATCH_SIZE * batch_index + j][i];
            }
        }
    }

    // for i in 0..training_inputs.rows() {
    //     for j in 0..training_inputs.cols() {
    //         training_inputs[(i, j)] = image_train[j][i];
    //     }
    // }

    // shady_neural_net.add_input_batch(training_inputs);
    // shady_neural_net.add_label_batch(label_inputs);

    for (input_batch, label_batch) in training_inputs.into_iter().zip(label_inputs.into_iter()) {
        shady_neural_net.add_input_batch(input_batch);
        shady_neural_net.add_label_batch(label_batch);
    }

    shady_neural_net.add_layer(10, ActivationFunction::ReLU);
    shady_neural_net.add_layer(10, ActivationFunction::Softmax);

    for iteration in 0..100 {
        for batch_number in 0..shady_neural_net.num_batches() {
            shady_neural_net.feed_forward(batch_number).expect("Failed");
            shady_neural_net
                .back_propogate(batch_number)
                .expect("Failed");
            shady_neural_net.update_parameters().expect("Failed");

            if iteration % 10 == 0 {
                println!("Iteration: {}", iteration);
                println!(
                    "Cost: {} {}",
                    batch_number,
                    shady_neural_net.get_cost(batch_number).expect("Failed")
                );
            }
        }
    }

    // for output in outputs.iter() {
    //     println!("{}", output);
    // }

    // println!(
    //     "Costs: {:#?}",
    //     shady_neural_net.get_costs(&outputs).expect("Failed")
    // );
}
