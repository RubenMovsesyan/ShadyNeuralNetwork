use std::io::{Write, stdout};
use std::os::unix::fs::FileExt;

use activation::ActivationFunction;
#[allow(unused_imports)]
use log::*;
use loss::LossFunction;

use shady_neural_net::*;

use std::fs::File;

fn create_neural_net() -> Result<NeuralNet, Box<dyn std::error::Error>> {
    let mut neural_net = NeuralNet::new().expect("Could not initialize Neural Net");
    neural_net
        .add_input_layer(784)?
        .add_dense_layer(10, ActivationFunction::ReLU)?
        .add_dense_layer(10, ActivationFunction::ReLU)?
        .add_output_layer(10)?;

    neural_net.set_learning_rate(0.1);
    neural_net
        .set_loss_function(LossFunction::LogLoss)
        .expect("Could Not Set the loss function of the network");

    Ok(neural_net)
}

fn print_progress(progress: usize, total: usize) {
    let bar_size = 50;
    let ratio = progress as f32 / total as f32;
    let amount = (bar_size as f32 * ratio) as usize;
    print!("\r[");
    for _ in 0..amount {
        print!("=");
    }
    for _ in amount..bar_size - 1 {
        print!(" ");
    }
    print!("]");
}

fn get_accuracy(prediction: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    println!("\nPred: {:#?} Exp: {:#?}", prediction, expected);
    prediction
        .iter()
        .zip(expected.iter())
        .map(|(pred, exp)| pred - exp)
        .sum::<f32>()
        / expected.len() as f32
}

#[allow(dead_code)]
fn train() {
    let neural_net = create_neural_net().expect("Could not create neural net");
    let passes = 5000;

    // TEMP
    let image_file = File::open("test_files/train_images").expect("R");
    let mut image_buffer = vec![0; 784];

    let label_file = File::open("test_files/train_labels").expect("R");
    let mut label_buffer = vec![0];
    let mut v = vec![0.0; 10];

    for i in 0..passes {
        image_file
            .read_exact_at(&mut image_buffer, 784 * i as u64)
            .expect("F");
        label_file
            .read_exact_at(&mut label_buffer, i as u64)
            .expect("F");

        v[..].fill(0.0);

        v[label_buffer[0] as usize] = 1.0;

        let input = image_buffer
            .iter()
            .map(|value| (*value as f32) / 255.0)
            .collect::<Vec<f32>>();
        _ = neural_net.feed_forward(&input).expect("C");

        neural_net.compute_loss(&v).expect("G");
        neural_net.back_propogate();
        let predictions = neural_net.get_output();
        print_progress(i, passes);
        print!(" Accuracy {:>5.2}", get_accuracy(&predictions, &v));

        _ = stdout().flush();
    }

    let cost = neural_net.get_cost().expect("G");
    _ = neural_net.save_model_to_file("test_files/nn_test.json");

    println!();
    println!("Done! With Cost: {cost}");
}

#[allow(dead_code)]
fn test() {
    let mut neural_net = NeuralNet::load_model_from_file("test_files/nn_test.json").expect("F");
    neural_net.set_loss_function(LossFunction::MSE).expect("G");

    let image_file = File::open("test_files/train_images").expect("R");
    let mut image_buffer = vec![0; 784];

    let label_file = File::open("test_files/train_labels").expect("R");
    let mut label_buffer = vec![0];
    let mut v = vec![0.0; 10];

    for i in 1000..1500 {
        image_file
            .read_exact_at(&mut image_buffer, 784 * i as u64)
            .expect("F");
        label_file
            .read_exact_at(&mut label_buffer, i as u64)
            .expect("F");

        v[..].fill(0.0);

        v[label_buffer[0] as usize] = 1.0;

        let input = image_buffer
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<f32>>();

        _ = neural_net.feed_forward(&input);
        let vals = neural_net.get_output();

        println!("{:#?} {:#?}", vals, v);
    }
}

fn main() {
    pretty_env_logger::init();

    train();
    // test();
    // let (image, label) = get_image_training_data(
    //     "test_files/train_images/image_array_0",
    //     "test_files/train_labels",
    //     0,
    // )
    // .expect("Could Not get Image data");

    // println!("Image: {:#?}", image);
    // println!("Label: {:#?}", label);
}
