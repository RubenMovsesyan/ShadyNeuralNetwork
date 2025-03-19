use std::io::{Write, stdout};
use std::os::unix::fs::FileExt;
use std::time::Instant;

use activation::ActivationFunction;
#[allow(unused_imports)]
use log::*;
use loss::LossFunction;

use data_generator::*;
use shady_neural_net::*;

use std::fs::File;

fn create_neural_net() -> Result<NeuralNet, Box<dyn std::error::Error>> {
    let mut neural_net = NeuralNet::new().expect("Could not initialize Neural Net");
    neural_net
        .add_input_layer(784)?
        .add_dense_layer(16, ActivationFunction::ReLU)?
        .add_dense_layer(16, ActivationFunction::ReLU)?
        .add_output_layer(10)?;

    neural_net.set_learning_rate(0.0001);
    neural_net
        .set_loss_function(LossFunction::LogLoss)
        .expect("Could Not Set the loss function of the network");

    Ok(neural_net)
}

fn print_progress(progress: usize, total: usize) {
    let bar_size = 20;
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

#[allow(dead_code)]
fn train() {
    let neural_net = create_neural_net().expect("Could not create neural net");
    let passes = 60000;

    // TEMP
    let image_file = File::open("test_files/train_images").expect("R");
    let mut image_buffer = vec![0; 784];

    let label_file = File::open("test_files/train_labels").expect("R");
    let mut label_buffer = vec![0];
    let mut v = vec![0.0; 10];

    for i in 0..passes {
        // let rand_x = rand::random_range(0.0..=1.0);
        // let rand_y = rand::random_range(0.0..=1.0);

        // let expected = generate_x_y_function(rand_x, rand_y);

        let now = Instant::now();
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
        let image_reading_time = now.elapsed();

        let now = Instant::now();
        let _vals = neural_net.feed_forward(&input).expect("C");
        let feed_forward_time = now.elapsed();

        let now = Instant::now();
        neural_net.set_loss(&v).expect("G");
        let set_loss_time = now.elapsed();
        let now = Instant::now();
        neural_net.back_propogate();
        let back_prop_time = now.elapsed();
        // let now = Instant::now();
        // let cost_time = now.elapsed();

        // if cost.is_nan() {
        //     println!("Failed");
        //     break;
        // }

        // _ = stdout().flush();

        // neural_net.set_learning_rate((cost * 0.01).min(0.1));

        // if i % 1000 == 0 {
        let cost = neural_net.get_cost().expect("G");
        print_progress(i, passes);
        print!(" Cost: {:>8.4} ", cost);
        print!(
            "IR Time: {:>5}us FF Time: {:>5}us SL Time: {:>5}us BP Time: {:>5}us",
            image_reading_time.as_micros(),
            feed_forward_time.as_micros(),
            set_loss_time.as_micros(),
            back_prop_time.as_micros(),
            // cost_time.as_micros(),
        );

        _ = stdout().flush();
        // }
    }

    // let cost = neural_net.get_cost().expect("G");
    _ = neural_net.save_model_to_file("test_files/nn_test.json");

    println!();
    println!("Done!");
}

#[allow(dead_code)]
fn test() {
    let mut neural_net = NeuralNet::load_model_from_file("test_files/nn_test.json").expect("F");
    neural_net.set_loss_function(LossFunction::MSE).expect("G");

    for _ in 0..100 {
        let rand_x = rand::random_range(-1.0..=1.0);
        let rand_y = rand::random_range(-1.0..=1.0);

        let expected = generate_x_y_function(rand_x, rand_y);

        _ = neural_net.feed_forward(&vec![rand_x, rand_y]);
        let vals = neural_net.get_output();

        let cost = {
            let diffs = vals
                .iter()
                .zip(expected.iter())
                .map(|(val, expected)| val - expected)
                .collect::<Vec<f32>>();
            diffs.iter().sum::<f32>() / diffs.len() as f32
        };

        println!(
            "Vals: {:#?}\nExpected: {:#?}\n Cost: {}",
            vals, expected, cost,
        );
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
