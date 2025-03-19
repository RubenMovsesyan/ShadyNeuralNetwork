use std::io::{Write, stdin, stdout};

use activation::ActivationFunction;
#[allow(unused_imports)]
use log::*;
use loss::LossFunction;

use data_generator::*;
use shady_neural_net::*;

fn create_neural_net() -> Result<NeuralNet, Box<dyn std::error::Error>> {
    let mut neural_net = NeuralNet::new().expect("Could not initialize Neural Net");
    neural_net
        .add_input_layer(2)?
        .add_dense_layer(3, ActivationFunction::ReLU)?
        .add_output_layer(2)?;

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

fn train() {
    let mut neural_net = create_neural_net().expect("Could not create neural net");
    neural_net.set_learning_rate(0.01);
    neural_net.set_loss_function(LossFunction::MSE).expect("G");
    let passes = 500;

    for i in 0..passes {
        let rand_x = rand::random_range(0.0..=1.0);
        let rand_y = rand::random_range(0.0..=1.0);

        let expected = generate_x_y_function(rand_x, rand_y);

        let _vals = neural_net.feed_forward(vec![rand_x, rand_y]).expect("C");

        neural_net.set_loss(expected.to_vec()).expect("G");
        neural_net.back_propogate();
        let cost = neural_net.get_cost().expect("G");

        if cost.is_nan() {
            println!("Failed");
            break;
        }

        _ = stdout().flush();

        neural_net.set_learning_rate(cost * 0.01);

        print_progress(i, passes);
        print!(" Cost: {cost} ");
    }
    _ = neural_net.save_model_to_file("test_files/nn_test.json");

    println!();
    println!("Done!");
}

fn test() {
    let mut neural_net = NeuralNet::load_model_from_file("test_files/nn_test.json").expect("F");
    neural_net.set_loss_function(LossFunction::MSE).expect("G");

    for _ in 0..100 {
        let rand_x = rand::random_range(-1.0..=1.0);
        let rand_y = rand::random_range(-1.0..=1.0);

        let expected = generate_x_y_function(rand_x, rand_y);

        _ = neural_net.feed_forward(vec![rand_x, rand_y]);
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

    // train();
    test();
}
