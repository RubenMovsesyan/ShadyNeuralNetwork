use std::io::{Write, stdout};

use activation::{ActivationFunction, BinarySigmoidFunction};
#[allow(unused_imports)]
use log::*;

use data_generator::*;
use shady_neural_net::*;

fn create_neural_net() -> Result<NeuralNet, Box<dyn std::error::Error>> {
    let mut neural_net = NeuralNet::new().expect("Could not initialize Neural Net");
    neural_net
        .add_input_layer(2)?
        .add_dense_layer(
            4,
            ActivationFunction::BinarySigmoid(BinarySigmoidFunction { k: 1.0 }),
            // ActivationFunction::Step,
        )?
        .add_dense_layer(
            4,
            ActivationFunction::BinarySigmoid(BinarySigmoidFunction { k: 1.0 }),
            // ActivationFunction::Step,
        )?
        .add_output_layer(2)?;

    Ok(neural_net)
}

fn train() {
    let neural_net = create_neural_net().expect("Could not create neural net");
    neural_net.set_learning_rate(0.1);

    for _ in 0..1000 {
        let rand_x = rand::random_range(-1.0..=1.0);
        let rand_y = rand::random_range(-1.0..=1.0);

        let expected = generate_x_y_function(rand_x, rand_y);

        let _vals = neural_net.feed_forward(vec![rand_x, rand_y]).expect("C");
        // let cost = neural_net.get_cost(expected.to_vec()).expect("G");

        // if cost.is_nan() {
        //     println!("Failed");
        //     break;
        // }

        neural_net.set_loss(expected.to_vec()).expect("G");
        neural_net.back_propogate();
        // neural_net.gradient_decent();

        print!("\rCost: {}", neural_net.get_cost().expect("F"));
        _ = stdout().flush();
    }
    println!();
    println!("Done!");

    _ = neural_net.save_model_to_file("test_files/nn_test.json");
}

fn test() {
    let neural_net = NeuralNet::load_model_from_file("test_files/nn_test.json").expect("F");

    for _ in 0..100 {
        let rand_x = rand::random_range(-1.0..=1.0);
        let rand_y = rand::random_range(-1.0..=1.0);

        let expected = generate_x_y_function(rand_x, rand_y);

        _ = neural_net.feed_forward(vec![rand_x, rand_y]);

        // let cost = vals
        //     .iter()
        //     .zip(expected.iter())
        //     .map(|(pred, expe)| pred - expe)
        //     .collect::<Vec<f32>>();

        // println!(
        //     "Vals: {:#?}\nExpected: {:#?}\n Cost: {}",
        //     vals,
        //     expected,
        //     cost.iter().sum::<f32>() / cost.len() as f32
        // );
    }
}

fn main() {
    pretty_env_logger::init();

    train();
    // test();
}
