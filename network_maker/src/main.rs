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

fn main() {
    pretty_env_logger::init();

    let neural_net = create_neural_net().expect("Could not create neural net");
    neural_net.set_learning_rate(0.1);

    const THRESHOLD: f32 = 0.0005;

    for _ in 0..1000 {
        let rand_x = rand::random_range(-1.0..=1.0);
        let rand_y = rand::random_range(-1.0..=1.0);

        if f32::abs(rand_x) < THRESHOLD || f32::abs(rand_y) < THRESHOLD {
            continue;
        }

        let expected = generate_x_y_function(rand_x, rand_y);

        if f32::abs(expected[0]) < THRESHOLD || f32::abs(expected[1]) < THRESHOLD {
            continue;
        }

        let _vals = neural_net.feed_forward(vec![rand_x, rand_y]).expect("C");
        let cost = neural_net.get_cost(expected.to_vec()).expect("G");

        if cost.is_nan() {
            println!("Failed");
            break;
        }

        neural_net.back_propogate();
        neural_net.gradient_decent();

        print!("\rCost: {}", cost);
        _ = stdout().flush();
    }
    println!();
    println!("Done!");

    _ = neural_net.save_model_to_file("test_files/nn_test.json");

    // Test the trained network
    // for _ in 0..100 {
    //     let rand_x = rand::random_range(-1.0..=1.0);
    //     let rand_y = rand::random_range(-1.0..=1.0);

    //     let expected = generate_x_y_function(rand_x, rand_y);

    //     // println!(
    //     //     "Outputs: {:#?}",
    //     //     neural_net.feed_forward(vec![rand_x, rand_y])
    //     // );
    //     // println!("Cost: {:#?}", neural_net.get_cost(expected.to_vec()),);
    // }

    // println!(
    //     "Outputs: {:#?}",
    //     neural_net.feed_forward(vec![-0.1, -0.2, -0.3, 1.0, 2.0, 3.0])
    // );
    // _ = neural_net.save_model_to_file("test_files/nn_test.json");
    // drop(neural_net);

    // let neural_net = NeuralNet::load_model_from_file("test_files/nn_test.json")
    //     .expect("Could not load model from file");

    // println!(
    //     "Outputs: {:#?}",
    //     neural_net.feed_forward(vec![-0.1, -0.2, -0.3, 1.0, 2.0, 3.0])
    // );

    // // println!(
    // //     "Cost: {}",
    // //     neural_net
    // //         .get_cost(vec![0.2, 0.2, 6.0])
    // //         .expect("Could Not Get Cost")
    // // );

    // _ = neural_net.get_cost(vec![0.6, 0.2, 0.2]);
    // neural_net.back_propogate();

    // neural_net.set_learning_rate(0.1);

    // neural_net.gradient_decent();
}
