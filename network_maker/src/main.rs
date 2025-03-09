use activation::ActivationFunction;
#[allow(unused_imports)]
use log::*;

use shady_neural_net::*;

fn create_neural_net() -> Result<NeuralNet, Box<dyn std::error::Error>> {
    let mut neural_net = NeuralNet::new().expect("Could not initialize Neural Net");
    neural_net
        .add_input_layer(6)?
        .add_dense_layer(
            16,
            // ActivationFunction::BinarySigmoid(BinarySigmoidFunction { k: 1.0 }),
            ActivationFunction::Step,
        )?
        .add_dense_layer(
            16,
            // ActivationFunction::BinarySigmoid(BinarySigmoidFunction { k: 1.0 }),
            ActivationFunction::Step,
        )?
        .add_output_layer(3)?;

    Ok(neural_net)
}

fn main() {
    pretty_env_logger::init();

    let neural_net = create_neural_net().expect("Could not create neural net");
    println!(
        "Outputs: {:#?}",
        neural_net.feed_forward(vec![0.1, 0.2, 0.3, -1.0, -2.0, -3.0])
    );

    println!(
        "Cost: {}",
        neural_net
            .get_cost(vec![0.2, 0.2, 6.0])
            .expect("Could Not Get Cost")
    );
}
