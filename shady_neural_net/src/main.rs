use shady_neural_net::{
    create_training_batches_from_csv,
    layers::{activation_function::ActivationFunction, loss_function::LossFunction},
    neural_network::{NeuralNetwork, save_network},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let inputs_size = args[6]
        .parse()
        .expect(format!("Failed to parse arg: {:#?}", args[5]).as_str());

    let batch_size = args[7]
        .parse()
        .expect(format!("Failed to parse arg: {:#?}", args[6]).as_str());

    let data_normalization_value = args[5]
        .parse()
        .expect(format!("Failed to parse arg: {:#?}", args[4]).as_str());

    let mut shady_neural_net = NeuralNetwork::new(
        inputs_size,
        batch_size,
        LossFunction::LogLoss,
        0.1,
        Some(1024 * 1024 * 1024),
    );

    let (inputs, outputs) = create_training_batches_from_csv(
        &args[1],                          // csv path
        &args[2],                          // labels header
        args[3].clone()..=args[4].clone(), // data headers
        data_normalization_value,
        inputs_size,
        batch_size,
        None,
    )
    .expect("Failed");

    for (input, output) in inputs.into_iter().zip(outputs.into_iter()) {
        shady_neural_net.add_input_batch(input);
        shady_neural_net.add_label_batch(output);
    }

    shady_neural_net.add_layer(10, ActivationFunction::ReLU);
    shady_neural_net.add_layer(10, ActivationFunction::Softmax);

    match shady_neural_net.gradient_descent(1000) {
        Ok(_) => {}
        Err(err) => println!("{:#?}", err),
    }

    _ = save_network(&shady_neural_net, "test_files/nn.json");
}
