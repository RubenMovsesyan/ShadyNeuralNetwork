use shady_neural_net::{
    create_training_batches_from_csv,
    layers::{activation_function::ActivationFunction, loss_function::LossFunction},
    neural_network::NeuralNetwork,
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

    match shady_neural_net.gradient_descent(500) {
        Ok(_) => {}
        Err(err) => println!("{:#?}", err),
    }

    // for (input_batch, label_batch) in training_inputs.into_iter().zip(label_inputs.into_iter()) {
    //     shady_neural_net.add_input_batch(input_batch);
    //     shady_neural_net.add_label_batch(label_batch);
    // }

    // shady_neural_net.add_layer(10, ActivationFunction::ReLU);
    // shady_neural_net.add_layer(10, ActivationFunction::Softmax);

    // for iteration in 0..500 {
    //     for batch_number in 0..shady_neural_net.num_batches() {
    //         shady_neural_net.feed_forward(batch_number).expect("Failed");
    //         shady_neural_net
    //             .back_propogate(batch_number)
    //             .expect("Failed");
    //         shady_neural_net.update_parameters().expect("Failed");

    //         if iteration % 10 == 0 {
    //             println!("Iteration: {}", iteration);
    //             println!(
    //                 "Cost: {} {}",
    //                 batch_number,
    //                 shady_neural_net.get_cost(batch_number).expect("Failed")
    //             );
    //         }
    //     }
    // }

    // for output in outputs.iter() {
    //     println!("{}", output);
    // }

    // println!(
    //     "Costs: {:#?}",
    //     shady_neural_net.get_costs(&outputs).expect("Failed")
    // );
}
