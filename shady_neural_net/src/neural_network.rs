use std::{
    error::Error,
    fs::File,
    io::{Write, stdout},
    rc::Rc,
};

use gpu_math::math::matrix::Matrix;
use pollster::FutureExt;
use serde::{Deserialize, Serialize};
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

use crate::layers::{
    activation_function::ActivationFunction, input::Input, layer::Layer,
    loss_function::LossFunction, output::Output,
};

//                  weights    biases   activation function    inputs   outputs
//                      \/        \/        \/                  \/      \/
pub type Parameters = (Vec<f32>, Vec<f32>, ActivationFunction, usize, usize);

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkDescriptor {
    layers: Vec<Parameters>,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    inputs: Vec<Input>,
    layers: Vec<Layer>,
    expected: Vec<Output>,

    num_inputs: usize,
    batch_size: usize,

    loss_function: LossFunction,

    learning_rate: f32,

    // WGPU
    device: Rc<Device>,
    queue: Rc<Queue>,
}

impl NeuralNetwork {
    pub fn new(
        num_inputs: usize,
        batch_size: usize,
        loss_function: LossFunction,
        learning_rate: f32,
        max_buffer_size: Option<u32>,
    ) -> Self {
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
                    required_limits: match max_buffer_size {
                        Some(max_size) => Limits {
                            max_storage_buffer_binding_size: max_size,
                            ..Default::default()
                        },
                        None => Limits::default(),
                    },
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        Self {
            inputs: Vec::new(),
            layers: Vec::new(),
            expected: Vec::new(),
            num_inputs,
            batch_size,
            loss_function,
            learning_rate,
            device,
            queue,
        }
    }

    pub fn set_from_network_descriptor(&mut self, descriptor: NetworkDescriptor) {
        let mut link = match self.inputs.first() {
            Some(input_layer) => input_layer.get_inputs(),
            None => panic!("Input Layer has not been added yet"),
        };

        self.layers.clear();

        for layer_descriptor in descriptor.layers.iter() {
            self.layers.push(Layer::from_parameters(
                layer_descriptor,
                self.batch_size,
                link,
                self.device.clone(),
                self.queue.clone(),
            ));

            unsafe {
                link = self.layers.last().unwrap_unchecked().output_link();
            }
        }
    }

    pub fn add_layer(&mut self, num_nodes: usize, activation_function: ActivationFunction) {
        let link = match self.layers.last() {
            Some(prev_layer) => prev_layer.output_link(),
            None => self
                .inputs
                .first()
                .expect("Input Layer has not been added yet")
                .get_inputs(), // FIXME: This should check if inputs are here
        };

        self.layers.push(Layer::new(
            if let Some(prev_layer) = self.layers.last() {
                prev_layer.get_num_nodes()
            } else {
                self.num_inputs
            },
            num_nodes,
            link,
            self.batch_size,
            activation_function,
            self.device.clone(),
            self.queue.clone(),
        ));
    }

    pub fn add_input_batch(&mut self, input_batch: Matrix) {
        self.inputs.push(Input::new(
            input_batch,
            self.device.clone(),
            self.queue.clone(),
        ));
    }

    pub fn add_label_batch(&mut self, label_batch: Matrix) {
        self.expected.push(Output::new(
            label_batch,
            self.loss_function,
            self.device.clone(),
            self.queue.clone(),
        ));
    }

    pub fn feed_forward(&mut self, batch_number: usize) -> Result<(), Box<dyn Error>> {
        match self.layers.first_mut() {
            Some(first_layer) => first_layer.input_link(self.inputs[batch_number].get_inputs()),
            None => panic!(),
        }

        for layer in self.layers.iter_mut() {
            layer.feed_forward()?;
        }

        Ok(())
    }

    pub fn get_cost(&self, batch_number: usize) -> Result<f32, Box<dyn Error>> {
        Ok(self.expected[batch_number]
            .get_cost(&self.layers.last().unwrap().output_link().borrow())?)
    }

    pub fn back_propogate(&mut self, batch_number: usize) -> Result<(), Box<dyn Error>> {
        let mut next_layer = self.expected[batch_number]
            .get_loss_gradient(&self.layers.last().unwrap().output_link().borrow())?;
        for layer in self.layers.iter_mut().rev() {
            next_layer = layer.back_propogate(next_layer)?;
        }

        Ok(())
    }

    pub fn update_parameters(&mut self) -> Result<(), Box<dyn Error>> {
        for layer in self.layers.iter_mut() {
            layer.update_parameters(self.learning_rate)?;
        }

        Ok(())
    }

    pub fn num_batches(&self) -> usize {
        self.inputs.len()
    }

    pub fn gradient_descent(&mut self, iterations: usize) -> Result<(), Box<dyn Error>> {
        for iteration in 0..iterations {
            if iteration % 10 == 0 {
                print_progress(iteration, iterations);
            }

            for batch_number in 0..self.num_batches() {
                self.feed_forward(batch_number)?;
                self.back_propogate(batch_number)?;
                self.update_parameters()?;
            }
        }

        Ok(())
    }

    pub fn get_model_descriptor(&self) -> NetworkDescriptor {
        let layer_descriptors = self
            .layers
            .iter()
            .map(|layer| layer.save_parameters().expect("Failed to Save parameters"))
            .collect::<Vec<Parameters>>();

        NetworkDescriptor {
            layers: layer_descriptors,
        }
    }
}

fn print_progress(progress: usize, total: usize) {
    let bar_size = 40;
    let ratio = progress as f32 / total as f32;
    let bars = (bar_size as f32 * ratio).ceil() as usize;

    print!("\r[");
    for _ in 0..bars {
        print!("=");
    }
    for _ in bars..bar_size {
        print!(" ");
    }
    print!("] ");

    print!("Progress: {:>6.2}%", ratio * 100.0);
    _ = stdout().flush();
}

pub fn save_network(neural_network: &NeuralNetwork, file_path: &str) -> std::io::Result<()> {
    let serialized = neural_network.get_model_descriptor();

    let mut file = File::create(file_path)?;

    write!(file, "{}", serde_json::to_string(&serialized)?)?;
    Ok(())
}
