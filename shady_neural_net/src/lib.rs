use layer_structs::activation::ActivationFunction;
use pollster::*;
use regularization::{Regularization, RegularizationFunction};
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fs::File,
    io::{Read, Write},
};

#[allow(unused_imports)]
use log::*;

use layer::{
    BackPropogationConnection, BackPropogationLayer, DenseLayer, DenseLayerDescriptor, InputLayer,
    InputLayerDescriptor, NeuralNetLayer, NeuralNetLayerDescriptor, OutputLayer,
    OutputLayerDescriptor, errors::*,
};
use wgpu::{
    Backends, Buffer, BufferDescriptor, BufferUsages, Device, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, Limits, PowerPreference, Queue, RequestAdapterOptions,
};

pub use layer_structs::*;

mod layer;
mod layer_structs;
mod utils;

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetDesciriptor {
    input_layer: NeuralNetLayerDescriptor,
    hidden_layers: Vec<NeuralNetLayerDescriptor>,
    output_layer: NeuralNetLayerDescriptor,
}

// Neural Network API
#[derive(Debug)]
pub struct NeuralNet {
    // WGPU nececities
    device: Device,
    queue: Queue,

    input_layer: Option<NeuralNetLayer>,
    hidden_layers: Vec<NeuralNetLayer>,
    output_layer: Option<NeuralNetLayer>,

    learning_rate: Buffer,
}

impl NeuralNet {
    pub fn new() -> Result<Self, Box<dyn Error>> {
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
            .ok_or(AdapterNotCreatedError)?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()?;

        let learning_rate = device.create_buffer(&BufferDescriptor {
            label: Some("Learning Rate Buffer"),
            mapped_at_creation: false,
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        Ok(Self {
            device,
            queue,
            input_layer: None,
            hidden_layers: Vec::new(),
            output_layer: None,
            learning_rate,
        })
    }

    pub fn add_input_layer(
        &mut self,
        num_inputs: u64,
    ) -> Result<&mut Self, InputLayerAlreadyAddedError> {
        if let Some(_) = self.input_layer {
            return Err(InputLayerAlreadyAddedError);
        }

        self.input_layer = Some(NeuralNetLayer::Input(InputLayer::new(
            num_inputs,
            &self.device,
        )));

        Ok(self)
    }

    fn load_input_layer(
        &mut self,
        input_layer_descriptor: &InputLayerDescriptor,
    ) -> Result<&mut Self, InputLayerAlreadyAddedError> {
        if let Some(_) = self.input_layer {
            return Err(InputLayerAlreadyAddedError);
        }

        self.input_layer = Some(NeuralNetLayer::Input(InputLayer::from_descriptor(
            input_layer_descriptor,
            &self.device,
        )));

        Ok(self)
    }

    pub fn add_dense_layer(
        &mut self,
        num_nodes: u64,
        activation_function: ActivationFunction,
    ) -> Result<&mut Self, Box<dyn Error>> {
        if let None = self.input_layer {
            return Err(Box::new(NoInputLayerAddedError));
        }

        let previous_layer = match self.hidden_layers.last_mut() {
            Some(hidden_layer) => hidden_layer,
            None => self.input_layer.as_mut().unwrap(),
        };

        let connecting_buffer = previous_layer.get_connecting_bind_group().unwrap();
        let mut new_layer = DenseLayer::new(
            &connecting_buffer,
            num_nodes,
            activation_function,
            &self.device,
        );

        new_layer.link_gradient_descent_pipeline(&self.device, &self.learning_rate);

        previous_layer.link_next_layer_weights(
            &self.device,
            BackPropogationConnection {
                gradient_coefficient_buffer: new_layer.get_gradient_coefficient_buffer(),
                weights_buffer: new_layer.get_weights_buffer(),
                dimensions_buffer: new_layer.get_dimensions_buffer(),
            },
        );

        self.hidden_layers.push(NeuralNetLayer::Dense(new_layer));

        Ok(self)
    }

    fn load_dense_layer(
        &mut self,
        dense_layer_descriptor: &DenseLayerDescriptor,
    ) -> Result<&mut Self, Box<dyn Error>> {
        if let None = self.input_layer {
            return Err(Box::new(NoInputLayerAddedError));
        }

        let previous_layer = match self.hidden_layers.last_mut() {
            Some(hidden_layer) => hidden_layer,
            None => self.input_layer.as_mut().unwrap(),
        };

        let connecting_buffer = previous_layer.get_connecting_bind_group().unwrap();
        let mut new_layer =
            DenseLayer::from_descriptor(dense_layer_descriptor, &connecting_buffer, &self.device);

        new_layer.link_gradient_descent_pipeline(&self.device, &self.learning_rate);

        previous_layer.link_next_layer_weights(
            &self.device,
            BackPropogationConnection {
                gradient_coefficient_buffer: new_layer.get_gradient_coefficient_buffer(),
                weights_buffer: new_layer.get_weights_buffer(),
                dimensions_buffer: new_layer.get_dimensions_buffer(),
            },
        );

        self.hidden_layers.push(NeuralNetLayer::Dense(new_layer));

        Ok(self)
    }

    pub fn add_output_layer(
        &mut self,
        num_outputs: u64,
    ) -> Result<&mut Self, NoHiddenLayersAddedError> {
        let previous_layer = match self.hidden_layers.last_mut() {
            Some(hidden_layer) => hidden_layer,
            None => self.input_layer.as_mut().unwrap(),
        };

        let connecting_buffer = previous_layer.get_connecting_bind_group().unwrap();
        let mut new_output_layer = OutputLayer::new(&connecting_buffer, num_outputs, &self.device);

        new_output_layer.link_gradient_descent_pipeline(&self.device, &self.learning_rate);

        previous_layer.link_next_layer_weights(
            &self.device,
            BackPropogationConnection {
                gradient_coefficient_buffer: new_output_layer.get_gradient_coefficient_buffer(),
                weights_buffer: new_output_layer.get_weights_buffer(),
                dimensions_buffer: new_output_layer.get_dimensions_buffer(),
            },
        );

        self.output_layer = Some(NeuralNetLayer::Output(new_output_layer));

        Ok(self)
    }

    fn load_output_layer(
        &mut self,
        output_layer_descriptor: &OutputLayerDescriptor,
    ) -> Result<&mut Self, NoHiddenLayersAddedError> {
        let previous_layer = match self.hidden_layers.last_mut() {
            Some(hidden_layer) => hidden_layer,
            None => self.input_layer.as_mut().unwrap(),
        };

        let connecting_buffer = previous_layer.get_connecting_bind_group().unwrap();
        let mut new_output_layer =
            OutputLayer::from_descriptor(output_layer_descriptor, &connecting_buffer, &self.device);

        new_output_layer.link_gradient_descent_pipeline(&self.device, &self.learning_rate);

        previous_layer.link_next_layer_weights(
            &self.device,
            BackPropogationConnection {
                gradient_coefficient_buffer: new_output_layer.get_gradient_coefficient_buffer(),
                weights_buffer: new_output_layer.get_weights_buffer(),
                dimensions_buffer: new_output_layer.get_dimensions_buffer(),
            },
        );

        self.output_layer = Some(NeuralNetLayer::Output(new_output_layer));

        Ok(self)
    }

    pub fn set_learning_rate(&self, learning_rate: f32) {
        self.queue.write_buffer(
            &self.learning_rate,
            0,
            bytemuck::cast_slice(&[learning_rate]),
        );
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
        // FIXME fix this
        match self.input_layer.as_ref().unwrap() {
            NeuralNetLayer::Input(input_layer) => input_layer
                .set_inputs(inputs, &self.device, &self.queue)
                .expect("Failed to set inputs"),
            _ => return Err(Box::new(NoInputLayerAddedError)),
        }

        for layer in self.hidden_layers.iter() {
            match layer {
                NeuralNetLayer::Dense(dense_layer) => {
                    dense_layer.feed_forward(&self.device, &self.queue);
                }
                _ => return Err(Box::new(NoHiddenLayersAddedError)),
            }
        }

        match self.output_layer.as_ref().unwrap() {
            NeuralNetLayer::Output(output_layer) => {
                Ok(output_layer.feed_forward(&self.device, &self.queue))
            }
            _ => return Err(Box::new(NoHiddenLayersAddedError)),
        }
    }

    pub fn get_cost(&self, mut expected_values: Vec<f32>) -> Result<f32, Box<dyn Error>> {
        // Normalize the input vector
        {
            let avg = expected_values.iter().sum::<f32>() / expected_values.len() as f32;
            expected_values = expected_values
                .iter_mut()
                .map(|value| *value / avg)
                .collect();
        }

        match self.output_layer.as_ref().unwrap() {
            NeuralNetLayer::Output(output_layer) => {
                Ok(output_layer.compute_loss(&expected_values, &self.device, &self.queue))
            }
            _ => return Err(Box::new(NoHiddenLayersAddedError)),
        }
    }

    pub fn back_propogate(&self) {
        if let Some(layer) = &self.output_layer {
            if let NeuralNetLayer::Output(output_layer) = layer {
                output_layer.back_propogate(
                    Regularization {
                        function: RegularizationFunction::ElasticNetRegression,
                        hyper_parameter_1: 1.0,
                        hyper_parameter_2: 1.0,
                    },
                    &self.device,
                    &self.queue,
                );
            }
        }

        for layer in self.hidden_layers.iter().rev() {
            if let NeuralNetLayer::Dense(dense_layer) = layer {
                dense_layer.back_propogate(
                    Regularization {
                        function: RegularizationFunction::ElasticNetRegression,
                        hyper_parameter_1: 1.0,
                        hyper_parameter_2: 1.0,
                    },
                    &self.device,
                    &self.queue,
                );
            }
        }
    }

    pub fn gradient_decent(&self) {
        for layer in self.hidden_layers.iter() {
            if let NeuralNetLayer::Dense(dense_layer) = layer {
                dense_layer.gradient_descent(&self.device, &self.queue);
            }
        }

        if let Some(layer) = self.output_layer.as_ref() {
            if let NeuralNetLayer::Output(output_layer) = layer {
                output_layer.gradient_descent(&self.device, &self.queue);
            }
        }
    }

    pub fn save_model(&self) -> Result<NeuralNetDesciriptor, Box<dyn Error>> {
        let input_layer_descriptor: InputLayerDescriptor = self
            .input_layer
            .as_ref()
            .ok_or(NoInputLayerAddedError)?
            .as_input()?
            .to_descriptor();

        // TODO: if adding more layer types this needs to be some sort of trait
        let mut hidden_layer_descriptors: Vec<DenseLayerDescriptor> = Vec::new();

        for layer in self.hidden_layers.iter() {
            let hidden_layer_descriptor: DenseLayerDescriptor =
                layer.as_dense()?.to_descriptor(&self.device, &self.queue);

            hidden_layer_descriptors.push(hidden_layer_descriptor);
        }

        let output_layer_descriptor: OutputLayerDescriptor = self
            .output_layer
            .as_ref()
            .ok_or(NoOutputLayerAddedError)?
            .as_output()?
            .to_descriptor(&self.device, &self.queue);

        Ok(NeuralNetDesciriptor {
            input_layer: NeuralNetLayerDescriptor::Input(input_layer_descriptor),
            hidden_layers: hidden_layer_descriptors
                .into_iter()
                .map(|dense_layer_descriptor| {
                    NeuralNetLayerDescriptor::Dense(dense_layer_descriptor)
                })
                .collect(),
            output_layer: NeuralNetLayerDescriptor::Output(output_layer_descriptor),
        })
    }

    pub fn save_model_to_file(&self, file_name: &str) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(file_name)?;

        let serialized_data = self.save_model()?;

        write!(file, "{}", serde_json::to_string(&serialized_data)?)?;

        Ok(())
    }

    pub fn load_model_from_file(file_name: &str) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(file_name)?;

        let mut serialized_data = String::new();
        file.read_to_string(&mut serialized_data)?;

        let network_descriptor: NeuralNetDesciriptor = serde_json::from_str(&serialized_data)?;

        let mut neural_network = Self::new()?;

        let input_descriptor = network_descriptor.input_layer.as_input()?;
        neural_network.load_input_layer(input_descriptor)?;

        for layer in network_descriptor.hidden_layers.iter() {
            let dense_descriptor = layer.as_dense()?;
            neural_network.load_dense_layer(dense_descriptor)?;
        }

        let output_descriptor = network_descriptor.output_layer.as_output()?;
        neural_network.load_output_layer(output_descriptor)?;

        Ok(neural_network)
    }
}
