use layer_structs::activation::ActivationFunction;
use pollster::*;
use std::{error::Error, fmt::Display};

#[allow(unused_imports)]
use log::*;

use layer::{DenseLayer, InputLayer, NeuralNetLayer, OutputLayer};
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

pub use layer_structs::activation;

mod layer;
mod layer_structs;
mod utils;

// Error Structs
#[derive(Debug)]
pub struct InputLayerAlreadyAddedError;

impl Error for InputLayerAlreadyAddedError {}

impl Display for InputLayerAlreadyAddedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input layer has already been added")
    }
}

#[derive(Debug)]
pub struct NoInputLayerAddedError;

impl Error for NoInputLayerAddedError {}

impl Display for NoInputLayerAddedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No Input layers have been added yet")
    }
}

#[derive(Debug)]
pub struct NoHiddenLayersAddedError;

impl Error for NoHiddenLayersAddedError {}

impl Display for NoHiddenLayersAddedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No Dense layers have been added yet")
    }
}

#[derive(Debug)]
pub struct AdapterNotCreatedError;

impl Error for AdapterNotCreatedError {}

impl Display for AdapterNotCreatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Couldn't Create Adapter")
    }
}

#[derive(Debug)]
pub struct LayerMismatchError;

impl Error for LayerMismatchError {}

impl Display for LayerMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layers Mismatched")
    }
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

        Ok(Self {
            device,
            queue,
            input_layer: None,
            hidden_layers: Vec::new(),
            output_layer: None,
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

    pub fn add_dense_layer(
        &mut self,
        num_nodes: u64,
        activation_function: ActivationFunction,
    ) -> Result<&mut Self, Box<dyn Error>> {
        if let None = self.input_layer {
            return Err(Box::new(NoInputLayerAddedError));
        }

        // WARN Unwrapping because at this point the buffers should already exist
        let connecting_buffer = {
            match self.hidden_layers.last() {
                Some(hidden_layer) => hidden_layer.get_connecting_bind_group().unwrap(),
                None => self
                    .input_layer
                    .as_ref()
                    .unwrap()
                    .get_connecting_bind_group()
                    .unwrap(),
            }
        };

        self.hidden_layers
            .push(NeuralNetLayer::Dense(DenseLayer::new(
                &connecting_buffer,
                num_nodes,
                activation_function,
                &self.device,
            )));

        Ok(self)
    }

    pub fn add_output_layer(
        &mut self,
        num_outputs: u64,
    ) -> Result<&mut Self, NoHiddenLayersAddedError> {
        let connecting_buffer = match self.hidden_layers.last() {
            Some(hidden_layer) => hidden_layer.get_connecting_bind_group().unwrap(),
            None => return Err(NoHiddenLayersAddedError),
        };

        self.output_layer = Some(NeuralNetLayer::Output(OutputLayer::new(
            &connecting_buffer,
            num_outputs,
            &self.device,
        )));

        Ok(self)
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
                    dense_layer.feed_forward(&self.device, &self.queue)
                }
                _ => return Err(Box::new(NoHiddenLayersAddedError)),
            }
        }

        match self.output_layer.as_ref().unwrap() {
            NeuralNetLayer::Output(output_layer) => {
                output_layer.feed_forward(&self.device, &self.queue);
                Ok(output_layer.get_output(&self.device))
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
        match &self.output_layer {
            Some(layer) => match layer {
                NeuralNetLayer::Output(output_layer) => {
                    let frobenius_norm =
                        output_layer.generate_weights_frobenius_norm(&self.device, &self.queue);

                    println!("Frobenius Norm: {}", frobenius_norm);
                }
                _ => {}
            },
            None => {}
        }

        for layer in &self.hidden_layers {
            match layer {
                NeuralNetLayer::Dense(dense_layer) => {
                    let frobenius_norm =
                        dense_layer.generate_weights_frobenius_norm(&self.device, &self.queue);

                    println!("Frobenius Norm: {}", frobenius_norm);
                }
                _ => {}
            }
        }
    }
}
