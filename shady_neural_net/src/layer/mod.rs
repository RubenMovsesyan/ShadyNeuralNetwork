use std::rc::Rc;

#[allow(unused_imports)]
use log::*;

use regularization::Regularization;
use serde::{Deserialize, Serialize};
use wgpu::{Buffer, Device, Queue};

// Module for inner structures in the Neural Network
use crate::{LayerMismatchError, layer_structs::*};

pub use dense_layer::DenseLayer;
pub use input_layer::InputLayer;
pub use output_layer::OutputLayer;

pub use dense_layer::DenseLayerDescriptor;
pub use input_layer::InputLayerDescriptor;
pub use output_layer::OutputLayerDescriptor;

mod bind_group_macro;
mod dense_layer;
mod input_layer;
mod output_layer;
mod weight_distribution;

pub mod errors;

const WORK_GROUP_SIZE: u32 = 256;
const D2_WORK_GROUP_SIZE: u32 = 16;

// Helper functions
fn compute_workgroup_size(nodes: u32, work_group_size: u32) -> u32 {
    (nodes + work_group_size - 1) / work_group_size
}

fn compute_2d_workgroup_size(
    (width, height): (u32, u32),
    (work_group_width, work_group_height): (u32, u32),
) -> (u32, u32) {
    let x = (width + work_group_width - 1) / work_group_width;
    let y = (height + work_group_height - 1) / work_group_height;

    (x, y)
}

// Trait for layers that have a feed forawrd connection to the next layer
pub trait FeedForwardLayerConnection {
    fn get_output_buffer(&self) -> Rc<Buffer>;
}

// Trait for layers that have a back propogation connection to the previous layer
pub trait BackPropogationLayerConnection {
    fn get_ceoff_back_prop_buffer(&self) -> Rc<Buffer>;
}

// Trait for feeding data forward
pub trait FeedForwardLayer {
    fn feed_forward(&self, device: &Device, queue: &Queue);
}

// Trait for performing back propogation and gradiend descent
pub trait BackPropogationLayer {
    fn back_propogate(&self, regularization: Regularization, device: &Device, queue: &Queue);
}

pub struct FeedForwardConnection {
    pub buffer: Rc<Buffer>,
    pub num_inputs: u64,
}

pub struct BackPropogationConnection {
    pub grad_coeff_back_prop_buffer: Rc<Buffer>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum NeuralNetLayerDescriptor {
    Input(InputLayerDescriptor),
    Dense(DenseLayerDescriptor),
    Output(OutputLayerDescriptor),
}

impl NeuralNetLayerDescriptor {
    pub fn as_input(&self) -> Result<&InputLayerDescriptor, LayerMismatchError> {
        match self {
            NeuralNetLayerDescriptor::Input(input_layer_descriptor) => Ok(&input_layer_descriptor),
            _ => Err(LayerMismatchError),
        }
    }

    pub fn as_dense(&self) -> Result<&DenseLayerDescriptor, LayerMismatchError> {
        match self {
            NeuralNetLayerDescriptor::Dense(dense_layer_descriptor) => Ok(&dense_layer_descriptor),
            _ => Err(LayerMismatchError),
        }
    }

    pub fn as_output(&self) -> Result<&OutputLayerDescriptor, LayerMismatchError> {
        match self {
            NeuralNetLayerDescriptor::Output(output_layer_descriptor) => {
                Ok(&output_layer_descriptor)
            }
            _ => Err(LayerMismatchError),
        }
    }
}

#[derive(Debug)]
pub enum NeuralNetLayer {
    Input(InputLayer),
    Dense(DenseLayer),
    Output(OutputLayer),
}

impl NeuralNetLayer {
    pub fn get_connecting_bind_group(&self) -> Option<FeedForwardConnection> {
        use NeuralNetLayer::*;
        match self {
            Input(input_layer) => Some(FeedForwardConnection {
                buffer: input_layer.get_output_buffer(),
                num_inputs: input_layer.num_inputs,
            }),
            Dense(dense_layer) => Some(FeedForwardConnection {
                buffer: dense_layer.get_output_buffer(),
                num_inputs: dense_layer.num_nodes,
            }),
            _ => None,
        }
    }

    pub fn link_next_layer_weights(
        &mut self,
        device: &Device,
        back_propogation_connection: BackPropogationConnection,
    ) {
        use NeuralNetLayer::*;
        match self {
            Dense(dense_layer) => {
                dense_layer.link_next_layer(device, &back_propogation_connection);
            }
            _ => {}
        }
    }

    pub fn as_input(&self) -> Result<&InputLayer, LayerMismatchError> {
        match self {
            NeuralNetLayer::Input(input_layer) => Ok(&input_layer),
            _ => Err(LayerMismatchError),
        }
    }

    pub fn as_dense(&self) -> Result<&DenseLayer, LayerMismatchError> {
        match self {
            NeuralNetLayer::Dense(dense_layer) => Ok(&dense_layer),
            _ => Err(LayerMismatchError),
        }
    }

    pub fn as_output(&self) -> Result<&OutputLayer, LayerMismatchError> {
        match self {
            NeuralNetLayer::Output(output_layer) => Ok(&output_layer),
            _ => Err(LayerMismatchError),
        }
    }
}
