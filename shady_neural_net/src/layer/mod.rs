use std::rc::Rc;

#[allow(unused_imports)]
use log::*;

use wgpu::{Buffer, Device};

// Module for inner structures in the Neural Network
use crate::layer_structs::*;

pub use dense_layer::DenseLayer;
pub use input_layer::InputLayer;
pub use output_layer::OutputLayer;

mod bind_group_macro;
mod dense_layer;
mod input_layer;
mod output_layer;

mod errors;

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
pub trait FeedForwardLayer {
    fn get_output_buffer(&self) -> Rc<Buffer>;
}

// Trait for layers that have a back propogation connection to the previous layer
pub trait BackPropogationLayer {
    fn get_gradient_coefficient_buffer(&self) -> Rc<Buffer>;

    fn get_weights_buffer(&self) -> Rc<Buffer>;

    fn get_dimensions_buffer(&self) -> Rc<Buffer>;
}

pub struct FeedForwardConnection {
    pub buffer: Rc<Buffer>,
    pub num_inputs: u64,
}

pub struct BackPropogationConnection {
    pub gradient_coefficient_buffer: Rc<Buffer>,
    pub weights_buffer: Rc<Buffer>,
    pub dimensions_buffer: Rc<Buffer>,
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
}
