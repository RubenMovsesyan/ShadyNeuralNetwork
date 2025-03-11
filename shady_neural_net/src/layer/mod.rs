use std::rc::Rc;

#[allow(unused_imports)]
use log::*;

use wgpu::Buffer;

// Module for inner structures in the Neural Network
use crate::layer_structs::*;

pub use dense_layer::DenseLayer;
pub use input_layer::InputLayer;
pub use output_layer::OutputLayer;

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
    fn get_connecting_weight_buffer(&self) -> Rc<Buffer>;
}

pub struct FeedForwardConnection {
    pub buffer: Rc<Buffer>,
    pub num_inputs: u64,
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
            Output(_) => None,
        }
    }
}
