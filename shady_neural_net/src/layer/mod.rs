use std::rc::Rc;

#[allow(unused_imports)]
use log::*;

use wgpu::{BindGroup, BindGroupLayout, Buffer};

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

// Helper functions
fn compute_workgroup_size(nodes: u32, work_group_size: u32) -> u32 {
    (nodes + work_group_size - 1) / work_group_size
}

// Error Structs

// Trait for each layer to get the connecting buffer
pub trait Layer {
    fn get_connecting_bind_group(&self) -> Rc<BindGroup>;

    fn get_connecting_bind_group_layout(&self) -> Rc<BindGroupLayout>;

    fn get_connecting_buffer(&self) -> Rc<Buffer>;
}

pub struct ConnectingBindGroup {
    pub bind_group_layout: Rc<BindGroupLayout>,
    pub bind_group: Rc<BindGroup>,
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
    pub fn get_connecting_bind_group(&self) -> Option<ConnectingBindGroup> {
        use NeuralNetLayer::*;
        match self {
            Input(input_layer) => Some(ConnectingBindGroup {
                bind_group_layout: input_layer.get_connecting_bind_group_layout(),
                bind_group: input_layer.get_connecting_bind_group(),
                buffer: input_layer.get_connecting_buffer(),
                num_inputs: input_layer.num_inputs,
            }),
            Dense(dense_layer) => Some(ConnectingBindGroup {
                bind_group_layout: dense_layer.get_connecting_bind_group_layout(),
                bind_group: dense_layer.get_connecting_bind_group(),
                buffer: dense_layer.get_connecting_buffer(),
                num_inputs: dense_layer.num_nodes,
            }),
            Output(_) => None,
        }
    }
}
