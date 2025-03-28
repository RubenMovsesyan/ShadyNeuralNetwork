use std::rc::Rc;
use std::{cell::RefCell, error::Error};

use activation_function::ActivationFunction;
use gpu_math::math::matrix::Matrix;
use input::Input;
use layer::Layer;
use loss_function::LossFunction;
use output::Output;
use pollster::FutureExt;
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

pub mod activation_function;
pub mod input;
pub mod layer;
pub mod loss_function;
pub mod output;

type MatrixRef = Rc<RefCell<Matrix>>;

#[derive(Debug)]
pub struct NeuralNetwork {
    inputs: Option<Input>,
    layers: Vec<Layer>,
    expected: Option<Output>,

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
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        Self {
            inputs: None,
            layers: Vec::new(),
            expected: None,
            num_inputs,
            batch_size,
            loss_function,
            learning_rate,
            device,
            queue,
        }
    }

    pub fn add_layer(&mut self, num_nodes: usize, activation_function: ActivationFunction) {
        let link = match self.layers.last() {
            Some(prev_layer) => prev_layer.output_link(),
            None => self
                .inputs
                .as_ref()
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

    pub fn set_input_batch(&mut self, input_batch: Matrix) {
        self.inputs = Some(Input::new(
            input_batch,
            self.device.clone(),
            self.queue.clone(),
        ));
    }

    pub fn set_label_batch(&mut self, label_batch: Matrix) {
        self.expected = Some(Output::new(
            label_batch,
            self.loss_function,
            self.device.clone(),
            self.queue.clone(),
        ));
    }

    pub fn feed_forward(&mut self) -> Result<(), Box<dyn Error>> {
        for layer in self.layers.iter_mut() {
            layer.feed_forward()?;
        }

        Ok(())
    }

    pub fn get_cost(&self) -> Result<f32, Box<dyn Error>> {
        Ok(self
            .expected
            .as_ref()
            .expect("Outputs not set yet")
            .get_cost(&self.layers.last().unwrap().output_link().borrow())?)
    }

    pub fn back_propogate(&mut self) -> Result<(), Box<dyn Error>> {
        let mut next_layer = self
            .expected
            .as_mut()
            .expect("Outputs not set yet")
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
}
