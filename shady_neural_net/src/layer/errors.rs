use std::{error::Error, fmt::Display};

#[derive(Debug)]
pub struct InputLengthMismatchError;

impl Error for InputLengthMismatchError {}

impl Display for InputLengthMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input Length is not the same as the buffer")
    }
}

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
pub struct NoOutputLayerAddedError;

impl Error for NoOutputLayerAddedError {}

impl Display for NoOutputLayerAddedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No Output layer has been added yet")
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
