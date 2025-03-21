// This is a module for the errors that can occur in the math
// module

use std::{error::Error, fmt::Display};

#[derive(Debug)]
pub struct MatrixDotError(pub String);

impl Error for MatrixDotError {}

impl Display for MatrixDotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct MatrixAddError(pub String);

impl Error for MatrixAddError {}

impl Display for MatrixAddError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
