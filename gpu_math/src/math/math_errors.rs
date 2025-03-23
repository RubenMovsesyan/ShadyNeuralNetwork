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

#[derive(Debug)]
pub struct MatrixSubError(pub String);

impl Error for MatrixSubError {}

impl Display for MatrixSubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct MatrixMultError(pub String);

impl Error for MatrixMultError {}

impl Display for MatrixMultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct MatrixExpError(pub String);

impl Error for MatrixExpError {}

impl Display for MatrixExpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct MatrixSumError(pub String);

impl Error for MatrixSumError {}

impl Display for MatrixSumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct MatrixCustomError(pub String);

impl Error for MatrixCustomError {}

impl Display for MatrixCustomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct MatrixVariantError(pub String);

impl Error for MatrixVariantError {}

impl Display for MatrixVariantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
