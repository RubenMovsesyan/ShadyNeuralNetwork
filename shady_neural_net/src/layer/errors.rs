use std::{error::Error, fmt::Display};

#[derive(Debug)]
pub struct InputLengthMismatchError;

impl Error for InputLengthMismatchError {}

impl Display for InputLengthMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input Length is not the same as the buffer")
    }
}
