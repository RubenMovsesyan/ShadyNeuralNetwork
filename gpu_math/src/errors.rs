use std::{error::Error, fmt::Display};

#[derive(Debug)]
pub struct GpuMathNotInitializedError(pub String);

impl Display for GpuMathNotInitializedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gpu Math Library Has not Been Initialized, Error Occured in: {}",
            self.0
        )
    }
}

impl Error for GpuMathNotInitializedError {}
