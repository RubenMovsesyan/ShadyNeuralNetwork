use bytemuck::{Pod, Zeroable};

pub enum LossFunction {
    LogLoss, // Binary Cross Entropy Loss
    HingeLoss,
    MSE,   // Mean squared Error, quadratic loss, L2 loss
    MAE,   // Mean Absolute Error, L1 loss
    Huber, // Smooth mean absolute error
    LogCosh,
    Quantile,
}

impl LossFunction {
    pub fn as_repr(&self) -> LossRepr {
        use LossFunction::*;
        LossRepr {
            function: match self {
                LogLoss => 0,
                HingeLoss => 1,
                MSE => 2,
                MAE => 3,
                Huber => 4,
                LogCosh => 5,
                Quantile => 6,
            },
        }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
pub struct LossRepr {
    pub function: u32,
}
