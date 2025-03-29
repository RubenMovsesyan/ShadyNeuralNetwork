use std::cell::RefCell;
use std::rc::Rc;

use gpu_math::math::matrix::Matrix;

pub mod activation_function;
pub mod input;
pub mod layer;
pub mod loss_function;
pub mod output;

type MatrixRef = Rc<RefCell<Matrix>>;
