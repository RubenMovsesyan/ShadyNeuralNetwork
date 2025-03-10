pub enum RegularizationFunction {
    Lasso, // Least Absolute Shrinkage and Selection Operator
    Ridge, // Squared magnitude
    ElasticNetRegression,
}

pub struct Regularization {
    pub function: RegularizationFunction,
    pub hyper_parameter_1: f32,
    pub hyper_parameter_2: f32,
}
