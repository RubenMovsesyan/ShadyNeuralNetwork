pub enum RegularizationFunction {
    Lasso, // Least Absolute Shrinkage and Selection Operator
    Ridge, // Squared magnitude
    ElasticNetRegression,
}

pub struct Regularization {
    pub function: RegularizationFunction,
    pub hyper_parameter: f32,
}
