// Dimensions of the weigths matrix for this layer
@group(0) @binding(0)
var<uniform> dims: vec2<u32>;

// The output buffer where the feed forward computation is stored
@group(0) @binding(1)
var<storage, read> output: array<f32>;

// The buffer where the expected values are stored
@group(0) @binding(2)
var<storage, read> expected_values_buffer: array<f32>;

// The buffer where the loss function will be stored
@group(0) @binding(3)
var<storage, read_write> loss_function_buffer: array<f32>;

// This is the gradient coefficient to use for back propogation in this layer
@group(0) @binding(4)
var<storage, read_write> gradient_coefficient: array<f32>;

// This is the gradient coefficient multiplied by the weights to be sent back to the next layer
@group(0) @binding(5)
var<storage, read_write> gradient_back_prop: array<f32>;

// This is the current layers weights to compute the back propogation gradient coefficent
@group(0) @binding(6)
var<storage, read> weights: array<f32>;

// This is the buffer that stores the information for which loss function to use
@group(0) @binding(7)
var<uniform> loss_function_info: u32;

// Constants
const LOG_LOSS: u32 = 0; // Binary Cross Entropy Loss
const HINGE_LOSS: u32 = 1;
const MSE: u32 = 2; // Mean Squared Error, Quadratic Loss, L2 Loss
const MAE: u32 = 3; // Mean Absolute Error,
const HUBER: u32 = 4; // Smooth mean absolute error
const LOG_COSH: u32 = 5;
const QUANTILE: u32 = 6;

// Helper functions to compute the loss and the gradient

// Log Loss / Binary Cross Entropy Loss
// -((o_n * ln(y_n) + ((1 - y_n) * ln(1 - o_n)))
fn binary_cross_entropy_loss_gradient(predicted: f32, expected: f32) -> f32 {
    let top = predicted - expected;
    let bottom = (predicted - 1.0) * predicted;

    if (bottom == 0.0) {
        return 0.0;
    }
    
    return -1.0 * (top / bottom);
}


// -(o - y) / ((o - 1)o)
fn binary_cross_entropy_loss(predicted: f32, expected: f32) -> f32 {
    let part_1 = expected * log(predicted);
    let part_2 = (1.0 - expected) * log(1.0 - predicted);

    return -1.0 * (part_1 + part_2);
}

// Hinge Loss
fn hinge_loss_gradient(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

fn hinge_loss(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

// Mean Squared Error Loss
fn mean_squared_error_loss_gradient(predicted: f32, expected: f32) -> f32 {
    return predicted - expected;
}

fn mean_squared_error_loss(predicted: f32, expected: f32) -> f32 {
    return 0.5 * pow(expected - predicted, 2.0);
}

// Mean Absolute Error Loss
fn mean_absolute_error_loss_gradient(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

fn mean_absolute_error_loss(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

// Huber Loss
fn huber_loss_gradient(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

fn huber_loss(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

// Log Cosh Loss
fn log_cosh_loss_gradient(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

fn log_cosh_loss(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

// Quantile Loss
fn quantile_loss_gradient(predicted: f32, expected: f32) -> f32 {
    return 0.0;
}

fn quantile_loss(predicted: f32, expexted: f32) -> f32 {
    return 0.0;
}

@compute @workgroup_size(256)
fn output_layer_loss_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Binary cross entropy loss
    let row = global_id.x;
    // Num outputs
    let num_outputs = dims.x;
    // Num Inputs
    let num_inputs = dims.y;

    if (row < num_outputs) {
        let predicted = output[row];
        let expected = expected_values_buffer[row];
        // This is the derivative of the loss function
        // This is the first level of the gradient coefficient
        // dJ/do_N

        switch loss_function_info {
            case LOG_LOSS: {
                loss_function_buffer[row] = binary_cross_entropy_loss(predicted, expected);
                gradient_coefficient[row] = binary_cross_entropy_loss_gradient(predicted, expected);
            }
            case HINGE_LOSS: {
                loss_function_buffer[row] = hinge_loss(predicted, expected);
                gradient_coefficient[row] = hinge_loss_gradient(predicted, expected);
            }
            case MSE: {
                loss_function_buffer[row] = mean_squared_error_loss(predicted, expected);
                gradient_coefficient[row] = mean_squared_error_loss_gradient(predicted, expected);
            }
            case MAE: {
                loss_function_buffer[row] = mean_absolute_error_loss(predicted, expected);
                gradient_coefficient[row] = mean_absolute_error_loss_gradient(predicted, expected);
            }
            case HUBER: {
                loss_function_buffer[row] = huber_loss(predicted, expected);
                gradient_coefficient[row] = huber_loss_gradient(predicted, expected);
            }
            case LOG_COSH: {
                loss_function_buffer[row] = log_cosh_loss(predicted, expected);
                gradient_coefficient[row] = log_cosh_loss_gradient(predicted, expected);
            }
            case QUANTILE: {
                loss_function_buffer[row] = quantile_loss(predicted, expected);
                gradient_coefficient[row] = quantile_loss_gradient(predicted, expected);
            }
            default: {}
        }
        
    }

    workgroupBarrier();

    // HACK This is a kinda sketchy way to do this
    if (row < num_inputs) {
        //          [ 1 2 3 4 ]
        //          [ a b c d ] <- this is the weights matrix
        //          [ a b c d ]
        // [ x y z ] <- this is the current coefficient
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < num_outputs; k++) {
            let index = row * num_outputs + k;
            sum += weights[index] * gradient_coefficient[k];
        }

        // This is the first step of sending the gradient coefficient
        // to the previous layer. Since It is easier to multiply the
        // coefficient with the weight matrix here we do it here
        // this is dJ/do_N * W^(N)
        gradient_back_prop[row] = sum;
    } 

    workgroupBarrier();
}
