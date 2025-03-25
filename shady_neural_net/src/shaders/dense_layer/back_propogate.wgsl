// Custom Structs
struct RegularizationFunction {
    function_type: u32,
    hyper_parameter_1: f32,
    hyper_parameter_2: f32,
}

// Constants
const LASSO: u32 = 0;
const RIDGE: u32 = 1;
const ELASTIC_NET_REGRESSION: u32 = 2;

// The buffer where the L1 Norm is stored
@group(0) @binding(0)
var<uniform> l_1_norm: f32;

// The buffer where the frobenius norm is stored
@group(0) @binding(1)
var<uniform> frobenius_norm: f32;

// Diemsnsions of the weights buffer
@group(0) @binding(2)
var<uniform> dims: vec2<u32>;

// The buffer of this layers weights
@group(0) @binding(3)
var<storage, read_write> weights: array<f32>;

// The buffer to store the gradient in
@group(0) @binding(4)
var<storage, read_write> gradient: array<f32>;

// The buffer where the gradient coefficient calculated in the coeff step is stored
@group(0) @binding(5)
var<storage, read> gradient_coefficient: array<f32>;

// The input buffer for this layer
@group(0) @binding(6)
var<storage, read> input_buffer: array<f32>;

// The buffer describing the regularization for this layer
@group(0) @binding(7)
var<uniform> regularization_info: RegularizationFunction;

// The buffer to store the regularization info into
@group(0) @binding(8)
var<storage, read_write> regularization_output: array<f32>;

@group(0) @binding(9)
var<storage, read_write> bias_gradient: array<f32>;

@group(0) @binding(10)
var<storage, read_write> biases: array<f32>;

// Uniform for the learning rate
@group(1) @binding(0)
var<uniform> learning_rate: f32;

// fn calculate_gradient(index: u32, row: u32, col: u32) -> f32 {
//     //      [ x y z w t ] <- this is the input buffer
//     // [ 1 ]
//     // [ a ]
//     // [ a ] <- this is the gradient coefficient
//     // [ a ]
//     let dJdo = gradient_coefficient[row];
//     let h = input_buffer[col];

//     let regularization = regularization_output[index];
//     return dJdo * h + regularization;
// }

fn calculate_gradient(row: u32, col: u32) -> f32 {
    return input_buffer[row] * gradient_coefficient[col];
}

@compute @workgroup_size(16, 16)
fn dense_layer_back_propogation_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    // Num outputs
    let num_outputs = dims.x;
    // Num Inputs (size of the input buffer)
    let num_inputs = dims.y;

    // The index of the weights buffer
    let index = row * num_inputs + col;

    if (row < num_outputs && col < num_inputs) {
        gradient[index] = calculate_gradient(row, col);
    }

    workgroupBarrier();

    if (row < num_outputs && col < num_inputs) {
        weights[index] = weights[index] - (learning_rate * gradient[index]);
        bias_gradient[row] = biases[row] - (learning_rate * gradient_coefficient[row]);
    }

    workgroupBarrier();

    if (row < num_outputs) {
        biases[row] = bias_gradient[row];
    }

    workgroupBarrier();
}
