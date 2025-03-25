// Custom Structs
struct ActivationFunctionDescriptor {
    function_type: u32,
    function_parameter: f32,
}

// Constants
const STEP: u32 = 0;
const THRESHOLD: u32 = 1;
const BINARY_SIGMOID: u32 = 2;
const BIPOLAR_SIGMOID: u32 = 3;
const RELU: u32 = 4;
const LEAKY_RELU: u32 = 5;
const HYPERBOLIC_TANGENT: u32 = 6;
 
// Inputs from the previous layer
@group(0) @binding(0)
var<storage, read> input_buffer: array<f32>;

// Dimensions of the weights matrix
@group(1) @binding(0)
var<uniform> dims: vec2<u32>;

// Weights of the current layer
@group(1) @binding(1)
var<storage, read> weights_buffer: array<f32>;

// Biases of the current layer
@group(1) @binding(2)
var<storage, read> bias_buffer: array<f32>;

// Activation function information
@group(1) @binding(3)
var<uniform> activation_function: ActivationFunctionDescriptor;

// Intermediary buffer to be used for back propogation
@group(1) @binding(4)
var<storage, read_write> intermediary_buffer: array<f32>;

// Output Buffer after the activation function is applied
@group(1) @binding(5)
var<storage, read_write> output_buffer: array<f32>;

// Functions for the activation functions
fn step(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

fn threshold(x: f32, theta: f32) -> f32 {
    if (x >= theta) {
        return 1.0;
    } else {
        return 0.0;
    }
}

fn binary_sigmoid(x: f32, k: f32) -> f32 {
    let bottom = 1.0 + exp(-k * x);

    if (bottom == 0.0) {
        return 0.0;
    }

    return 1.0 / bottom;
}

fn bipolar_sigmoid(x: f32, k: f32) -> f32 {
    let top = 1.0 - exp(-k * x);
    let bottom = 1.0 + exp(-k * x);

    if (bottom == 0.0) {
        return 0.0;
    }

    return top / bottom;
}

fn relu(x: f32) -> f32 {
    if (x >= 0.0) {
        return x;
    } else {
        return 0.0;
    }
}

fn leaky_relu(x: f32, a: f32) -> f32 {
    if (x >= 0.0) {
        return x;
    } else {
        return a * x;
    }
}

fn hyperbolic_tangent(x: f32) -> f32 {
    let top = exp(x) - exp(-x);
    let bottom = exp(x) + exp(-x);

    if (bottom == 0.0) {
        return 0.0;
    }

    return top / bottom;
}

@compute @workgroup_size(256)
fn dense_layer_feed_forward_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    // Num Nodes
    let num_outputs = dims.x;
    // Num Inputs
    let num_inputs = dims.y;

    if (row < num_outputs) {
        // Matrix multiplication of the weights matrix to the input buffer
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < num_inputs; k++) {
            //          [x] 
            //          [y] 
            //          [z] 
            //          [w] 
            // [1 2 3 4]
            // [a b c d]
            // [a b c d]
            let index = row * num_inputs + k;

            sum += weights_buffer[index] * input_buffer[k];
        }

        // Add the bias to the sum buffer
        sum += bias_buffer[row];

        // Store the weighted sum in the intermediary buffer to
        // be used for back propogation before applying the
        // actication function
        intermediary_buffer[row] = sum;

        // Run the sum through the activation function
        switch activation_function.function_type {
            case STEP: {
                sum = step(sum);
            }
            case THRESHOLD: {
                sum = threshold(sum, activation_function.function_parameter);
            }
            case BINARY_SIGMOID: {
                sum = binary_sigmoid(sum, activation_function.function_parameter);
            }
            case BIPOLAR_SIGMOID: {
                sum = bipolar_sigmoid(sum, activation_function.function_parameter);
            }
            case RELU: {
                sum = relu(sum);
            }
            case LEAKY_RELU: {
                sum = leaky_relu(sum, activation_function.function_parameter);
            }
            case HYPERBOLIC_TANGENT: {
                sum = hyperbolic_tangent(sum);
            }
            default: {
                sum = step(sum);
            }
        }

        // Store the output of the activation function in
        // the output buffer
        output_buffer[row] = sum;
    }

    workgroupBarrier();
}
