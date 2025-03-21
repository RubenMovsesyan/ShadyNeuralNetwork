// Custom structs
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


// Input buffer for this layer
@group(0) @binding(0)
var<storage, read> inputs: array<f32>;

@group(1) @binding(0)
var<storage, read> intermediary: array<f32>;

// Output buffer for this layer
@group(1) @binding(1)
var<storage, read> outputs: array<f32>;

// Buffer containing the dimensions of the weights buffer for this layer
@group(1) @binding(2)
var<uniform> dims: vec2<u32>;

// Buffer containing all the weights
@group(1) @binding(3)
var<storage, read> weights: array<f32>;

// Uniform containing the information for the activation function of this layer
@group(1) @binding(4)
var<uniform> activation_function: ActivationFunctionDescriptor;

// Buffer to store the activation function derivative once it is computed
@group(1) @binding(5)
var<storage, read_write> gradient_coefficient_intermediary: array<f32>;

// Buffer containing the gradient information from the next layer
@group(1) @binding(6)
var<storage, read> next_layers_gradient_coefficient: array<f32>;

// This is the gradient coefficient to use for back propogation in this layer
@group(1) @binding(7)
var<storage, read_write> gradient_coefficient: array<f32>;

// This is the gradient coefficient multiplied by the weights 
@group(1) @binding(8)
var<storage, read_write> gradient_back_prop: array<f32>;

// Here are the gradients of the activation functions

// Even though the gradient is undefined at 0, we always return 0
// to avoid undefined behavior
fn step_gradient(x: f32) -> f32 {
    return 0.0;
}

// Same thing here, but now instead of at 0 it is at k 
fn threshold_gradient(x: f32, k: f32) -> f32 {
    return 0.0;
}

fn binary_sigmoid_gradient(x: f32, k: f32) -> f32 {
    let top = k * exp(-k * x);
    let bottom = pow(exp(-k * x) + 1, 2.0);

    if (bottom == 0.0) {
        return 0.0;
    }
    return top / bottom;
}

fn bipolar_sigmoid_gradient(x: f32, k: f32) -> f32 {
    let top_part_1 = k * exp(-k * x);
    let bot_part_1 = exp(-k * x) + 1;

    let top_part_2 = k * (1.0 - exp(-k * x)) * exp(-k * x);
    let bot_part_2 = pow(exp(-k * x) + 1, 2.0);

    if (bot_part_1 == 0.0 || bot_part_2 == 0.0) {
        return 0.0;
    }
    return (top_part_1 / bot_part_1) + (top_part_2 / bot_part_2);
}

// Return 0 if x is 0 to avoid undefined behavior
fn relu_gradient(x: f32) -> f32 {
    if (x > 0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

// Return a if x is 0 to avoid undefined behaviour
fn leaky_relu_gradient(x: f32, a: f32) -> f32 {
    if (x > 0) {
        return 1.0;
    } else {
        return a;
    }
}

fn hyperbolic_tangent_gradient(x: f32) -> f32 {
    let top = pow(exp(x) - exp(-x), 2.0);
    let bot = exp(x) + exp(-x);

    if (bot == 0.0) {
        return 0.0;
    }
    return 1.0 - (top / bot);
}

@compute @workgroup_size(256)
fn dense_layer_coefficient_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    // Num nodes
    let num_outputs = dims.x;
    // Num inputs
    let num_inputs = dims.y;

    // Compute the derivative of the activation function
    if (row < num_outputs) {
        switch activation_function.function_type {
            case STEP: {
                gradient_coefficient_intermediary[row] = step_gradient(intermediary[row]);
            }
            case THRESHOLD: {
                gradient_coefficient_intermediary[row] = threshold_gradient(intermediary[row], activation_function.function_parameter);
            }
            case BINARY_SIGMOID: {
                gradient_coefficient_intermediary[row] = binary_sigmoid_gradient(intermediary[row], activation_function.function_parameter);
            }
            case BIPOLAR_SIGMOID: {
                gradient_coefficient_intermediary[row] = bipolar_sigmoid_gradient(intermediary[row], activation_function.function_parameter);
            }
            case RELU: {
                gradient_coefficient_intermediary[row] = relu_gradient(intermediary[row]);
            }
            case LEAKY_RELU: {
                gradient_coefficient_intermediary[row] = leaky_relu_gradient(intermediary[row], activation_function.function_parameter);
            }
            case HYPERBOLIC_TANGENT: {
                gradient_coefficient_intermediary[row] = hyperbolic_tangent_gradient(intermediary[row]);
            }
            default: {}
        }
    }

    workgroupBarrier();
    
    //         \/ this is the gradient back prop from the next layer
    // [ 1 ] [ a ] 
    // [ 2 ] [ b ] 
    // [ 3 ] [ c ] 
    // [ 4 ] [ d ] 
    //   /\ this is the current activation function derivative input
    if (row < num_outputs) {
        gradient_coefficient[row] = gradient_coefficient_intermediary[row] * next_layers_gradient_coefficient[row];
    }


    // HACK This is a kinda sketchy way to do this
    if (row < num_inputs) {
        //            [ 1 2 3 4 5 ]
        //            [ a b c d e ]
        //            [ a b c d e ] <- this is the weights matrix
        //            [ a b c d e ]
        // [ x y z w ] <- this is the current coefficient
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < num_outputs; k++) {
            // let index = row * num_outputs + k;
            let index = k * num_inputs + row;
            sum += weights[index] * gradient_coefficient[k];
        }

        // This is the computation of the current coefficient multiplied
        // With the weight matrix as the first step in passing the
        // gradient coefficient back pipeline
        // This is dJ/do_l
        gradient_back_prop[row] = sum;
    }

    workgroupBarrier();
}
