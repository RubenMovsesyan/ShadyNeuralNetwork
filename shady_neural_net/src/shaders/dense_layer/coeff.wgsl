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
var<storage, read> gradient_coefficient: array<f32>;

@group(1) @binding(7)
var<storage, read_write> gradient_coefficient_output: array<f32>;

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
    // Num inputs
    let m = dims.x;
    // Num outputs
    let n = dims.y;

    // Compute the derivative of the activation function
    if (row < n) {
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
    
    //         \/ this is the current layers outputs
    // [ 1 ] [ a ] 
    // [ 2 ] [ b ]
    // [ 3 ] [ c ] 
    // [ 4 ] [ d ]
    //   /\ this is the current activation function derivative input
    if (row < n) {
        gradient_coefficient_output[row] = gradient_coefficient_intermediary[row] * gradient_coefficient[row];
    }

    workgroupBarrier();
}
