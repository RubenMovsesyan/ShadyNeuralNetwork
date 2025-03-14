@group(0) @binding(0)
var<storage, read> next_layer_gradient_coefficient_buffer: array<f32>;

@group(0) @binding(1)
var<storage, read> next_layer_weights_buffer: array<f32>;

@group(0) @binding(2)
var<uniform> next_layer_dimensions: vec2<u32>;



@group(1) @binding(0)
var<storage, read_write> gradient_coefficient_buffer: array<f32>;


struct ActivationFunctionDescriptor {
    function_type: u32,
    function_parameter: f32,
}

// Activation function informaiton
// The function type
// and the parameter if necessary
@group(1) @binding(1)
var<uniform> activation_function: ActivationFunctionDescriptor;

@group(1) @binding(2)
var<storage, read> input_buffer: array<f32>;


// Constants
const STEP: u32 = 0;
const THRESHOLD: u32 = 1;
const BINARY_SIGMOID: u32 = 2;
const BIPOLAR_SIGMOID: u32 = 3;
const RELU: u32 = 4;
const LEAKY_RELU: u32 = 5;
const HYPERBOLIC_TANGENT: u32 = 6;


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

    return top / bottom;
}

fn bipolar_sigmoid_gradient(x: f32, k: f32) -> f32 {
    let top_part_1 = k * exp(-k * x);
    let bot_part_1 = exp(-k * x) + 1;

    let top_part_2 = k * (1.0 - exp(-k * x)) * exp(-k * x);
    let bot_part_2 = pow(exp(-k * x) + 1, 2.0);

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

    return 1.0 - (top / bot);
}



@compute @workgroup_size(256)
fn dense_layer_coefficient_forming_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let m = next_layer_dimensions.x;
    let n = next_layer_dimensions.y;

    if (row < m) {
        // Multiply the last gradient coefficient vector with the next layers weight matrix
        // as that is what do_n+1/dh_n is
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < n; k++) {
            let index = row * n + k;
            sum += next_layer_weights_buffer[index] * next_layer_gradient_coefficient_buffer[k];
        }

        switch activation_function.function_type {
            case STEP: {
                sum *= step_gradient(input_buffer[row]);
            }
            case THRESHOLD: {
                sum *= threshold_gradient(input_buffer[row], activation_function.function_parameter);
            }
            case BINARY_SIGMOID: {
                sum *= binary_sigmoid_gradient(input_buffer[row], activation_function.function_parameter);
            }
            case BIPOLAR_SIGMOID: {
                sum *= bipolar_sigmoid_gradient(input_buffer[row], activation_function.function_parameter);
            }
            case RELU: {
                sum *= relu_gradient(input_buffer[row]);
            }
            case LEAKY_RELU: {
                sum *= leaky_relu_gradient(input_buffer[row], activation_function.function_parameter);
            }
            case HYPERBOLIC_TANGENT: {
                sum *= hyperbolic_tangent_gradient(input_buffer[row]);
            }
            default: {
                sum *= step_gradient(input_buffer[row]);
            }
        }

        gradient_coefficient_buffer[row] = sum;
    }

    workgroupBarrier();
}
