@group(0) @binding(0)
var<storage, read_write> input_buffer: array<f32>;

@group(1) @binding(0)
var<storage, read> weights_buffer: array<f32>;

struct Bias {
    bias: f32,
    bias_weight: f32,
}

@group(1) @binding(1)
var<storage, read> bias_buffer: array<Bias>;

@group(1) @binding(2)
var<uniform> dims: vec2<u32>;

struct ActivationFunctionDescriptor {
    function_type: u32,
    function_parameter: f32,
}

@group(1) @binding(3)
var<uniform> activation_function: ActivationFunctionDescriptor;

@group(2) @binding(0)
var<storage, read_write> output_buffer: array<f32>;


// Here are the different activation functions
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
    let bottom = 1.0 + exp(-k  * x);

    return 1.0 / bottom;
}

fn bipolar_sigmoid(x: f32, k: f32) -> f32 {
    let top = 1.0 - exp(-k * x);
    let bottom = 1.0 + exp(-k * x);

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

    return top / bottom;
}


@compute @workgroup_size(256)
fn dense_layer_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let m = dims.x;
    let n = dims.y;

    if (row < m) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < n; k++) {
            let index = row * n + k;
            sum += weights_buffer[index] * input_buffer[k];
        }

        sum += bias_buffer[row].bias * bias_buffer[row].bias_weight;


        // Run the sum through the activation function
        switch activation_function.function_type {
            case 0u: {
                sum = step(sum);
            }
            case 1u: {
                sum = threshold(sum, activation_function.function_parameter);
            }
            case 2u: {
                sum = binary_sigmoid(sum, activation_function.function_parameter);
            }
            case 3u: {
                sum = bipolar_sigmoid(sum, activation_function.function_parameter);
            }
            case 4u: {
                sum = relu(sum);
            }
            case 5u: {
                sum = leaky_relu(sum, activation_function.function_parameter);
            }
            case 6u: {
                sum = hyperbolic_tangent(sum);
            }
            default: {
                sum = step(sum);
            }
        }

        
        output_buffer[row] = sum;
    }
    
    workgroupBarrier();
}
