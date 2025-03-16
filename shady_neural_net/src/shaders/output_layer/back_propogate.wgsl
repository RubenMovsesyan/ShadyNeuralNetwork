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

// The buffer where the Frobenius Norm is stored
@group(0) @binding(1)
var<uniform> frobenius_norm: f32;

// The buffer where the regularization function information is stored
@group(0) @binding(2)
var<uniform> regularization_info: RegularizationFunction;

// The output buffer where the regularization is stored after it is computed
@group(0) @binding(3)
var<storage, read_write> regularization_output: array<f32>;

// Dimensions of the weights buffer
@group(0) @binding(4)
var<uniform> dims: vec2<u32>;

// The buffer where the weigths are stored
@group(0) @binding(5)
var<storage, read> weights: array<f32>;

// The buffer where the gradient is calculated
@group(0) @binding(6)
var<storage, read_write> gradient: array<f32>;

// The buffer where the gradient coefficient is stored
@group(0) @binding(7)
var<storage, read> gradient_coefficient: array<f32>;

// The inputs of the layer
@group(1) @binding(0)
var<storage, read> input_buffer: array<f32>;

fn calculate_gradient(index: u32, row: u32, col: u32) -> f32 {
    let dJdo = gradient_coefficient[row];
    let h = input_buffer[col];
    let regularization = regularization_output[index];

    return dJdo * h + regularization;
}

@compute @workgroup_size(16, 16)
fn output_layer_back_propogate_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    // Num inputs
    let m = dims.x;
    // Num outputs
    let n = dims.y;

    // The index of the weights buffer
    //            [ x ]
    //            [ y ]
    //            [ z ]
    //            [ w ]
    // [ 1 2 3 4 ]
    // [ a b c d ]
    // [ a b c d ]
    let index = row * m + col;

    if (row < m && col < n) {
        let weight = weights[index];

        // Get the regularization term here basd on the weights
        switch regularization_info.function_type {
            case LASSO: {
                // Find the gradient of the l1 norm
                var grad: f32 = 0.0;

                if (weight > 0.0) {
                    grad = 1.0;
                } else if (weight < 0.0) {
                    grad = -1.0;
                }

                regularization_output[index] = lambda_1 * grad * l_1_norm * weight;
            }
            case RIDGE: {
                regularization_output[index] = lambda_1 * (weight / frobenius_norm);
            }
            case ELASTIC_NET_REGRESSION: {
                // Find the gradient of the L1 norm
                var grad: f32 = 0.0;

                if (weight > 0.0) {
                    grad = 1.0;
                } else if (weight < 0.0) {
                    grad = -1.0;
                }

                regularization_output[index] = lambda_1 * grad * l_1_norm * weight + lambda_2 * (weight / frobenius_norm);
            }
        }

        gradient[index] = calculate_gradient(index, col, row);
    }
}
