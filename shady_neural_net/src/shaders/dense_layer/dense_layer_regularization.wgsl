@group(0) @binding(0)
var<uniform> l_1_norm_uniform: f32;

@group(0) @binding(1)
var<uniform> frobenius_norm_uniform: f32;

struct RegularizationFunction {
    function_type: u32,
    hyper_parameter_1: f32,
    hyper_parameter_2: f32,
}

@group(0) @binding(2)
var<uniform> regularization_info_buffer: RegularizationFunction;

@group(0) @binding(3)
var<storage, read_write> regularization_output_buffer: array<f32>;

@group(0) @binding(4)
var<uniform> dims: vec2<u32>;

@group(0) @binding(5)
var<storage, read> weights_buffer: array<f32>;


const LASSO: u32 = 0;
const RIDGE: u32 = 1;
const ELASTIC_NET_REGRESSION: u32 = 2;

@compute @workgroup_size(16, 16)
fn dense_layer_regularization_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let m = dims.x;
    let n = dims.y;

    if (row < m && col < n) {
        let index = row * n + col;

        let weight = weights_buffer[index];
        let lambda_1 = regularization_info_buffer.hyper_parameter_1;
        let lambda_2 = regularization_info_buffer.hyper_parameter_2;

        if (regularization_info_buffer.function_type == LASSO) {
            var grad: f32 = 0.0;

            // Finding the derivative of the L1 Norm
            if (weight > 0.0) {
                grad = 1.0;
            } else if (weight < 0.0) {
                grad = -1.0;
            }

            regularization_output_buffer[index] = lambda_1 * grad * l_1_norm_uniform * weight;
        } else if (regularization_info_buffer.function_type == RIDGE) {
            regularization_output_buffer[index] = lambda_1 * (weight / frobenius_norm_uniform);
        } else if (regularization_info_buffer.function_type == ELASTIC_NET_REGRESSION) {
            var grad: f32 = 0.0;

            // Finding the derivative of the L1 Norm
            if (weight > 0.0) {
                grad = 1.0;
            } else if (weight < 0.0) {
                grad = -1.0;
            }

            regularization_output_buffer[index] = lambda_1 * grad * l_1_norm_uniform * weight + lambda_2 * (weight / frobenius_norm_uniform);
        }
    }
}
