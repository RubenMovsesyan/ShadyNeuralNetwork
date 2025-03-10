@group(0) @binding(0)
var<storage, read_write> weights_buffer: array<f32>;

struct Bias {
    bias: f32,
    bias_weight: f32,
}

@group(0) @binding(2)
var<uniform> dims: vec2<u32>;


@group(1) @binding(0)
var<uniform> norm_uniform: f32;

struct RegularizationFunction {
    function_type: u32,
    hyper_parameter: f32,
}

@group(1) @binding(1)
var<uniform> regularization_buffer: RegularizationFunction;

@group(1) @binding(2)
var<storage, read_write> regularization_output_buffer: array<f32>;

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
        let lambda = regularization_buffer.hyper_parameter;

        regularization_output_buffer[index] = lambda * (weight / norm_uniform);
    }
}
