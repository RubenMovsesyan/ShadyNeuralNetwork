@group(0) @binding(0)
var<storage, read_write> input_buffer: array<f32>;

@group(1) @binding(0)
var<storage, read_write> output_buffer: array<f32>;

@group(1) @binding(1)
var<uniform> dims: vec2<u32>;

@group(1) @binding(2)
var<storage, read> weights_buffer: array<f32>;

struct Bias {
    bias: f32,
    bias_weight: f32,
}

@group(1) @binding(3)
var<storage, read> bias_buffer: array<Bias>;

@group(2) @binding(0)
var<storage, read_write> loss_function_buffer: array<f32>;

@group(2) @binding(1)
var<storage, read> expected_values_buffer: array<f32>;


fn binary_cross_entropy_loss(predicted: f32, expected: f32) -> f32 {
    // log is natural logarithm
    let part_1 = expected * log(predicted);
    let part_2 = (1.0 - expected) * log(1.0 - predicted);

    return -1.0 * (part_1 + part_2);
}


@compute @workgroup_size(256)
fn output_layer_cost_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Binary Cross Entropy Loss
    let row = global_id.x;

    let predicted = output_buffer[row];
    let expected = expected_values_buffer[row];

    loss_function_buffer[row] = binary_cross_entropy_loss(predicted, expected);
}
