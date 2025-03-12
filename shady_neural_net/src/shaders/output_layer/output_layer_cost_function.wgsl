@group(0) @binding(0)
var<uniform> dims: vec2<u32>;

@group(0) @binding(1)
var<storage, read> output_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read> expected_values_buffer: array<f32>;

@group(0) @binding(3)
var<storage, read_write> loss_function_buffer: array<f32>;

@group(0) @binding(4)
var<storage, read_write> loss_function_gradient_buffer: array<f32>;

fn binary_cross_entropy_loss_gradient(predicted: f32, expected: f32) -> f32 {
    let top = predicted - expected;
    let bottom = (predicted - 1.0) * predicted;

    return -1.0 * (top / bottom);
}

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
    let m = dims.x;

    if (row < m) {
        let predicted = output_buffer[row];
        let expected = expected_values_buffer[row];

        loss_function_buffer[row] = binary_cross_entropy_loss(predicted, expected);
        loss_function_gradient_buffer[row] = binary_cross_entropy_loss_gradient(predicted, expected);
    }
}
