// Dimensions of the weigths matrix for this layer
@group(0) @binding(0)
var<uniform> dims: vec2<u32>;

// The output buffer where the feed forward computation is stored
@group(0) @binding(1)
var<storage, read> output_buffer: array<f32>;

// The buffer where the expected values are stored
@group(0) @binding(2)
var<storage, read> expected_values_buffer: array<f32>;

// The buffer where the loss function will be stored
@group(0) @binding(3)
var<storage, read_write> loss_function_buffer: array<f32>;


// TEMP
@group(0) @binding(4)
var<storage, read_write> gradient_coefficient_buffer: array<f32>;

// -((o_n * ln(y_n) + ((1 - y_n) * ln(1 - o_n)))
fn binary_cross_entropy_loss_gradient(predicted: f32, expected: f32) -> f32 {
    let top = predicted - expected;
    let bottom = (predicted - 1.0) * predicted;

    if (bottom == 0.0) {
        return 0.0;
    }
    
    return -1.0 * (top / bottom);
}


// -(o - y) / ((o - 1)o)
fn binary_cross_entropy_loss(predicted: f32, expected: f32) -> f32 {
    let part_1 = expected * log(predicted);
    let part_2 = (1.0 - expected) * log(1.0 - predicted);

    return -1.0 * (part_1 + part_2);
}

@compute @workgroup_size(256)
fn output_layer_loss_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Binary cross entropy loss
    let row = global_id.x;
    // Num outputs
    let n = dims.y;

    if (row < n) {
        let predicted = output_buffer[row];
        let expected = expected_values_buffer[row];

        loss_function_buffer[row] = binary_cross_entropy_loss(predicted, expected);

        // TEMP
        gradient_coefficient_buffer[row] = binary_cross_entropy_loss_gradient(predicted, expected);
    }

    workgroupBarrier();
}
