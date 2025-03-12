@group(0) @binding(0)
var<storage, read> next_layer_gradient_coefficient_buffer: array<f32>;

@group(0) @binding(1)
var<storage, read> next_layer_weights_buffer: array<f32>;

@group(0) @binding(2)
var<uniform> next_layer_dimensions: vec2<u32>;



@group(1) @binding(0)
var<storage, read_write> gradient_coefficient_buffer: array<f32>;


@compute @workgroup_size(256)
fn dense_layer_coefficient_forming_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let m = next_layer_dimensions.x;
    let n = next_layer_dimensions.y;

    if (row < n) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < m; k++) {
            let index = row * m + k;
            sum += next_layer_weights_buffer[index] * next_layer_gradient_coefficient_buffer[k];
        }

        // TODO: add the last term necessary to get the gradient coefficient buffer
        // to work
        gradient_coefficient_buffer[row] = sum;
    }

    workgroupBarrier();
}
