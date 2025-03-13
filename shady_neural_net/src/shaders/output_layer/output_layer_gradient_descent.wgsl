@group(0) @binding(0)
var<uniform> learning_rate: f32;

@group(0) @binding(1)
var<storage, read> gradient_buffer: array<f32>;

@group(0) @binding(2)
var<storage, read_write> weights_buffer: array<f32>;

@group(0) @binding(3)
var<uniform> dims: vec2<u32>;


@compute @workgroup_size(16, 16)
fn output_layer_gradient_descent_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let row = global_id.x;
    let col = global_id.y;
    let m = dims.x;
    let n = dims.y;

    if (row < m && col < n) {
        let index = row * n + col;

        let weight = weights_buffer[index];
        let gradient = gradient_buffer[index];

        weights_buffer[index] = weight - (learning_rate * gradient);
    }

    workgroupBarrier();
}
