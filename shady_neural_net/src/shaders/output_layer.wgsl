@group(0) @binding(0)
var<storage, read_write> input_buffer: array<f32>;

@group(1) @binding(0)
var<storage, read_write> output_buffer: array<f32>;

@group(1) @binding(1)
var<uniform> dims: vec2<u32>;


@compute @workgroup_size(256)
fn output_layer_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let m = dims.x;
    let n = dims.y;

    if (row < m) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < n; k++) {
            // let index = row * n + k;
            sum += input_buffer[k];
        }

        output_buffer[row] = sum;
    }

    workgroupBarrier();
}
