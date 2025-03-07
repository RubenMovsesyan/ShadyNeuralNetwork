@group(0) @binding(0)
var<storage, read_write> input_buffer: array<f32>;

@compute @workgroup_size(256)
fn input_layer_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // let index = global_id.x;

    // input_buffer[index] = 12.0;
    workgroupBarrier();
}
