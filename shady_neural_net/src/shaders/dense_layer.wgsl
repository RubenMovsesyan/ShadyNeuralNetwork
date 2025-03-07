@group(0) @binding(0)
var<storage, read_write> input_buffer: array<f32>;

@group(1) @binding(0)
var<storage, read> weights_buffer: array<f32>;

struct Bias {
    bias: f32,
    bias_weight: f32,
}

@group(1) @binding(1)
var<storage, read> bias_buffer: array<Bias>;

@group(1) @binding(2)
var<uniform> dims: vec2<u32>;

@group(2) @binding(0)
var<storage, read_write> output_buffer: array<f32>;



@compute @workgroup_size(256)
fn dense_layer_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let m = dims.x;
    let n = dims.y;

    if (row < m) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < n; k++) {
            let index = row * n + k;
            sum += weights_buffer[index] * input_buffer[k];
        }

        sum += bias_buffer[row].bias * bias_buffer[row].bias_weight;

        output_buffer[row] = sum;
    }
    
    workgroupBarrier();
}
