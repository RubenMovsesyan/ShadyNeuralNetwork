@group(0) @binding(0)
var<storage, read> input_buffer: array<f32>;

@group(1) @binding(0)
var<uniform> dims: vec2<u32>;

@group(1) @binding(1)
var<storage, read> weights_buffer: array<f32>;

struct Bias {
    bias: f32,
    bias_weight: f32,
}

@group(1) @binding(2)
var<storage, read> bias_buffer: array<Bias>;

@group(1) @binding(3)
var<storage, read_write> intermediary_buffer: array<f32>;

@group(1) @binding(4)
var<storage, read_write> output_buffer: array<f32>;


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
            let index = row * n + k;
            sum += weights_buffer[index] * input_buffer[k];
        }

        sum += bias_buffer[row].bias * bias_buffer[row].bias_weight;

        // output_buffer[row] = sum;
        intermediary_buffer[row] = sum;
    }

    workgroupBarrier();

    
    var max_val: f32 = intermediary_buffer[0];

    for (var i = 0u; i < m; i++) {
        max_val = max(max_val, intermediary_buffer[i]);
    }

    workgroupBarrier();

    if (row < m) {
        output_buffer[row] = exp(intermediary_buffer[row] - max_val);
    }

    workgroupBarrier();
    
    var exp_sum: f32 = 0.0;
    for (var i = 0u; i < m; i++) {
        exp_sum += output_buffer[i];
    }

    workgroupBarrier();
    
    if (row < m) {
        output_buffer[row] /= exp_sum;
    }

    workgroupBarrier();
}
