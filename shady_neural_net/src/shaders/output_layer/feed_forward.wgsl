// Custom Structs
struct Bias {
    bias: f32,
    bias_weight: f32,
}

// Buffer of inputs from the previous layer
@group(0) @binding(0)
var<storage, read> input_buffer: array<f32>;

// Buffer of dimensions of the weights matrix in this layer
@group(1) @binding(0)
var<uniform> dims: vec2<u32>;

// This layers Weights matrix
@group(1) @binding(1)
var<storage, read> weights_buffer: array<f32>;

// This Layers Bias vector
@group(1) @binding(2)
var<storage, read> bias_buffer: array<Bias>;

// Intermediary buffer for storing values before softmax
@group(1) @binding(3)
var<storage, read_write> intermediary_buffer: array<f32>;

// Output Buffer for values after softmax
@group(1) @binding(4)
var<storage, read_write> output_buffer: array<f32>;

@compute @workgroup_size(256)
fn output_layer_feed_forward_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    // Num Outputs
    let num_outputs = dims.x;
    // Num Inputs
    let num_inputs = dims.y;

    if (row < num_outputs) {
        // Compute the matrix multiplication with the
        // input vector
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < num_inputs; k++) {
            //          [x] 
            //          [y] 
            //          [z] 
            //          [w] 
            // [1 2 3 4]
            // [a b c d]
            // [a b c d]
            let index = row * num_inputs + k;

            sum += weights_buffer[index] * input_buffer[k];
        }

        // Add the biases to the output sum
        sum += bias_buffer[row].bias * bias_buffer[row].bias_weight;

        // store the sum in the intermediary buffer
        // to then compute the softmax
        intermediary_buffer[row] = sum;
        // output_buffer[row] = sum;
    }

    workgroupBarrier();

    // Compute the softmax of the output
    if (row < num_outputs) {
        var max_val: f32 = intermediary_buffer[0];
        for (var i: u32 = 0; i < num_outputs; i++) {
            max_val = max(max_val, intermediary_buffer[i]);
        }

        output_buffer[row] = exp(intermediary_buffer[row] - max_val);
    }

    // Make sure to synchronize the workgroups before overwritting the
    // output buffer
    workgroupBarrier();

    var exp_sum: f32 = 0.0;
    if (row < num_outputs) {
        for (var i: u32 = 0; i < num_outputs; i++) {
            exp_sum += output_buffer[i];
        }
    }

    // Make sure to synchronize the workgroups before overwritting the
    // output buffer
    workgroupBarrier();

    if (row < num_outputs) {
        output_buffer[row] /= exp_sum;
    }

    workgroupBarrier();
}
