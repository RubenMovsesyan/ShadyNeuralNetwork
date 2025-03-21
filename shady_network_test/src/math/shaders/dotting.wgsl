// X = AB

// Buffer of the A matrix
@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

// Uniform of the dimensions of the A matrix
@group(0) @binding(1)
var<uniform> a_dimensions: vec2<u32>;

// Buffer of the B matrix
@group(1) @binding(0)
var<storage, read> matrix_b: array<f32>;

// Uniform of the dimensions of the B matrix
@group(1) @binding(1)
var<uniform> b_dimensions: vec2<u32>;

// Buffer of the output matrix
@group(2) @binding(0)
var<storage, read_write> matrix_x: array<f32>;

// Uniform of the output dimensions
@group(2) @binding(1)
var<uniform> output_dimensions: vec2<u32>;

@compute @workgroup_size(16, 16)
fn dot_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;

    let a_rows = a_dimensions.x;
    let a_cols = a_dimensions.y;

    let b_rows = b_dimensions.x;
    let b_cols = b_dimensions.y;

    let output_rows = output_dimensions.x;
    let output_cols = output_dimensions.y;

    let dot_size = a_cols;

    if (row < output_rows && col < output_cols) {
        var sum: f32 = 0.0;
        let output_index = row * output_cols + col;
        for (var k: u32 = 0; k < dot_size; k++) {
            let a_index = row * a_cols + k;
            let b_index = k * b_cols + col;

            sum += matrix_a[a_index] * matrix_b[b_index];
        }

        matrix_x[output_index] = sum;
    }

    workgroupBarrier();
}
