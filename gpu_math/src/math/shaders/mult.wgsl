// X = m * A

// Matirx A ------------------------------------
// Buffer of the A matrix
@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

// Uniform of the dimensions of the A matrix
@group(0) @binding(1)
var<uniform> a_dimensions: vec2<u32>;

// Uniform of the transpose of the A matrix
@group(0) @binding(2)
var<uniform> a_transpose: u32;

// Uniform of the scalar to use for the multiplication
@group(0) @binding(3)
var<uniform> scalar: f32;

// Matirx X ------------------------------------
// Buffer of the output matrix
@group(1) @binding(0)
var<storage, read_write> matrix_x: array<f32>;

// Uniform of the dimensions of the output matrix
@group(1) @binding(1)
var<uniform> x_dimensions: vec2<u32>;

// Uniform of the transpose of the output matrix
@group(1) @binding(2)
var<uniform> x_transpose: u32;

@compute @workgroup_size(16, 16)
fn mult_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;

    let a_rows = a_dimensions.x;
    let a_cols = a_dimensions.y;

    let output_rows = x_dimensions.x;
    let output_cols = x_dimensions.y;

    if (row < output_rows && col < output_cols) {
        var a_index: u32 = 0;
        var x_index: u32 = 0;

        if (a_transpose == 1) {
            a_index = row + a_rows * col;
        } else {
            a_index = row * a_cols + col;
        }

        if (x_transpose == 1) {
            x_index = row + output_rows * col;
        } else {
            x_index = row * output_cols + col;
        }

        matrix_x[x_index] = scalar * matrix_a[a_index];
    }
}
