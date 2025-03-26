// A = A + B

// Matirx A ------------------------------------
// Buffer of the A matrix
@group(0) @binding(0)
var<storage, read_write> matrix_a: array<f32>;

// Uniform of the dimensions of the A matrix
@group(0) @binding(1)
var<uniform> a_dimensions: vec2<u32>;

// Uniform of the transpose of the A matrix
@group(0) @binding(2)
var<uniform> a_transpose: u32;

// Matirx B ------------------------------------
// Buffer of the B matrix
@group(1) @binding(0)
var<storage, read> matrix_b: array<f32>;

// Uniform of the dimensions of the B matrix
@group(1) @binding(1)
var<uniform> b_dimensions: vec2<u32>;

// Uniform of the transpose of the A matrix
@group(1) @binding(2)
var<uniform> b_transpose: u32;

@compute @workgroup_size(16, 16)
fn vectored_add_in_place_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;

    let a_rows = a_dimensions.x;
    let a_cols = a_dimensions.y;

    let other_rows = b_dimensions.x;
    let other_cols = b_dimensions.y;

    if (row < a_rows && col < a_cols) {
        var a_index: u32 = 0;
        var other_index: u32 = 0;

        if (a_transpose == 1) {
            a_index = row + a_rows * col;
        } else {
            a_index = row * a_cols + col;
        }

        if (other_rows == a_rows) {
            matrix_a[a_index] += matrix_b[row];
        } else if (other_cols == a_cols) {
            matrix_a[a_index] += matrix_b[col];
        } else if (other_rows == a_cols) {
            matrix_a[a_index] += matrix_b[col];
        } else {
            matrix_a[a_index] += matrix_b[row];
        }
    }

    workgroupBarrier();
}
