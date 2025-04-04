// A = A - B

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
fn sub_in_place_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;

    let a_rows = a_dimensions.x;
    let a_cols = a_dimensions.y;

    let b_rows = b_dimensions.x;
    let b_cols = b_dimensions.y;

    if (row < a_rows && col < a_cols) {
        var a_index: u32 = 0;
        var b_index: u32 = 0;

        if (a_transpose == 1) {
            a_index = row + a_cols * col;
        } else {
            a_index = row * a_cols + col;
        }

        if (b_transpose == 1) {
            b_index = row + b_rows * col;
        } else {
            b_index = row * b_cols + col;
        }

        matrix_a[a_index] -= matrix_b[b_index];
    }

    workgroupBarrier();
}
