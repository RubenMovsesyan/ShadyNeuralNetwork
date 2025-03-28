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


@compute @workgroup_size(16, 16)
fn op_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;

    let a_rows = a_dimensions.x;
    let a_cols = a_dimensions.y;

    // Find the Max value of the inputs
    var max_value: f32 = matrix_a[0];
    if (row < a_rows && col < a_cols) {
        var a_index: u32 = 0;

        for (var i: u32 = 0; i < a_rows; i++) {
            if (a_transpose == 1) {
                a_index = i + a_rows * col;
            } else {
                a_index = i * a_cols + col;
            }

            max_value = max(max_value, matrix_a[a_index]);
        }
    }

    workgroupBarrier();

    // Get the exponential of all the elements of the matrix
    if (row < a_rows && col < a_cols) {
        var a_index: u32 = 0;

        if (a_transpose == 1) {
            a_index = row + a_rows * col;
        } else {
            a_index = row * a_cols + col;
        }

        matrix_a[a_index] = exp(matrix_a[a_index] - max_value);
    }

    workgroupBarrier();
    var sum: f32 = 0.0;

    // Sum all the exponentials together
    if (row < a_rows && col < a_cols) {
        var a_index: u32 = 0;

        for (var i: u32 = 0; i < a_rows; i++) {
            if (a_transpose == 1) {
                a_index = i + a_rows * col;
            } else {
                a_index = i * a_cols + col;
            }

            sum += matrix_a[a_index];
        }
    }

    workgroupBarrier();

    // Divide each element with the exponential sum
    if (row < a_rows && col < a_cols) {
        var a_index: u32 = 0;

        if (a_transpose == 1) {
            a_index = row + a_rows * col;
        } else {
            a_index = row * a_cols + col;
        }

        matrix_a[a_index] /= sum;
    }
}
