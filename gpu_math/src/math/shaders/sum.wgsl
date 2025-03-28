// X = sum(A)

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

@group(0) @binding(3)
var<storage, read> _scalar: f32;

@group(0) @binding(4)
var<storage, read_write> reduce_buffer: array<f32>;

// Shared memory for the workgroups
var<workgroup> shared_reduce: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = global_id.x;
    let local_row = local_id.x;

    let size = a_dimensions.x * a_dimensions.y;

    if (row < size) {
        shared_reduce[local_row] = matrix_a[row];
    } else {
        shared_reduce[local_row] = 0.0;
    }

    workgroupBarrier();

    //                      \/ 256 / 2
    for (var stride: u32 = 128; stride > 0; stride /= 2) {
        if (local_row < stride) {
            shared_reduce[local_row] += shared_reduce[local_row + stride];
        }

        workgroupBarrier();
    }

    if (local_row == 0) {
        reduce_buffer[workgroup_id.x] = shared_reduce[0];
    }

    // // Compute half to the nearset power of 2
    // var half = u32(ceil(f32(size) / 2.0));
    // half = u32(pow(2.0, ceil(log(f32(half)) / log(2.0))));

    // if (row < half) {
    //     if (row + half < size) {
    //         reduce_buffer[row] = matrix_a[row] + matrix_a[row + half];
    //     } else if (row < size) {
    //         reduce_buffer[row] = matrix_a[row];
    //     } else {
    //         reduce_buffer[row] = 0.0;
    //     }
    // }

    // half /= 2;

    // workgroupBarrier();

    // while (half > 0) {
    //     if (row < half) {
    //         reduce_buffer[row] += reduce_buffer[row + half];
    //     }

    //     half /= 2;

    //     workgroupBarrier();
    // }

    // workgroupBarrier();
}
