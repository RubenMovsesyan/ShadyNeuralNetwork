use criterion::Criterion;

use gpu_math::matrix::Matrix;

pub fn bench_matrix_dot(c: &mut Criterion) {
    gpu_math::init();

    let mat1 = Matrix::new(
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let mat2 = Matrix::new(
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let dest = Matrix::new((3, 3), None).expect("Failed");

    c.bench_function("dot", |b| {
        b.iter(|| {
            Matrix::dot(&mat1, &mat2, &dest).expect("Failed");
        })
    });
}
