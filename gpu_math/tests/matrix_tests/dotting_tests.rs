#[cfg(test)]
mod tests {
    use gpu_math::matrix::Matrix;

    #[test]
    fn test_matrix_dotting() {
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

        Matrix::dot(&mat1, &mat2, &dest).expect("Failed");

        let expected = Matrix::new(
            (3, 3),
            Some(vec![15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0, 111.0]),
        )
        .expect("Failed");

        assert_eq!(dest, expected);
    }
}
