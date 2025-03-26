mod adding_tests;
mod custom_tests;
mod dotting_tests;
mod exp_tests;
mod multiplying_tests;
mod subtracting_tests;
mod sum_tests;
mod transpose_tests;

#[cfg(test)]
mod test {
    use gpu_math::math::matrix::*;

    #[test]
    fn test_rand_with_shape() {
        let mat = Matrix::rand_with_shape((10, 5));

        println!("{}", mat);
        assert!(true);
    }

    #[test]
    fn test_setting_values() {
        let mut mat = Matrix::with_shape((10, 10));

        for i in 0..10 {
            mat[(i, i)] = 1.0;
        }

        println!("{}", mat);
        assert!(true);

        let mut mat = Matrix::with_shape((10, 5));

        for i in 0..10 {
            mat[(i, 0)] = 1.0;
        }

        println!("{}", mat);
        assert!(true);
    }
}
