pub fn generate_x_y_function(x: f32, y: f32) -> [f32; 2] {
    let mut new_x = x * y + (x / y);
    let mut new_y = 0.5 * x + (y / x);

    let max_val = new_x.max(new_y);

    new_x = f32::exp(new_x - max_val);
    new_y = f32::exp(new_y - max_val);

    let exp_sum = new_x + new_y;

    new_x /= exp_sum;
    new_y /= exp_sum;

    [new_x, new_y]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = generate_x_y_function(0.4, -0.9);
        assert_eq!(result, [0.77652955, 0.22347045]);
    }
}
