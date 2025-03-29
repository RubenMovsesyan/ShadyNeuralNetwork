use std::error::Error;
use std::ops::RangeBounds;

use damage::csv::parse_csv;
use gpu_math::math::matrix::Matrix;

pub mod layers;
pub mod neural_network;

fn one_hot_y(y: &Matrix) -> Matrix {
    let mut output = Matrix::with_shape((10, y.rows()));

    for i in 0..output.rows() {
        for j in 0..output.cols() {
            output[(i, j)] = if i as f32 == y[(j, 0)] { 1.0 } else { 0.0 };
        }
    }

    output
}

pub fn create_training_batches_from_csv<'a, R>(
    csv_file_path: &str,
    labels_header: &String,
    data_headers: R,
    data_normalization_value: f32,
    data_size: usize,
    batch_size: usize,
    batch_limit: Option<usize>,
) -> Result<(Vec<Matrix>, Vec<Matrix>), Box<dyn Error>>
where
    R: RangeBounds<String>,
{
    let data = parse_csv(csv_file_path)?;

    let batches = if let Some(limit) = batch_limit {
        limit
    } else {
        1
    };

    let training_labels = data
        .column_slice(labels_header, 0..(batch_size * batches))?
        .iter()
        .map(|&label_data| label_data.as_float().expect("Failed to parse labels data"))
        .collect::<Vec<f32>>();

    let mut training_labels_batches = Vec::with_capacity(batches);
    for _ in 0..batches {
        training_labels_batches.push(Matrix::with_shape((batch_size, 1)));
    }

    for (batch_index, label_batch) in training_labels_batches.iter_mut().enumerate() {
        for i in 0..label_batch.rows() {
            label_batch[(i, 0)] = training_labels[batch_size * batch_index + i];
        }

        *label_batch = one_hot_y(&label_batch);
    }

    let image_train = data
        .columns_slice(data_headers, 0..(batch_size * batches))?
        .iter()
        .map(|&data_array| {
            data_array
                .iter()
                .map(|data| {
                    data.as_float().expect("Failed to parse data") / data_normalization_value
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();

    let mut training_data_inputs = Vec::with_capacity(batches);
    for _ in 0..batches {
        training_data_inputs.push(Matrix::with_shape((data_size, batch_size)));
    }

    for (batch_index, training_batch) in training_data_inputs.iter_mut().enumerate() {
        for i in 0..training_batch.rows() {
            for j in 0..training_batch.cols() {
                training_batch[(i, j)] = image_train[batch_size * batch_index + j][i];
            }
        }
    }

    Ok((training_data_inputs, training_labels_batches))
}
