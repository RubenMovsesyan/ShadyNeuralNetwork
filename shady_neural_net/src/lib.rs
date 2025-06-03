use std::error::Error;
use std::ops::RangeBounds;

use anyhow::Result;
use damage::csv::parse_csv;
use gpu_math::GpuMath;
use gpu_math::math::matrix::Matrix;

pub mod layers;
pub mod neural_network;

fn one_hot_y(gpu_math: &GpuMath, y: &mut Matrix) -> Result<Matrix> {
    // let mut output = Matrix::with_shape((10, y.rows()));
    let output = Matrix::new(
        gpu_math,
        (10, y.rows),
        Some({
            let y_vals = y.get_inner()?;
            let mut output_vals = vec![0.0; 10 * y.rows as usize];

            for i in 0..10 {
                for j in 0..y.rows {
                    let output_index = (i * y.rows + j) as usize;
                    let y_index = j as usize;

                    if i as f32 == y_vals[y_index] {
                        output_vals[output_index] = 1.0;
                    } else {
                        output_vals[output_index] = 0.0;
                    }
                }
            }

            output_vals
        }),
    )?;

    Ok(output)
}

pub fn create_training_batches_from_csv<'a, R>(
    gpu_math: &GpuMath,
    csv_file_path: &str,
    labels_header: &String,
    data_headers: R,
    data_normalization_value: f32,
    data_size: usize,
    batch_size: u32,
    batch_limit: Option<u32>,
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
        .column_slice(labels_header, 0..(batch_size as usize * batches as usize))?
        .iter()
        .map(|&label_data| label_data.as_float().expect("Failed to parse labels data"))
        .collect::<Vec<f32>>();

    let mut training_labels_batches = Vec::with_capacity(batches as usize);

    for batch_index in 0..batches {
        training_labels_batches.push(Matrix::new(
            gpu_math,
            (batch_size, 1),
            Some({
                (0..batch_size)
                    .into_iter()
                    .map(|b| {
                        training_labels[batch_size as usize * batch_index as usize + b as usize]
                    })
                    .collect::<Vec<_>>()
            }),
        )?);
    }

    // for _ in 0..batches {
    //     training_labels_batches.push(Matrix::new(gpu_math, (batch_size, 1), None)?);
    // }

    // for (batch_index, label_batch) in training_labels_batches.iter_mut().enumerate() {
    //     for i in 0..label_batch.rows() {
    //         label_batch[(i, 0)] = training_labels[batch_size as usize * batch_index as usize + i];
    //     }

    //     *label_batch = one_hot_y(gpu_math, &mut label_batch)?;
    // }

    let image_train = data
        .columns_slice(data_headers, 0..(batch_size as usize * batches as usize))?
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

    let mut training_data_inputs = Vec::with_capacity(batches as usize);
    // for _ in 0..batches {
    //     training_data_inputs.push(Matrix::with_shape((data_size, batch_size)));
    // }

    // for (batch_index, training_batch) in training_data_inputs.iter_mut().enumerate() {
    //     for i in 0..training_batch.rows() {
    //         for j in 0..training_batch.cols() {
    //             training_batch[(i, j)] = image_train[batch_size * batch_index + j][i];
    //         }
    //     }
    // }

    println!("Image Train Rows: {}", image_train.len());
    println!("Image Train Cols: {}", image_train[0].len());

    for batch_index in 0..batches {
        training_data_inputs.push(Matrix::new(
            gpu_math,
            (data_size as u32, batch_size),
            Some({
                (0..(data_size as u32 * batch_size))
                    .into_iter()
                    .map(|b| {
                        let i = b as usize / data_size;
                        let j = b as usize % data_size;

                        let index = batch_size as usize * batch_index as usize + j;
                        let jndex = i;

                        image_train[jndex][index]
                    })
                    .collect::<Vec<_>>()
            }),
        )?);
    }

    Ok((training_data_inputs, training_labels_batches))
}
