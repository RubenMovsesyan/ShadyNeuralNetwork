use std::{
    error::Error,
    fmt::{Debug, Display},
    fs::File,
    io::{BufRead, BufReader},
    ops::{Index, Range},
    path::Path,
    slice::SliceIndex,
};

#[derive(Debug)]
pub struct CSV {
    headers: Vec<String>,
    records: Vec<Record>,
}

impl Display for CSV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for header in self.headers.iter() {
            write!(f, " {:>10} |", header)?;
        }
        writeln!(f)?;

        for _ in 0..(self.headers.len() * 13) {
            write!(f, "-")?;
        }

        for record in self.records.iter() {
            for data in record.data.iter() {
                write!(f, " {:>10} |", data)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

impl<Idx> Index<Idx> for CSV
where
    Idx: SliceIndex<[Record]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.records[index]
    }
}

impl CSV {
    pub fn column_slice<'a>(
        &'a self,
        header: &'a str,
        index: Range<usize>,
    ) -> Option<Vec<&'a Data>> {
        let data_index = {
            let mut ind = -1;

            for (heading_index, heading) in self.headers.iter().enumerate() {
                if heading == header {
                    ind = heading_index as isize;
                    break;
                }
            }

            if ind < 0 {
                return None;
            }

            ind as usize
        };

        Some(
            self.records[index]
                .iter()
                .map(|record| &record.data[data_index])
                .collect(),
        )
    }

    pub fn columns_slice<'a>(
        &'a self,
        headers: Range<&str>,
        index: Range<usize>,
    ) -> Option<Vec<&'a [Data]>> {
        let data_range = {
            let start = match self
                .headers
                .iter()
                .position(|header| header.as_str() == headers.start)
            {
                Some(start) => start,
                None => return None,
            };

            let end = match self
                .headers
                .iter()
                .position(|header| header.as_str() == headers.end)
            {
                Some(end) => end,
                None => return None,
            };

            start..end
        };

        Some(
            self.records[index]
                .iter()
                .map(|record| &record.data[data_range.clone()])
                .collect(),
        )
    }
}

#[derive(Debug)]
pub struct IncompatibleDataError;

impl Display for IncompatibleDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Incompatible Data Type")
    }
}

impl Error for IncompatibleDataError {}

#[derive(Debug)]
pub enum Data {
    Integer(u32),
    Float(f32),
    Str(String),
}

impl Data {
    pub fn as_integer(&self) -> Result<u32, IncompatibleDataError> {
        use Data::*;
        match self {
            Integer(int) => Ok(*int),
            Float(float) => Ok(*float as u32),
            Str(_) => Err(IncompatibleDataError),
        }
    }

    pub fn as_float(&self) -> Result<f32, IncompatibleDataError> {
        use Data::*;
        match self {
            Integer(int) => Ok(*int as f32),
            Float(float) => Ok(*float),
            Str(_) => Err(IncompatibleDataError),
        }
    }

    pub fn as_str(&self) -> Result<String, IncompatibleDataError> {
        use Data::*;
        match self {
            Integer(int) => Ok(int.to_string()),
            Float(float) => Ok(float.to_string()),
            Str(string) => Ok(string.to_string()),
        }
    }
}

impl Display for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer(val) => write!(f, "{:>10}", val)?,
            Self::Float(val) => write!(f, "{:>10.2}", val)?,
            Self::Str(string) => write!(f, "{:10}", string)?,
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct Record {
    data: Vec<Data>,
}

impl<Idx> Index<Idx> for Record
where
    Idx: SliceIndex<[Data]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.data[index]
    }
}

pub fn parse_csv<P>(path: P) -> std::io::Result<CSV>
where
    P: AsRef<Path>,
{
    // Open the file
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut headers: Vec<String> = Vec::new();
    // For making sure the size of the vecs are efficiently created at the start
    let mut data_size: usize = 0;
    let mut records: Vec<Record> = Vec::new();
    // Read the lines of the csv file
    for (index, line) in reader.lines().enumerate() {
        let line = line?;

        if index == 0 {
            for header in line.split(",") {
                data_size += 1;
                headers.push(String::from(header));
            }

            continue;
        }

        let mut data: Vec<Data> = Vec::with_capacity(data_size);

        // For each line store the data as the appropriate value
        for element in line.split(",") {
            if let Ok(val) = element.parse::<u32>() {
                data.push(Data::Integer(val));
                continue;
            }

            if let Ok(val) = element.parse::<f32>() {
                data.push(Data::Float(val));
                continue;
            }

            data.push(Data::Str(String::from(element)));
        }

        records.push(Record { data });
    }

    Ok(CSV { headers, records })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv() {
        println!(
            "{}",
            parse_csv("../test_files/mnist_test.csv").expect("Failed")
        )
    }

    #[test]
    fn test_csv_heading_slice() {
        let csv = parse_csv("../test_files/mnist_train.csv").expect("Failed");

        println!(
            "Labels 0 - 10: {:#?}",
            csv.column_slice("label", 0..10).unwrap()
        );

        assert!(true);
    }

    #[test]
    fn test_csv_slice_heading_slice() {
        let csv = parse_csv("../test_files/mnist_train.csv").expect("Failed");

        println!(
            "{} - {} 8 - 14: {:#?}",
            "6x13",
            "6x22",
            csv.columns_slice("6x13".."6x22", 8..14).unwrap()
        );

        assert!(true);
    }
}
