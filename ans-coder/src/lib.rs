use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

pub mod internal;

use internal::{
    distributions::{Leaky, NonLeaky},
    AnsCoder,
};

#[pymodule]
fn ans(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Coder>()?;
    Ok(())
}

/// Wrapper around [`internal::AnsCoder`] with a python-compatible API.
///
/// [`internal::AnsCoder`]: internal/struct.AnsCoder.html
#[pyclass]
pub struct Coder {
    inner: AnsCoder,
}

#[pymethods]
impl Coder {
    #[new]
    pub fn new(compressed: PyReadonlyArray1<u32>) -> Self {
        let compressed = compressed.as_slice().unwrap();
        let inner = if compressed.is_empty() {
            AnsCoder::new()
        } else {
            AnsCoder::with_compressed_data(compressed.iter().cloned().collect())
        };

        Self { inner }
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    pub fn num_words(&self) -> usize {
        self.inner.num_words()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn copy_compressed(&self, destination: &PyArray1<u32>) {
        assert_eq!(destination.len(), self.num_words());

        let destination = destination.as_cell_slice().unwrap();
        let buf = self.inner.get_buf();
        let state = self.inner.get_state();

        for (&src, dest) in buf.iter().zip(destination) {
            dest.set(src)
        }
        destination[buf.len()].set(state as u32);
        destination[buf.len() + 1].set((state >> 32) as u32);
    }

    pub fn push_gaussian_symbols(
        &mut self,
        symbols: PyReadonlyArray1<i32>,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        means: PyReadonlyArray1<f64>,
        stds: PyReadonlyArray1<f64>,
        leaky: bool,
    ) {
        if leaky {
            self.inner.push_gaussian_symbols::<Leaky>(
                symbols.as_slice().unwrap(),
                min_supported_symbol,
                max_supported_symbol,
                means.as_slice().unwrap(),
                stds.as_slice().unwrap(),
            );
        } else {
            self.inner.push_gaussian_symbols::<NonLeaky>(
                symbols.as_slice().unwrap(),
                min_supported_symbol,
                max_supported_symbol,
                means.as_slice().unwrap(),
                stds.as_slice().unwrap(),
            );
        }
    }

    pub fn pop_gaussian_symbols(
        &mut self,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        means: PyReadonlyArray1<f64>,
        stds: PyReadonlyArray1<f64>,
        symbols_out: &PyArray1<i32>,
        leaky: bool,
    ) {
        assert_eq!(means.len(), symbols_out.len());

        let symbols = if leaky {
            self.inner
                .pop_gaussian_symbols::<Leaky>(
                    min_supported_symbol,
                    max_supported_symbol,
                    means.as_slice().unwrap(),
                    stds.as_slice().unwrap(),
                )
                .unwrap()
        } else {
            self.inner
                .pop_gaussian_symbols::<NonLeaky>(
                    min_supported_symbol,
                    max_supported_symbol,
                    means.as_slice().unwrap(),
                    stds.as_slice().unwrap(),
                )
                .unwrap()
        };

        let symbols_out = symbols_out.as_cell_slice().unwrap();
        for (src, dest) in symbols.into_iter().zip(symbols_out) {
            dest.set(src)
        }
    }

    pub fn push_iid_categorical_symbols(
        &mut self,
        symbols: PyReadonlyArray1<i32>,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        min_provided_symbol: i32,
        probabilities: PyReadonlyArray1<f64>,
    ) {
        self.inner.push_iid_categorical_symbols(
            symbols.as_slice().unwrap(),
            min_supported_symbol,
            max_supported_symbol,
            min_provided_symbol,
            probabilities.as_slice().unwrap(),
        );
    }

    pub fn pop_iid_categorical_symbols(
        &mut self,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        min_provided_symbol: i32,
        probabilities: PyReadonlyArray1<f64>,
        symbols_out: &PyArray1<i32>,
    ) {
        let symbols = self
            .inner
            .pop_iid_categorical_symbols(
                symbols_out.len(),
                min_supported_symbol,
                max_supported_symbol,
                min_provided_symbol,
                probabilities.as_slice().unwrap(),
            )
            .unwrap();

        let symbols_out = symbols_out.as_cell_slice().unwrap();
        for (src, dest) in symbols.into_iter().zip(symbols_out) {
            dest.set(src)
        }
    }
}
