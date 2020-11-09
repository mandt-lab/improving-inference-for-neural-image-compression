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

/// An entropy coder based on Asymmetric Numeral Systems (ANS).
///
/// Note that this entropy coder is a stack (a "last in first out" data
/// structure). You can push symbols on the stack using the methods
/// `push_gaussian_symbols` or `push_iid_categorical_symbols`, and then pop
/// them off *in reverse order* using the methods `pop_gaussian_symbols` or
/// `pop_iid_categorical_symbols`, respectively.
///
/// To retrieve the compressed data that is currently on the stack, first
/// query for the size of the compressed data using the method `num_words()`,
/// then allocate a numpy array of this size and dtype `uint32`, and finally
/// call pass this array to the method `copy_compressed`.
///
/// To decompress data, pass the compressed data to the constructor (and then
/// decompress the symbols in reverse order).
///
/// # Constructor
///
/// Coder(compressed)
///
/// Arguments:
/// compressed (optional) -- compressed data, as a numpy array with dtype
///     `uint32`. Only needed for decompression. If not supplied or empty then
///     the entropy coder will be constructed with the smallest allowed original
///     state (64 bits).
#[pyclass]
pub struct Coder {
    inner: AnsCoder,
}

#[pymethods]
impl Coder {
    /// Constructs a new entropy coder, optionally passing initial compressed data.
    #[new]
    pub fn new(compressed: Option<PyReadonlyArray1<u32>>) -> Self {
        let inner = match compressed {
            Some(c) if !c.is_empty() => {
                AnsCoder::with_compressed_data(c.as_slice().unwrap().iter().cloned().collect())
            }
            _ => AnsCoder::new(),
        };

        Self { inner }
    }

    /// Resets the coder for compression.
    ///
    /// After calling this method, the method `is_empty` will return `True`.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns the number of compressed bits currently on the stack.
    ///
    /// This is always a multiple of 32, and always at least 64.
    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    /// Returns the number of `uint32` words currently on the stack.const
    ///
    /// See method `copy_compressed` for a usage example.
    pub fn num_words(&self) -> usize {
        self.inner.num_words()
    }

    /// Returns `True` iff the coder is in its default initial state.
    ///
    /// The default initial state is the state returned by the constructor when
    /// called without arguments, or the state to which the coder is set when
    /// calling `clear`.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Copies the compressed data to the provided numpy array.
    ///
    /// The argument `destination` must by a one-dimensional numpy array with
    /// dtype `uint32` and with the exact correct size. Use the method `num_words`
    /// to find out the correct size.
    ///
    /// Example:
    ///
    ///     coder = ans.Coder()
    ///     # ... push some symbols on coder ...
    ///     compressed_len = coder.num_words()
    ///     compressed = np.empty((compressed_len,), dtype=np.uint32)
    ///     coder.copy_compressed(compressed)
    ///
    ///     # Optional: write the compressed data to a file in
    ///     #           platform-independent byte ordering.
    ///     if sys.byteorder == 'big':
    ///         compressed.byteswap()
    ///     with open('path/to/file', 'wb') as file:
    ///         compressed.tofile(file)
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

    /// Encodes a sequence of symbols using Gaussian entropy models.
    ///
    /// The provided numpy arrays `symbols`, `means`, and `stds` must all have the
    /// same size.
    ///
    /// Arguments:
    /// symbols -- the symbols to be encoded. Must be a contiguous one-dimensional
    ///     numpy array (call `.copy()` on it if it is not contiguous) with dtype
    ///     `int32`. Each value in the array must be no smaller than
    ///     `min_supported_symbol` and no larger than `max_supported_symbol`.
    /// min_supported_symbol -- lower bound of the domain for argument `symbols`
    ///     (inclusively). Only relevant if `leaky` is `True`.
    /// max_supported_symbol -- upper bound of the domain for argument `symbols`
    ///     (inclusively). Only relevant if `leaky` is `True`.
    /// means -- the mean values of the Gaussian entropy models for each symbol.
    ///     Must be a contiguous one-dimensional numpy array with dtype `float64`
    ///     and with the exact same length as the argument `symbols`.
    /// stds -- the standard deviations of the Gaussian entropy models for each
    ///     symbol. Must be a contiguous one-dimensional numpy array with dtype
    ///     `float64` and with the exact same length as the argument `symbols`.
    ///     All entries must be strictly positive (i.e., nonzero and nonnegative)
    ///     and finite.
    /// leaky -- whether or not the entropy model should assign a nonzero
    ///     probability for all symbols in the range from `min_supported_symbol`
    ///     to `max_supported_symbol`, even if the nominal probability falls
    ///     below the smallest nonzero probability that can be resolved by the
    ///     entropy coder. This is usually a good idea as it makes sure that all
    ///     symbols within the supported domain can actually be encoded. When in
    ///     doubt, set this to `True`.
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

    /// Decodes a sequence of symbols *in reverse order* using Gaussian entropy
    /// models.
    ///
    /// The provided numpy arrays `means`, `stds`, and `symbols_out` must all have
    /// the same size. The provided `means` and `stds` (and `min_supported_symbol`
    /// and `max_supported_symbol` if `leaky` is `True`) must be the exact same
    /// values that were used for encoding. Even a tiny modification of these
    /// arguments can cause the coder to decode *completely* different symbols.
    ///
    /// The symbols will be popped off the stack and written to the target array in
    /// reverseorder so as to simplify usage, e.g.:
    ///
    ///     coder = ans.Coder()
    ///     symbols = np.array([2, 8, -5], dtype=np.int32)
    ///     decoded = np.empty((3,), dtype=np.int32)
    ///     means = np.array([0.1, 10.3, -3.2], dtype=np.float64)
    ///     stds = np.array([3.2, 1.3, 1.9], dtype=np.float64)
    ///
    ///     # Push symbols on the stack:
    ///     coder.push_gaussian_symbols(symbols, -10, 10, means, stds, True)
    ///
    ///     # Pop symbols off the stack in reverse order:
    ///     coder.pop_gaussian_symbols(-10, 10, means, stds, decoded, True)
    ///
    ///     # Check that the decoded symbols match the encoded ones.
    ///     assert np.all(symbols == decoded)
    ///     assert coder.is_empty()
    ///
    /// Arguments:
    /// min_supported_symbol -- lower bound of the domain for argument `symbols`
    ///     (inclusively). Only relevant if `leaky` is `True`.
    /// max_supported_symbol -- upper bound of the domain for argument `symbols`
    ///     (inclusively). Only relevant if `leaky` is `True`.
    /// means -- the mean values of the Gaussian entropy models for each symbol.
    ///     Must be a contiguous one-dimensional numpy array with dtype `float64`
    ///     and with the exact same length as the argument `symbols_out`.
    /// stds -- the standard deviations of the Gaussian entropy models for each
    ///     symbol. Must be a contiguous one-dimensional numpy array with dtype
    ///     `float64` and with the exact same length as the argument `symbols_out`.
    /// symbols -- the symbols to be encoded. Must be a contiguous one-dimensional
    ///     numpy array (call `.copy()` on it if it is not contiguous) with dtype
    ///     `int32`. Each value in the array must be no smaller than
    ///     `min_supported_symbol` and no larger than `max_supported_symbol`.
    /// leaky -- whether or not the entropy model should assign a nonzero
    ///     probability for all symbols in the range from `min_supported_symbol`
    ///     to `max_supported_symbol`, even if the nominal probability falls
    ///     below the smallest nonzero probability that can be resolved by the
    ///     entropy coder. This is usually a good idea as it makes sure that all
    ///     symbols within the supported domain can actually be encoded. When in
    ///     doubt, set this to `True`.
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

    /// Encodes a sequence of symbols using a fixed categorical distribution.
    ///
    /// This method is analogous to the method `push_gaussian_symbols` except that
    /// - all symbols are encoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian distribution.
    ///
    /// In detail, the categorical entropy model is constructed as follows:
    /// - each symbol from `min_supported_symbol` to `max_supported_symbol` gets
    ///   assigned at least the smallest nonzero probability that is representable
    ///   within the internally used precision.
    /// - the remaining probability mass is distributed among the symbols from
    ///   `min_provided_symbol` to `min_provided_symbol + len(probabilities) - 1`
    ///   (inclusively), proportionally to the provided probabilities.
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

    /// Encodes a sequence of categorically distributed symbols *in reverse order*.
    ///
    /// This method is analogous to the method `pop_gaussian_symbols` except that
    /// - all symbols are decoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian distribution.
    ///
    /// See documentation of `push_iid_categorical_symbols` for details of the
    /// categorical entropy model. See documentation of `pop_gaussian_symbols` for a
    /// discussion of the reverse order of decoding, and for a related usage
    /// example.
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
