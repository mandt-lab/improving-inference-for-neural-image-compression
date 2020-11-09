# ANS Entropy Coder

The code in directory implements a simple entropy coder based on [Asymmetric Numeral Systems](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems) as a Python extension module.

## Usage

Once the extension is [compiled (see below)](#compilation), using it from Python is very simple:

```python
import numpy as np
import ans

# Create some toy data and entropy models.
gaussian_symbols = np.array([2, 8, -5], dtype=np.int32)
gaussian_means = np.array([0.1, 10.3, -3.2], dtype=np.float64)
gaussian_stds = np.array([3.2, 1.3, 1.9], dtype=np.float64)

categorical_symbols = np.array([3, -2, -7, 8], dtype=np.int32)
# (Categorical probabilities don't need to be normalized.)
categorical_probabilities = np.arange(1, 22, dtype=np.float64)

# Construct an entropy coder with no initial data.
coder1 = ans.Coder()

# Encode some data ("push it on the stack").
coder1.push_gaussian_symbols(
    gaussian_symbols, -10, 10, gaussian_means, gaussian_stds, True)
coder1.push_iid_categorical_symbols(
    categorical_symbols, -10, 10, -10, categorical_probabilities)

# Allocate an array for the compressed data and copy it out.
compressed = np.empty((coder1.num_words(),), dtype=np.uint32)
coder1.copy_compressed(compressed)

# Construct a new entropy coder from the compressed data. (We could also just
# continue using `coder1` but we're pretending here that the compressed data
# was read from a file and that we no longer have access to `coder1`.)
coder2 = ans.Coder(compressed)

# Allocate space for the decoded symbols.
decoded_categorical_symbols = np.empty((4,), dtype=np.int32)
decoded_gaussian_symbols = np.empty((3,), dtype=np.int32)

# Decode the symbols *in reverse order* because the ANS coder is a stack.
coder2.pop_iid_categorical_symbols(
    -10, 10, -10, categorical_probabilities, decoded_categorical_symbols)
coder2.pop_gaussian_symbols(
    -10, 10, gaussian_means, gaussian_stds, decoded_gaussian_symbols, True)

# Verify that the decoded data matches the encoded data.
assert np.all(gaussian_symbols == decoded_gaussian_symbols)
assert np.all(categorical_symbols == decoded_categorical_symbols)
assert coder2.is_empty()
```

## Compilation

The entropy coder is implemented as a Python extension module written in the [Rust programming language](https://www.rust-lang.org) for runtime efficiency.
Follow these steps to compile the module:

1. Install a rust toolchain if it's not already installed on your system:
    <https://rustup.rs> (it's usually just a one-line command)

2. Compile the module (in `--release` mode, i.e., with optimizations turned on):

    ```bash
    cd ans-coder
    cargo build --release
    ```

    If you run this for the first time then it will take a few minutes because it downloads and compiles all the dependencies.
    Subsequent compilations (if you changed anything) will be much quicker.

3. Create a symlink to the compiled library in the parent directory:

    ```bash
    cd ..
    ln -s ans-coder/target/release/libans.so ans.so
    ```

4. Open a Python REPL and paste the [above example code](#usage) to verify that it works.

## Known Issues

- The Python/Rust interface includes some copying that isn't strictly necessary but it makes the implementation a bit simpler.
  At the moemnt, this is not an issue since entropy coding is not the computational bottleneck in current our use cases.
- It would also be nicer if, instead of passing an out-parameter to `copy_compressed`, the method could just instantiate a new numpy array, write to it, and then return it (passing ownership to Python's garbage collector).
