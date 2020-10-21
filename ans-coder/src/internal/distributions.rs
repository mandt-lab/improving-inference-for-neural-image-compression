use statrs::distribution::{InverseCDF, Univariate};
use std::marker::PhantomData;

pub trait DiscreteDistribution<S: Copy> {
    fn left_cumulative_and_probability(&self, symbol: S) -> (u32, u32);

    /// Returns (symbol, left_sided_cumulative, probability)
    fn quantile_function(&self, quantile: u32) -> (S, u32, u32);
}

/// Builder for [`LeakilyQuantizedDistribution`]
///
/// [`LeakilyQuantizedDistribution`]: struct.LeakilyQuantizedDistribution.html
pub struct Quantizer<L: Leakiness> {
    min_symbol: i32,
    max_symbol: i32,
    scale: f64,
    phantom: PhantomData<L>,
}

pub type NonLeakyQuantizer = Quantizer<NonLeaky>;
pub type LeakyQuantizer = Quantizer<Leaky>;

/// This is a hack that can be replaced by a boolean const generic once they're
/// stable. It should compile away just the same.
pub trait Leakiness {
    fn leakiness() -> u32;
}

pub struct NonLeaky;

impl Leakiness for NonLeaky {
    #[inline(always)]
    fn leakiness() -> u32 {
        0
    }
}

pub struct Leaky;

impl Leakiness for Leaky {
    #[inline(always)]
    fn leakiness() -> u32 {
        1
    }
}

impl<L: Leakiness> Quantizer<L> {
    pub fn new(min_symbol: i32, max_symbol: i32) -> Self {
        Self {
            min_symbol,
            max_symbol,
            scale: (1u32 << super::FREQUENCY_BITS)
                .wrapping_sub((max_symbol - min_symbol + 1) as u32) as f64,
            phantom: PhantomData::<L>,
        }
    }

    pub fn quantize<CD: Univariate<f64, f64> + InverseCDF<f64>>(
        &self,
        continuous_distribution: CD,
    ) -> QuantizedDistribution<L, CD> {
        QuantizedDistribution {
            inner: continuous_distribution,
            quantizer: self,
        }
    }
}

/// Wrapper that turns a [ContinuousDistribution] into a [DiscreteDistribution]
///
/// [ContinuousDistribution]: trait.ContinuousDistribution.html
/// [DiscreteDistribution]: trait.DiscreteDistribution.html
pub struct QuantizedDistribution<'a, L: Leakiness, CD: Univariate<f64, f64> + InverseCDF<f64>> {
    inner: CD,
    quantizer: &'a Quantizer<L>,
}

impl<'a, L: Leakiness, CD: Univariate<f64, f64> + InverseCDF<f64>> DiscreteDistribution<i32>
    for QuantizedDistribution<'a, L, CD>
{
    fn left_cumulative_and_probability(&self, symbol: i32) -> (u32, u32) {
        let min_symbol = self.quantizer.min_symbol;
        let max_symbol = self.quantizer.max_symbol;
        let scale = self.quantizer.scale;

        assert!(symbol >= min_symbol && symbol <= max_symbol);
        let leakiness = L::leakiness();
        let slack = leakiness * (symbol - min_symbol) as u32;

        // Round both cumulatives *independently* to fixed point precision.
        let left_sided_cumulative = if symbol == min_symbol {
            // Corner case: only makes a difference if we're cutting off a fairly significant
            // left tail of the distribution.
            0
        } else {
            (scale * self.inner.cdf(symbol as f64 - 0.5)) as u32 + slack
        };

        let right_sided_cumulative = if symbol == max_symbol {
            // Corner case: make sure that the probabilities add up to one. The generic
            // calculation in the `else` branch may lead to a lower total probability
            // because we're cutting off the right tail of the distribution and we're
            // rounding down.
            1 << super::FREQUENCY_BITS
        } else {
            (scale * self.inner.cdf(symbol as f64 + 0.5)) as u32 + slack + leakiness
        };

        (
            left_sided_cumulative,
            right_sided_cumulative.wrapping_sub(left_sided_cumulative),
        )
    }

    fn quantile_function(&self, quantile: u32) -> (i32, u32, u32) {
        let min_symbol = self.quantizer.min_symbol;
        let max_symbol = self.quantizer.max_symbol;
        let scale = self.quantizer.scale;
        let leakiness = L::leakiness();

        // Make an initial guess for the inverse of the leaky CDF.
        let mut symbol = self
            .inner
            .inverse_cdf((quantile as f64 + 0.5) / (1u64 << super::FREQUENCY_BITS) as f64)
            as i32;

        let mut left_sided_cumulative = if symbol <= min_symbol {
            // Corner case: we're in the left cut off tail of the distribution.
            symbol = min_symbol;
            0
        } else {
            if symbol > max_symbol {
                // Corner case: we're in the right cut off tail of the distribution.
                symbol = max_symbol;
            }

            (scale * self.inner.cdf(symbol as f64 - 0.5)) as u32
                + leakiness * (symbol - min_symbol) as u32
        };

        let right_sided_cumulative = if left_sided_cumulative > quantile {
            // Our initial guess for `symbol` was too high. Reduce it until we're good.
            symbol -= 1;
            let mut right_sided_cumulative = left_sided_cumulative;
            loop {
                if symbol == min_symbol {
                    left_sided_cumulative = 0;
                    break;
                }

                left_sided_cumulative = (scale * self.inner.cdf(symbol as f64 - 0.5)) as u32
                    + leakiness * (symbol - min_symbol) as u32;
                if left_sided_cumulative <= quantile {
                    break;
                } else {
                    right_sided_cumulative = left_sided_cumulative;
                    symbol -= 1;
                }
            }

            right_sided_cumulative
        } else {
            // Our initial guess for `symbol` was either exactly right or too low.
            // Check validity of the right sided cumulative. If it isn't valid,
            // keep increasing `symbol` until it is.
            loop {
                if symbol == max_symbol {
                    break 1 << super::FREQUENCY_BITS;
                }

                let right_sided_cumulative = ((scale * self.inner.cdf(symbol as f64 + 0.5)) as u32
                    + leakiness * (symbol - min_symbol) as u32)
                    .wrapping_add(1);
                if right_sided_cumulative > quantile {
                    break right_sided_cumulative;
                }

                left_sided_cumulative = right_sided_cumulative;
                symbol += 1;
            }
        };

        (
            symbol,
            left_sided_cumulative,
            right_sided_cumulative.wrapping_sub(left_sided_cumulative),
        )
    }
}

pub struct Categorical {
    min_symbol: i32,
    cdf: Box<[u32]>,
}

impl Categorical {
    pub fn new(min_symbol: i32, probabilities: &[u32]) -> Self {
        let cdf = std::iter::once(&0)
            .chain(probabilities)
            .scan(0, |accum, prob| {
                *accum += *prob;
                Some(*accum)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        assert_eq!(cdf.last(), Some(&(1 << super::FREQUENCY_BITS)));
        if super::FREQUENCY_BITS == 32 {
            // `cdf[cdf.len() - 2]` is guaranteed to be within bounds since `cdf.len() >= 2`
            // by the above assertion and the fact that `cdf[0] == 0` by construction.
            assert!(cdf[cdf.len() - 2] != (1 << super::FREQUENCY_BITS))
        }

        Self { min_symbol, cdf }
    }

    pub fn from_continuous_probabilities(
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        min_provided_symbol: i32,
        probabilities: &[f64],
    ) -> Self {
        assert!(min_supported_symbol <= min_provided_symbol);
        assert!(min_provided_symbol + probabilities.len() as i32 - 1 <= max_supported_symbol);

        let probabilities = optimal_weights(
            probabilities,
            (min_provided_symbol - min_supported_symbol) as u32,
            (max_supported_symbol - min_provided_symbol + 1) as u32 - probabilities.len() as u32,
        );
        Self::new(min_supported_symbol, &probabilities)
    }

    /// Returns the entropy in units of bits (i.e., base 2).
    pub fn entropy(&self) -> f64 {
        let entropy_scaled = self
            .cdf
            .iter()
            .skip(1)
            .scan(0, |last, &cdf| {
                let prob = cdf.wrapping_sub(*last) as f64;
                *last = cdf;
                Some(prob * prob.log2())
            })
            .sum::<f64>();

        entropy_scaled / (1u64 << super::FREQUENCY_BITS) as f64 - super::FREQUENCY_BITS as f64
    }
}

impl DiscreteDistribution<i32> for Categorical {
    fn left_cumulative_and_probability(&self, symbol: i32) -> (u32, u32) {
        let index = symbol - self.min_symbol;

        let (cdf, next_cdf) = unsafe {
            // SAFETY: the assertion ensures we're not out of bounds.
            assert!(index >= 0 && index as usize + 1 < self.cdf.len());
            (
                *self.cdf.get_unchecked(index as usize),
                *self.cdf.get_unchecked(index as usize + 1),
            )
        };

        (cdf, next_cdf.wrapping_sub(cdf))
    }

    fn quantile_function(&self, quantile: u32) -> (i32, u32, u32) {
        if super::FREQUENCY_BITS != 32 {
            assert!(quantile < (1 << super::FREQUENCY_BITS));
        }

        let mut left = 0; // Smallest possible index.
        let mut right = self.cdf.len() - 1; // One above largest possible index.

        // Binary search for the last entry of `self.cdf` that is <= quantile,
        // exploiting the fact that `self.cdf[0] == 0` and
        // `*self.cdf.last().unwrap() == (1 << super::FREQUENCY_BITS) - 1`.
        while left + 1 != right {
            let mid = (left + right) / 2;

            // SAFETY: the loop maintains the invariants
            // `0 <= left <= mid < right < self.cdf.len()` and
            // `cdf[left] <= cdf[mid] <= cdf[right]`.
            let pivot = unsafe { *self.cdf.get_unchecked(mid) };
            if pivot <= quantile {
                left = mid;
            } else {
                right = mid;
            }
        }

        // SAFETY: invariant `0 <=left < right < self.cdf.len()` still holds.
        let cdf = unsafe { *self.cdf.get_unchecked(left) };
        let next_cdf = unsafe { *self.cdf.get_unchecked(right) };

        (
            self.min_symbol + left as i32,
            cdf,
            next_cdf.wrapping_sub(cdf),
        )
    }
}

fn optimal_weights(pmf: &[f64], padding_left: u32, padding_right: u32) -> Vec<u32> {
    assert!(pmf.len() >= 2);
    let max_weight = (1 << super::FREQUENCY_BITS) - 1;

    // Start by assigning each symbol weight 1 and then distributing no more than
    // the remaining weight approximately evenly across all symbols.
    let mut remaining_weight =
        (1u32 << super::FREQUENCY_BITS).wrapping_sub(padding_left + padding_right);
    let free_weight = remaining_weight.wrapping_sub(pmf.len() as u32);
    let scale = free_weight as f64 / pmf.iter().sum::<f64>();

    let mut indices_probs_weights_wins_losses = pmf
        .iter()
        .enumerate()
        .map(|(index, &prob)| {
            let weight = 1 + (prob * scale) as u32;
            remaining_weight = remaining_weight.wrapping_sub(weight);

            // How much the cross entropy would decrease when increasing the weight by one.
            let win = if weight == max_weight {
                std::f64::NEG_INFINITY
            } else {
                prob * (1.0 / weight as f64).ln_1p()
            };

            // How much the cross entropy would increase when decreasing the weight by one.
            let loss = if weight == 1 {
                std::f64::INFINITY
            } else {
                -prob * (-1.0 / weight as f64).ln_1p()
            };

            (index, prob, weight, win, loss)
        })
        .collect::<Vec<_>>();

    // Distribute remaining weight evenly among symbols with highest wins.
    while remaining_weight != 0 {
        indices_probs_weights_wins_losses
            .sort_by(|&(_, _, _, win1, _), &(_, _, _, win2, _)| win2.partial_cmp(&win1).unwrap());
        let batch = std::cmp::min(
            remaining_weight as usize,
            indices_probs_weights_wins_losses.len(),
        );
        for (_, prob, weight, win, loss) in &mut indices_probs_weights_wins_losses[..batch] {
            *weight += 1; // Cannot end up in `max_weight` because win would otherwise be -infinity.
            *win = if *weight == max_weight {
                std::f64::NEG_INFINITY
            } else {
                *prob * (1.0 / *weight as f64).ln_1p()
            };
            *loss = -*prob * (-1.0 / *weight as f64).ln_1p();
        }
        remaining_weight -= batch as u32;
    }

    loop {
        // Find element where increasing weight would incur the biggest win.
        let (buyer_index, &(_, _, _, buyer_win, _)) = indices_probs_weights_wins_losses
            .iter()
            .enumerate()
            .max_by(|(_, (_, _, _, win1, _)), (_, (_, _, _, win2, _))| {
                win1.partial_cmp(win2).unwrap()
            })
            .unwrap();
        let (seller_index, (_, seller_prob, seller_weight, seller_win, seller_loss)) =
            indices_probs_weights_wins_losses
                .iter_mut()
                .enumerate()
                .min_by(|(_, (_, _, _, _, loss1)), (_, (_, _, _, _, loss2))| {
                    loss1.partial_cmp(loss2).unwrap()
                })
                .unwrap();

        if buyer_index == seller_index {
            // This can only happen due to rounding errors. In this case, we can't expect
            // to be able to improve further.
            break;
        }

        if buyer_win <= *seller_loss {
            // We've found the optimal solution.
            break;
        }

        *seller_weight -= 1;
        *seller_win = *seller_prob * (1.0 / *seller_weight as f64).ln_1p();
        *seller_loss = if *seller_weight == 1 {
            std::f64::INFINITY
        } else {
            -*seller_prob * (-1.0 / *seller_weight as f64).ln_1p()
        };

        let (_, buyer_prob, buyer_weight, buyer_win, buyer_loss) =
            &mut indices_probs_weights_wins_losses[buyer_index];
        *buyer_weight += 1;
        *buyer_win = if *buyer_weight == max_weight {
            std::f64::NEG_INFINITY
        } else {
            *buyer_prob * (1.0 / *buyer_weight as f64).ln_1p()
        };
        *buyer_loss = -*buyer_prob * (-1.0 / *buyer_weight as f64).ln_1p();
    }

    indices_probs_weights_wins_losses.sort_by_key(|&(index, _, _, _, _)| index);

    std::iter::repeat(1)
        .take(padding_left as usize)
        .chain(
            indices_probs_weights_wins_losses
                .into_iter()
                .map(|(_, _, weight, _, _)| weight),
        )
        .chain(std::iter::repeat(1).take(padding_right as usize))
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    use statrs::distribution::Normal;

    #[test]
    fn leaky_quantized_normal() {
        let quantizer = LeakyQuantizer::new(-127, 127);

        for &std_dev in &[0.0001, 0.1, 3.5, 123.45, 1234.56] {
            for &mean in &[-300.6, -100.2, -5.2, 0.0, 50.3, 180.2, 2000.0] {
                let continuous_distribution = Normal::new(mean, std_dev).unwrap();
                test_discrete_distribution(quantizer.quantize(continuous_distribution));
            }
        }
    }

    /// Test that `optimal_weights` reproduces the same distribution when fed with an
    /// already quantized distribution.
    #[test]
    fn trivial_optimal_weights() {
        let hist = [
            526797u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772,
            1657269, 896675, 922197, 930672, 916665, 1, 1, 889553, 846665, 789559, 723031, 650522,
            572300, 494702, 418703, 347600, 1, 283500, 226158, 178194, 136301, 103158, 76823,
            55540, 39258, 27988, 54269,
        ];
        assert_eq!(
            hist.iter().sum::<u32>(),
            (1 << super::super::FREQUENCY_BITS) - 255 + hist.len() as u32
        );

        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();

        let weights = optimal_weights(&probabilities, 100, 255 - 100 - hist.len() as u32);

        assert_eq!(weights.len(), 255);
        assert_eq!(&weights[..100], vec![1; 100]);
        assert_eq!(
            &weights[100 + hist.len()..],
            vec![1; 255 - 100 - hist.len()]
        );
        assert_eq!(&weights[100..100 + hist.len()], &hist[..]);
    }

    #[test]
    fn nontrivial_optimal_weights() {
        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        assert!(
            hist.iter().sum::<u32>()
                != (1 << super::super::FREQUENCY_BITS) - 255 + hist.len() as u32
        );

        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();

        let weights = optimal_weights(&probabilities, 100, 255 - 100 - hist.len() as u32);
        assert_eq!(
            weights.iter().sum::<u32>(),
            1 << super::super::FREQUENCY_BITS
        );

        assert_eq!(weights.len(), 255);
        assert_eq!(&weights[..100], vec![1; 100]);
        assert_eq!(
            &weights[100 + hist.len()..],
            vec![1; 255 - 100 - hist.len()]
        );

        let mut weights_and_hist = weights[100..]
            .iter()
            .cloned()
            .zip(hist[..].iter().cloned())
            .collect::<Vec<_>>();

        // Check that sorting by weight is compatible with sorting by hist.
        weights_and_hist.sort();
        // TODO: replace the following with
        // `assert!(weights_and_hist.iter().map(|&(_, x)| x).is_sorted())`
        // when `is_sorted` becomes stable.
        let mut last = 0;
        for (_, hist) in weights_and_hist {
            assert!(hist >= last);
            last = hist;
        }
    }

    #[test]
    fn categorical() {
        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();

        let distribution =
            Categorical::from_continuous_probabilities(-127, 127, -10, &probabilities);
        test_discrete_distribution(distribution);
    }

    fn test_discrete_distribution(distribution: impl DiscreteDistribution<i32>) {
        let mut sum = 0;

        for symbol in -127..128 {
            let (left_cumulative, prob) = distribution.left_cumulative_and_probability(symbol);
            assert_eq!(left_cumulative as u64, sum);
            assert!(prob > 0);
            sum += prob as u64;

            let expected = (symbol, left_cumulative, prob);
            assert_eq!(distribution.quantile_function(left_cumulative), expected);
            assert_eq!(distribution.quantile_function((sum - 1) as u32), expected);
            assert_eq!(
                distribution.quantile_function(left_cumulative + prob / 2),
                expected
            );
        }

        assert_eq!(sum, 1 << super::super::FREQUENCY_BITS);
    }
}
