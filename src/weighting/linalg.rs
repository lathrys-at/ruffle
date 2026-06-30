//! A tiny symmetric-positive-definite solver for the coupling matrix (§5.4).
//!
//! Coupling assembles a regularized dimensionless covariance and needs its inverse,
//! or row-sums of its inverse, at `N` of a few channels (never more than ~16). That
//! is far too small to justify a linear-algebra dependency, so this hand-rolls a
//! Cholesky factorization: `A = L Lᵀ` with `L` lower-triangular. The factorization
//! exists iff `A` is symmetric positive-definite, so a `None` return doubles as the
//! positive-definiteness check the shrinkage step relies on.
//!
//! Matrices are `&[Vec<f64>]`, row-major and square. A non-square or ragged input,
//! or a matrix that is not positive-definite, returns `None`.

// Index-based loops are the clearest form for triangular matrix arithmetic, where a
// single index addresses several different rows.
#![allow(clippy::needless_range_loop)]

/// Number of rows, validated to be square. Returns `None` for a ragged matrix.
fn square_dim(a: &[Vec<f64>]) -> Option<usize> {
    let n = a.len();
    if a.iter().all(|row| row.len() == n) {
        Some(n)
    } else {
        None
    }
}

/// Cholesky factorization: the lower-triangular `L` with `L Lᵀ = A`.
///
/// Returns `None` when `A` is not square, contains a non-finite entry, or is not
/// positive-definite (a non-positive pivot appears). The upper triangle of the input
/// is ignored; `A` is treated as symmetric via its lower triangle.
pub fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = square_dim(a)?;
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            if !sum.is_finite() {
                return None;
            }
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                let pivot = l[j][j];
                if pivot == 0.0 || !pivot.is_finite() {
                    return None;
                }
                l[i][j] = sum / pivot;
            }
        }
    }
    Some(l)
}

/// Solve `L y = b` for `y`, with `L` lower-triangular (forward substitution).
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    y
}

/// Solve `Lᵀ x = y` for `x`, with `L` lower-triangular (back substitution).
fn back_substitution(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// Solve `A x = b` for symmetric-positive-definite `A`.
///
/// Returns `None` when `A` is not SPD (or not square) or when `b`'s length does not
/// match the dimension of `A`.
pub fn solve_spd(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = square_dim(a)?;
    if b.len() != n {
        return None;
    }
    let l = cholesky(a)?;
    let y = forward_substitution(&l, b);
    Some(back_substitution(&l, &y))
}

/// The inverse of a symmetric-positive-definite `A`.
///
/// Solves `A X = I` column by column through the shared Cholesky factor. Returns
/// `None` when `A` is not SPD or not square.
pub fn inverse_spd(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = square_dim(a)?;
    let l = cholesky(a)?;
    // Solve for each column of the inverse against a unit basis vector.
    let mut cols = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let mut e = vec![0.0f64; n];
        e[j] = 1.0;
        let y = forward_substitution(&l, &e);
        let x = back_substitution(&l, &y);
        for (i, xi) in x.into_iter().enumerate() {
            cols[i][j] = xi;
        }
    }
    Some(cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = b[0].len();
        let p = b.len();
        let mut c = vec![vec![0.0f64; m]; n];
        for (i, ci) in c.iter_mut().enumerate() {
            for (j, cij) in ci.iter_mut().enumerate() {
                let mut s = 0.0;
                for k in 0..p {
                    s += a[i][k] * b[k][j];
                }
                *cij = s;
            }
        }
        c
    }

    fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = a[0].len();
        let mut t = vec![vec![0.0f64; n]; m];
        for (i, row) in a.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                t[j][i] = v;
            }
        }
        t
    }

    /// A random SPD matrix: `MᵀM + εI` is symmetric positive-definite for any `M`.
    fn random_spd(n: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<f64>> {
        let m: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect())
            .collect();
        let mut a = matmul(&transpose(&m), &m);
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += 1e-3;
        }
        a
    }

    fn assert_identity(a: &[Vec<f64>], eps: f64) {
        for (i, row) in a.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let target = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(v, target, epsilon = eps);
            }
        }
    }

    #[test]
    fn cholesky_reconstructs_input() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for n in 1..=6 {
            let a = random_spd(n, &mut rng);
            let l = cholesky(&a).unwrap();
            let recon = matmul(&l, &transpose(&l));
            for (i, row) in a.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    assert_abs_diff_eq!(recon[i][j], v, epsilon = 1e-9);
                }
            }
        }
    }

    #[test]
    fn identity_inverse_is_identity() {
        for n in 1..=5 {
            let id: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect();
            let inv = inverse_spd(&id).unwrap();
            assert_identity(&inv, 1e-12);
        }
    }

    #[test]
    fn inverse_round_trip_is_identity() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for n in 1..=8 {
            let a = random_spd(n, &mut rng);
            let inv = inverse_spd(&a).unwrap();
            let prod = matmul(&a, &inv);
            assert_identity(&prod, 1e-6);
        }
    }

    #[test]
    fn solve_matches_inverse_times_b() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        for n in 1..=6 {
            let a = random_spd(n, &mut rng);
            let b: Vec<f64> = (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect();
            let x = solve_spd(&a, &b).unwrap();
            let inv = inverse_spd(&a).unwrap();
            // inv * b
            let xb: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| inv[i][j] * b[j]).sum())
                .collect();
            for i in 0..n {
                assert_abs_diff_eq!(x[i], xb[i], epsilon = 1e-7);
            }
            // And A x ≈ b.
            let ax: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| a[i][j] * x[j]).sum())
                .collect();
            for i in 0..n {
                assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn one_by_one() {
        let a = vec![vec![4.0]];
        assert_abs_diff_eq!(cholesky(&a).unwrap()[0][0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(inverse_spd(&a).unwrap()[0][0], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(solve_spd(&a, &[8.0]).unwrap()[0], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn two_by_two_hand_case() {
        // A = [[4, 2], [2, 3]]; det = 8; inv = (1/8)[[3, -2], [-2, 4]].
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let inv = inverse_spd(&a).unwrap();
        assert_abs_diff_eq!(inv[0][0], 3.0 / 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(inv[0][1], -2.0 / 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(inv[1][0], -2.0 / 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(inv[1][1], 4.0 / 8.0, epsilon = 1e-12);
    }

    #[test]
    fn three_by_three_diagonal() {
        let a = vec![
            vec![9.0, 0.0, 0.0],
            vec![0.0, 16.0, 0.0],
            vec![0.0, 0.0, 25.0],
        ];
        let l = cholesky(&a).unwrap();
        assert_abs_diff_eq!(l[0][0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(l[1][1], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(l[2][2], 5.0, epsilon = 1e-12);
        let x = solve_spd(&a, &[9.0, 32.0, 75.0]).unwrap();
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn non_pd_returns_none() {
        // Negative diagonal.
        assert_eq!(cholesky(&[vec![-1.0]]), None);
        // Zero matrix is positive-semidefinite, not positive-definite.
        assert_eq!(cholesky(&[vec![0.0, 0.0], vec![0.0, 0.0]]), None);
        // Indefinite symmetric matrix [[1,2],[2,1]] (eigenvalues 3, -1).
        assert_eq!(cholesky(&[vec![1.0, 2.0], vec![2.0, 1.0]]), None);
        assert_eq!(inverse_spd(&[vec![1.0, 2.0], vec![2.0, 1.0]]), None);
        assert_eq!(
            solve_spd(&[vec![1.0, 2.0], vec![2.0, 1.0]], &[1.0, 1.0]),
            None
        );
    }

    #[test]
    fn ragged_or_mismatched_returns_none() {
        assert_eq!(cholesky(&[vec![1.0, 0.0], vec![0.0]]), None);
        let a = vec![vec![4.0, 0.0], vec![0.0, 4.0]];
        assert_eq!(solve_spd(&a, &[1.0]), None);
    }

    #[test]
    fn non_finite_returns_none() {
        assert_eq!(cholesky(&[vec![f64::NAN]]), None);
        assert_eq!(cholesky(&[vec![f64::INFINITY, 0.0], vec![0.0, 1.0]]), None);
    }

    #[test]
    fn solve_dense_4x4_with_known_solution() {
        // A dense, symmetric, diagonally-dominant (hence SPD) 4x4 with a hand-chosen
        // solution x = [1,2,3,4]; b = A x. Solving exercises the full upper triangle of
        // back-substitution (every off-diagonal `l[k][i]` term participates), unlike the
        // diagonal-only cases above.
        let a = vec![
            vec![10.0, 1.0, 2.0, 0.0],
            vec![1.0, 12.0, 1.0, 3.0],
            vec![2.0, 1.0, 15.0, 1.0],
            vec![0.0, 3.0, 1.0, 20.0],
        ];
        let b = [18.0, 40.0, 53.0, 89.0]; // = a * [1,2,3,4]
        let x = solve_spd(&a, &b).unwrap();
        for (i, &xi) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert_abs_diff_eq!(x[i], xi, epsilon = 1e-9);
        }
    }

    // Two survivors in this module are genuinely EQUIVALENT mutants, kept undocumented as
    // tests because no input can distinguish them:
    //
    //  - cholesky's off-diagonal pivot guard `pivot == 0.0 || !pivot.is_finite()`
    //    (`||` -> `&&`). The pivot is `l[j][j]`, computed at the diagonal step of an
    //    earlier row, which already returns `None` unless its `sum` is finite and strictly
    //    positive. So a used pivot is always `sqrt(finite positive) > 0` and finite, and
    //    this guard never fires under either operator.
    //
    //  - back_substitution's loop bound `(i + 1)..n` (`+` -> `*`, i.e. `(i * 1)..n =
    //    i..n`). The extra `k == i` iteration subtracts `l[i][i] * x[i]`, but `x[i]` is
    //    still its initial `0.0` at that point (it is assigned only after the loop), so the
    //    term is exactly `0.0` and the result is bit-identical.
}
