#ifndef STAN_MATH_PRIM_FUN_EIGENVALUES_SYM_HPP
#define STAN_MATH_PRIM_FUN_EIGENVALUES_SYM_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Return the eigenvalues of the specified symmetric matrix
 * in ascending order of magnitude.  This function is more
 * efficient than the general eigenvalues function for symmetric
 * matrices.
 *
 * @tparam EigMat type of the matrix
 * @param m Specified matrix.
 * @return Eigenvalues of matrix.
 */
template <typename EigMat, require_eigen_matrix_dynamic_t<EigMat>* = nullptr,
          require_not_st_var<EigMat>* = nullptr>
Eigen::Matrix<value_type_t<EigMat>, -1, 1> eigenvalues_sym(const EigMat& m) {
  if (unlikely(m.size() == 0)) {
    return Eigen::Matrix<value_type_t<EigMat>, -1, 1>(0, 1);
  }
  using PlainMat = plain_type_t<EigMat>;
  const PlainMat& m_eval = m;
  check_symmetric("eigenvalues_sym", "m", m_eval);

  Eigen::SelfAdjointEigenSolver<PlainMat> solver(m_eval,
                                                 Eigen::EigenvaluesOnly);
  return solver.eigenvalues();
}

}  // namespace math
}  // namespace stan

#endif
