#include <string>
#include <locale>
#include <codecvt>
#include <stan/math.hpp>
#include <RcppEigen.h>

namespace internal {
  Rcpp::Function ll_fun("ls");
  Rcpp::Function grad_fun("ls");
}

template <typename T, stan::require_st_arithmetic<T>* = nullptr>
double r_function(const T& v, int finite_diff, std::ostream *pstream__) {
  return Rcpp::as<double>(internal::ll_fun(v));
}

template <typename T, stan::require_st_var<T>* = nullptr>
stan::math::var r_function(const T& v, int finite_diff, std::ostream *pstream__) {
  using stan::math::finite_diff_gradient_auto;
  using stan::math::make_callback_var;

  stan::arena_t<stan::plain_type_t<T>> arena_v = v;
  stan::arena_t<Eigen::VectorXd> arena_grad;
  double ret;
  Eigen::VectorXd grad;
  if (finite_diff == 1) {
    finite_diff_gradient_auto(
      [&](const auto& x) { return Rcpp::as<double>(internal::ll_fun(x)); }, v.val(), ret, grad);
      arena_grad = grad;
  } else {
    ret = Rcpp::as<double>(internal::ll_fun(v.val()));
    arena_grad = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(internal::grad_fun(v.val()));
  }

  return make_callback_var(
    ret, [arena_v, arena_grad](auto& vi) mutable { arena_v.adj() += vi.adj() * arena_grad; });
}

/**
 * This macro is set by R headers included above and no longer needed
*/
#ifdef USING_R
#undef USING_R
#endif
