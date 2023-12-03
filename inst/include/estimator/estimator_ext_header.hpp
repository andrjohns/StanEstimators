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
double r_function(const T& v, int finite_diff, std::ostream* pstream__) {

  return Rcpp::as<double>(internal::ll_fun(v));
}

template <typename T, stan::require_st_var<T>* = nullptr>
stan::math::var r_function(const T& v, int finite_diff, std::ostream* pstream__) {
  using stan::math::finite_diff_gradient_auto;
  using stan::math::make_callback_var;

  stan::arena_t<stan::plain_type_t<T>> arena_v = v;
  if (finite_diff == 1) {
    double ret;
    Eigen::VectorXd grad;
    finite_diff_gradient_auto(
      [&](const auto& x) { return Rcpp::as<double>(internal::ll_fun(x)); },
      v.val(), ret, grad);

    stan::arena_t<Eigen::VectorXd> arena_grad = grad;
    return make_callback_var(ret, [arena_v, arena_grad](auto& vi) mutable {
      arena_v.adj() += vi.adj() * arena_grad;
    });
  } else {
    return make_callback_var(
      Rcpp::as<double>(internal::ll_fun(v.val())),
      [arena_v](auto& vi) mutable {
        arena_v.adj() += vi.adj() * Rcpp::as<Eigen::Map<Eigen::VectorXd>>(internal::grad_fun(arena_v.val()));
    });
  }
}

/**
 * This macro is set by R headers included above and no longer needed
*/
#ifdef USING_R
#undef USING_R
#endif
