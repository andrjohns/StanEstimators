#include <string>
#include <locale>
#include <codecvt>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/log1p_exp.hpp>
#include <stan/math/prim/fun/inv_logit.hpp>
#include <stan/math/prim/fun/finite_diff_stepsize.hpp>
#include <stan/math/prim/fun/lub_free.hpp>
#include <stan/math/prim/fun/lub_constrain.hpp>
#include <stan/math/prim/functor/apply_scalar_ternary.hpp>
#include <stan/math/prim/functor/finite_diff_gradient_auto.hpp>
#include <stan/math/rev/core.hpp>
#include <RcppEigen.h>

namespace internal {
  Rcpp::Function ll_fun("ls");
  Rcpp::Function grad_fun("ls");
}

template <typename F, typename T>
Eigen::VectorXd fdiff(const F& f, const T& x) {
  static int h_scale[6] = {3, 2, 1, -3, -2, -1};
  static int mults[6] = {1, -9, 45, -1, 9, -45};
  Eigen::VectorXd x_temp(x);
  return Eigen::VectorXd::NullaryExpr(x.size(), [&f, &x, &x_temp](Eigen::Index i) {
    double h = stan::math::finite_diff_stepsize(x[i]);
    double delta_f = 0;
    for (int j = 0; j < 6; ++j) {
      x_temp[i] = x[i] + h * h_scale[j];
      delta_f += f(x_temp) * mults[j];
    }
    x_temp[i] = x[i];
    return delta_f / (60 * h);
  });
}

template <bool jacobian__, typename T, stan::require_st_arithmetic<T>* = nullptr>
double r_function(const T& v,
                  int finite_diff, int no_bounds,
                  std::vector<int> bounds_types,
                  const Eigen::Map<Eigen::VectorXd>& lower_bounds,
                  const Eigen::Map<Eigen::VectorXd>& upper_bounds,
                  std::ostream* pstream__) {
  if (jacobian__) {
    double lp = 0;
    auto v_cons = stan::math::lub_constrain(v, lower_bounds, upper_bounds, lp);
    return Rcpp::as<double>(internal::ll_fun(v_cons)) + lp;
  } else {
    auto v_cons = stan::math::lub_constrain(v, lower_bounds, upper_bounds);
    return Rcpp::as<double>(internal::ll_fun(v_cons));
  }
}

template <bool jacobian__, typename T, stan::require_st_var<T>* = nullptr>
stan::math::var r_function(const T& v,
                  int finite_diff, int no_bounds,
                  std::vector<int> bounds_types,
                  const Eigen::Map<Eigen::VectorXd>& lower_bounds,
                  const Eigen::Map<Eigen::VectorXd>& upper_bounds,
                  std::ostream* pstream__) {
  using stan::math::finite_diff_gradient_auto;
  using stan::math::make_callback_var;

  auto funwrap = [&](const auto& x) {
    return r_function<jacobian__>(x, finite_diff, no_bounds, bounds_types, lower_bounds, upper_bounds, pstream__);
  };
  stan::arena_t<Eigen::Matrix<stan::math::var, -1, 1>> arena_v = v;
  stan::arena_t<Eigen::VectorXd> arena_grad;
  if (finite_diff) {
    arena_grad = fdiff(funwrap, arena_v.val());
  } else {
    Eigen::VectorXd v_cons = stan::math::lub_constrain(arena_v.val(), lower_bounds, upper_bounds);
    arena_grad = Rcpp::as<Eigen::VectorXd>(internal::grad_fun(v_cons));
  }
  return make_callback_var(
    funwrap(v.val()),
    [arena_v, arena_grad](auto& vi) mutable {
      arena_v.adj() += vi.adj() * arena_grad;
  });
}

#ifdef USING_R
#undef USING_R
#endif
