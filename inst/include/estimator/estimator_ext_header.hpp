#include <string>
#include <locale>
#include <codecvt>
#include <stan/math.hpp>
#include <RcppEigen.h>

namespace internal {
  Rcpp::Function ll_fun("ls");
  Rcpp::Function grad_fun("ls");
}

void store_functions(SEXP ll_fun, SEXP grad_fun) {
  internal::ll_fun = Rcpp::Function(ll_fun);
  internal::grad_fun = Rcpp::Function(grad_fun);
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

double constrain_grads_impl(double x, double lb_val, double ub_val) {
  const bool is_lb_inf = lb_val == stan::math::NEGATIVE_INFTY;
  const bool is_ub_inf = ub_val == stan::math::INFTY;
  if (unlikely(is_ub_inf && is_lb_inf)) {
    return 1.0;
  } else if (unlikely(is_ub_inf)) {
    return 1.0 / (std::exp(x));
  } else if (unlikely(is_lb_inf)) {
    return - 1.0 / (std::exp(x));
  } else {
    double diff = ub_val - lb_val;
    double inv_logit_x = stan::math::inv_logit(x);
    return 1.0 / (diff * inv_logit_x * (1.0 - inv_logit_x));
  }
}

template <typename T, typename TLower, typename TUpper>
Eigen::VectorXd constrain_grads(const T& v, const TLower& lower_bounds, const TUpper& upper_bounds) {
  return stan::math::apply_scalar_ternary(
    [](const auto& x, const auto& lb, const auto& ub) {
      return constrain_grads_impl(x, lb, ub);
    }, v, lower_bounds, upper_bounds
  );
}

template <typename T, typename TLower, typename TUpper,
          stan::require_st_arithmetic<T>* = nullptr>
double r_function(const T& v, int finite_diff, int no_bounds,
                  std::vector<int> bounds_types,
                  const TLower& lower_bounds, const TUpper& upper_bounds,
                  std::ostream* pstream__) {
  return Rcpp::as<double>(internal::ll_fun(v));
}

template <typename T, typename TLower, typename TUpper,
          stan::require_st_var<T>* = nullptr>
stan::math::var r_function(const T& v, int finite_diff, int no_bounds,
                  std::vector<int> bounds_types,
                  const TLower& lower_bounds, const TUpper& upper_bounds,
                  std::ostream* pstream__) {
  using stan::math::finite_diff_gradient_auto;
  using stan::math::make_callback_var;

  stan::arena_t<stan::plain_type_t<T>> arena_v = v;
  if (finite_diff == 1) {
    double ret;
    Eigen::VectorXd grad;
    stan::arena_t<Eigen::VectorXd> arena_grad;
    if (no_bounds == 1) {
      grad = fdiff(
        [&](const auto& x) {
          return Rcpp::as<double>(internal::ll_fun(x));
        }, v.val());

      arena_grad = grad;
    } else {
      Eigen::VectorXd unconstrained = stan::math::lub_free(v.val(), lower_bounds, upper_bounds);
      grad = fdiff(
        [&](const auto& x) {
          return Rcpp::as<double>(internal::ll_fun(stan::math::lub_constrain(x, lower_bounds, upper_bounds)));
        }, unconstrained);

      arena_grad = grad.cwiseProduct(constrain_grads(unconstrained, lower_bounds, upper_bounds));
    }
    return make_callback_var(Rcpp::as<double>(internal::ll_fun(v.val())), [arena_v, arena_grad](auto& vi) mutable {
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
