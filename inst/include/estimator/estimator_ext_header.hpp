#include <string>
#include <locale>
#include <codecvt>
#include <stan/math.hpp>
#include <RcppEigen.h>

namespace internal {
  Rcpp::Function ll_fun("ls");
  Rcpp::Function grad_fun("ls");
}

enum boundsType {
  SINGLE = 1,
  BOTH = 2,
  NONE = 3
};

inline double single_b_step(double x, double lb, double hs) {
  return std::exp(hs) * (x - lb) + lb;
}

inline double both_b_step(double x, double lb, double ub, double hs) {
  return lb + (ub - lb) / (1 + (std::exp(-hs) * x - ub) / (lb - x));
}

inline double fdiff_step(int type, double x, double lb, double ub, double hs) {
  switch(type) {
    case SINGLE:
      return single_b_step(x, lb, hs);
    case BOTH:
      return both_b_step(x, lb, ub, hs);
    case NONE:
      return x + hs;
  }
  return stan::math::NOT_A_NUMBER;
}

template <typename F, typename T>
Eigen::VectorXd fdiff(const F& f, const T& x,
                            std::vector<int> cons_type,
                            Eigen::VectorXd lower,
                            Eigen::VectorXd upper) {
  static int h_scale[6] = {3, 2, 1, -3, -2, -1};
  static int mults[6] = {1, -9, 45, -1, 9, -45};
  Eigen::VectorXd x_temp(x);
  return Eigen::VectorXd::NullaryExpr(x.size(), [&f, &x, &x_temp, &cons_type, &lower, &upper](Eigen::Index i) {
    double h = stan::math::finite_diff_stepsize(x[i]);
    double delta_f = 0;
    for (int j = 0; j < 6; ++j) {
      x_temp[i] = fdiff_step(cons_type[i], x[i], lower[i], upper[i], h * h_scale[j]);
      delta_f += f(x_temp) * mults[j];
    }
    x_temp[i] = x[i];
    return delta_f / (60 * h * (cons_type[i] == 3 ? 1 : x[i]));
  });
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
  stan::arena_t<Eigen::VectorXd> arena_grad = fdiff(
        [&](const auto& x) { return Rcpp::as<double>(internal::ll_fun(x)); },
        v.val(), bounds_types, lower_bounds, upper_bounds);
    return make_callback_var(
      Rcpp::as<double>(internal::ll_fun(v.val())), [arena_v, arena_grad](auto& vi) mutable {
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
