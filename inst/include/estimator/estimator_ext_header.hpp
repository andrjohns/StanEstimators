#include <string>
#include <locale>
#include <codecvt>
#include <stan/math.hpp>
#include <RcppEigen.h>

namespace internal {
  Rcpp::Function ll_fun("ls");
  Rcpp::Function grad_fun("ls");
}

enum boundsType { LOWER = 1, UPPER = 2, BOTH = 3, NONE = 4 };
std::mutex m;

double call_ll(const Eigen::VectorXd& vals) {
  m.lock();
  double ret = Rcpp::as<double>(internal::ll_fun(vals));
  m.unlock();
  return ret;
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
    double scal = 0;
    switch (cons_type[i]) {
      case LOWER:
        scal = x[i] - lower[i];
        for (int j = 0; j < 6; ++j) {
          x_temp[i] = lower[i] + std::exp(h * h_scale[j]) * scal;
          delta_f += f(x_temp) * mults[j];
        }
        break;
      case UPPER:
        scal = x[i] - upper[i];
        for (int j = 0; j < 6; ++j) {
          x_temp[i] = upper[i] + std::exp(h * h_scale[j]) * scal;
          delta_f += f(x_temp) * mults[j];
        }
        break;
      case BOTH:
        scal = (x[i] - upper[i]) / (lower[i] - x[i]);
        for (int j = 0; j < 6; ++j) {
          x_temp[i] = 1 / (1 + std::exp(-h * h_scale[j]) * scal);
          delta_f += f(x_temp) * mults[j];
        }
        break;
      case NONE:
        for (int j = 0; j < 6; ++j) {
          x_temp[i] = x[i] + h * h_scale[j];
          delta_f += f(x_temp) * mults[j];
        }
        break;
    }
    x_temp[i] = x[i];
    return delta_f / (60 * h * (cons_type[i] == NONE ? 1 : x[i]));
  });
}

template <typename T, typename TLower, typename TUpper,
          stan::require_st_arithmetic<T>* = nullptr>
double r_function(const T& v, int finite_diff, int no_bounds,
                  std::vector<int> bounds_types,
                  const TLower& lower_bounds, const TUpper& upper_bounds,
                  std::ostream* pstream__) {
  return call_ll(v);
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
    stan::arena_t<Eigen::VectorXd> arena_grad =
      fdiff([&](const auto& x) { return call_ll(x); },
            v.val(), bounds_types, lower_bounds, upper_bounds);
    return make_callback_var(
      call_ll(v.val()),
      [arena_v, arena_grad](auto& vi) mutable {
        arena_v.adj() += vi.adj() * arena_grad;
      });
  } else {
    return make_callback_var(
      call_ll(v.val()),
      [arena_v](auto& vi) mutable {
        m.lock();
        Eigen::Map<Eigen::VectorXd> ret = Rcpp::as<Eigen::Map<Eigen::VectorXd>>(internal::grad_fun(arena_v.val()));
        m.unlock();
        arena_v.adj() += vi.adj() * ret;
    });
  }
}

/**
 * This macro is set by R headers included above and no longer needed
*/
#ifdef USING_R
#undef USING_R
#endif
