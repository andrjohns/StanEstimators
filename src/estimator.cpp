#include <headers_to_ignore.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/rev/core/var.hpp>

double r_function(const Eigen::VectorXd& v,
                  int finite_diff, int no_bounds,
                  std::vector<int> bounds_types,
                  const Eigen::Map<Eigen::VectorXd>& lower_bounds,
                  const Eigen::Map<Eigen::VectorXd>& upper_bounds,
                  std::ostream* pstream__);

stan::math::var r_function(const Eigen::Matrix<stan::math::var, -1, 1>& v,
                  int finite_diff, int no_bounds,
                  std::vector<int> bounds_types,
                  const Eigen::Map<Eigen::VectorXd>& lower_bounds,
                  const Eigen::Map<Eigen::VectorXd>& upper_bounds,
                  std::ostream* pstream__);

#include <estimator/estimator.hpp>
