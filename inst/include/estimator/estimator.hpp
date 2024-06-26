// Code generated by stanc v2.35.0
#include <stan/model/model_header.hpp>
namespace estimator_model_namespace {
using stan::model::model_base_crtp;
using namespace stan::math;
stan::math::profile_map profiles__;
static constexpr std::array<const char*, 13> locations_array__ =
  {" (found before start of program)",
  " (in 'inst/include/estimator/estimator.stan', line 15, column 2 to column 21)",
  " (in 'inst/include/estimator/estimator.stan', line 19, column 2 to column 95)",
  " (in 'inst/include/estimator/estimator.stan', line 6, column 2 to column 12)",
  " (in 'inst/include/estimator/estimator.stan', line 7, column 2 to column 18)",
  " (in 'inst/include/estimator/estimator.stan', line 8, column 2 to column 16)",
  " (in 'inst/include/estimator/estimator.stan', line 9, column 8 to column 13)",
  " (in 'inst/include/estimator/estimator.stan', line 9, column 2 to column 32)",
  " (in 'inst/include/estimator/estimator.stan', line 10, column 9 to column 14)",
  " (in 'inst/include/estimator/estimator.stan', line 10, column 2 to column 29)",
  " (in 'inst/include/estimator/estimator.stan', line 11, column 9 to column 14)",
  " (in 'inst/include/estimator/estimator.stan', line 11, column 2 to column 29)",
  " (in 'inst/include/estimator/estimator.stan', line 15, column 9 to column 14)"};
class estimator_model final : public model_base_crtp<estimator_model> {
 private:
  int Npars;
  int finite_diff;
  int no_bounds;
  std::vector<int> bounds_types;
  Eigen::Matrix<double,-1,1> lower_bounds_data__;
  Eigen::Matrix<double,-1,1> upper_bounds_data__;
  Eigen::Map<Eigen::Matrix<double,-1,1>> lower_bounds{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double,-1,1>> upper_bounds{nullptr, 0};
 public:
  ~estimator_model() {}
  estimator_model(stan::io::var_context& context__, unsigned int
                  random_seed__ = 0, std::ostream* pstream__ = nullptr)
      : model_base_crtp(0) {
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    using local_scalar_t__ = double;
    auto base_rng__ = stan::services::util::create_rng(random_seed__, 0);
    // suppress unused var warning
    (void) base_rng__;
    static constexpr const char* function__ =
      "estimator_model_namespace::estimator_model";
    // suppress unused var warning
    (void) function__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 3;
      context__.validate_dims("data initialization", "Npars", "int",
        std::vector<size_t>{});
      Npars = std::numeric_limits<int>::min();
      current_statement__ = 3;
      Npars = context__.vals_i("Npars")[(1 - 1)];
      current_statement__ = 4;
      context__.validate_dims("data initialization", "finite_diff", "int",
        std::vector<size_t>{});
      finite_diff = std::numeric_limits<int>::min();
      current_statement__ = 4;
      finite_diff = context__.vals_i("finite_diff")[(1 - 1)];
      current_statement__ = 5;
      context__.validate_dims("data initialization", "no_bounds", "int",
        std::vector<size_t>{});
      no_bounds = std::numeric_limits<int>::min();
      current_statement__ = 5;
      no_bounds = context__.vals_i("no_bounds")[(1 - 1)];
      current_statement__ = 6;
      stan::math::validate_non_negative_index("bounds_types", "Npars", Npars);
      current_statement__ = 7;
      context__.validate_dims("data initialization", "bounds_types", "int",
        std::vector<size_t>{static_cast<size_t>(Npars)});
      bounds_types = std::vector<int>(Npars, std::numeric_limits<int>::min());
      current_statement__ = 7;
      bounds_types = context__.vals_i("bounds_types");
      current_statement__ = 8;
      stan::math::validate_non_negative_index("lower_bounds", "Npars", Npars);
      current_statement__ = 9;
      context__.validate_dims("data initialization", "lower_bounds",
        "double", std::vector<size_t>{static_cast<size_t>(Npars)});
      lower_bounds_data__ = Eigen::Matrix<double,-1,1>::Constant(Npars,
                              std::numeric_limits<double>::quiet_NaN());
      new (&lower_bounds)
        Eigen::Map<Eigen::Matrix<double,-1,1>>(lower_bounds_data__.data(),
        Npars);
      {
        std::vector<local_scalar_t__> lower_bounds_flat__;
        current_statement__ = 9;
        lower_bounds_flat__ = context__.vals_r("lower_bounds");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= Npars; ++sym1__) {
          stan::model::assign(lower_bounds, lower_bounds_flat__[(pos__ - 1)],
            "assigning variable lower_bounds", stan::model::index_uni(sym1__));
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 10;
      stan::math::validate_non_negative_index("upper_bounds", "Npars", Npars);
      current_statement__ = 11;
      context__.validate_dims("data initialization", "upper_bounds",
        "double", std::vector<size_t>{static_cast<size_t>(Npars)});
      upper_bounds_data__ = Eigen::Matrix<double,-1,1>::Constant(Npars,
                              std::numeric_limits<double>::quiet_NaN());
      new (&upper_bounds)
        Eigen::Map<Eigen::Matrix<double,-1,1>>(upper_bounds_data__.data(),
        Npars);
      {
        std::vector<local_scalar_t__> upper_bounds_flat__;
        current_statement__ = 11;
        upper_bounds_flat__ = context__.vals_r("upper_bounds");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= Npars; ++sym1__) {
          stan::model::assign(upper_bounds, upper_bounds_flat__[(pos__ - 1)],
            "assigning variable upper_bounds", stan::model::index_uni(sym1__));
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 12;
      stan::math::validate_non_negative_index("pars", "Npars", Npars);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = Npars;
  }
  inline std::string model_name() const final {
    return "estimator_model";
  }
  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.35.0",
             "stancflags = --allow-undefined"};
  }
  // Base log prob
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr,
            stan::require_not_st_var<VecR>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "estimator_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      Eigen::Matrix<local_scalar_t__,-1,1> pars =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(Npars, DUMMY_VAR__);
      current_statement__ = 1;
      pars = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(Npars);
      {
        current_statement__ = 2;
        lp_accum__.add(r_function<jacobian__>(pars, finite_diff, no_bounds, bounds_types,
                         lower_bounds, upper_bounds, pstream__));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  // Reverse mode autodiff log prob
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr,
            stan::require_st_var<VecR>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "estimator_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      Eigen::Matrix<local_scalar_t__,-1,1> pars =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(Npars, DUMMY_VAR__);
      current_statement__ = 1;
      pars = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(Npars);
      {
        current_statement__ = 2;
        lp_accum__.add(r_function<jacobian__>(pars, finite_diff, no_bounds, bounds_types,
                         lower_bounds, upper_bounds, pstream__));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  template <typename RNG, typename VecR, typename VecI, typename VecVar,
            stan::require_vector_like_vt<std::is_floating_point,
            VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral,
            VecI>* = nullptr, stan::require_vector_vt<std::is_floating_point,
            VecVar>* = nullptr>
  inline void
  write_array_impl(RNG& base_rng__, VecR& params_r__, VecI& params_i__,
                   VecVar& vars__, const bool
                   emit_transformed_parameters__ = true, const bool
                   emit_generated_quantities__ = true, std::ostream*
                   pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    // suppress unused var warning
    (void) propto__;
    double lp__ = 0.0;
    // suppress unused var warning
    (void) lp__;
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    constexpr bool jacobian__ = false;
    // suppress unused var warning
    (void) jacobian__;
    static constexpr const char* function__ =
      "estimator_model_namespace::write_array";
    // suppress unused var warning
    (void) function__;
    try {
      Eigen::Matrix<double,-1,1> pars =
        Eigen::Matrix<double,-1,1>::Constant(Npars,
          std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 1;
      pars = in__.template read<Eigen::Matrix<local_scalar_t__,-1,1>>(Npars);
      out__.write(pars);
      if (stan::math::logical_negation(
            (stan::math::primitive_value(emit_transformed_parameters__) ||
            stan::math::primitive_value(emit_generated_quantities__)))) {
        return ;
      }
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, typename VecI,
            stan::require_vector_t<VecVar>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline void
  unconstrain_array_impl(const VecVar& params_r__, const VecI& params_i__,
                         VecVar& vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      Eigen::Matrix<local_scalar_t__,-1,1> pars =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(Npars, DUMMY_VAR__);
      current_statement__ = 1;
      stan::model::assign(pars,
        in__.read<Eigen::Matrix<local_scalar_t__,-1,1>>(Npars),
        "assigning variable pars");
      out__.write(pars);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, stan::require_vector_t<VecVar>* = nullptr>
  inline void
  transform_inits_impl(const stan::io::var_context& context__, VecVar&
                       vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      current_statement__ = 1;
      context__.validate_dims("parameter initialization", "pars", "double",
        std::vector<size_t>{static_cast<size_t>(Npars)});
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      Eigen::Matrix<local_scalar_t__,-1,1> pars =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(Npars, DUMMY_VAR__);
      {
        std::vector<local_scalar_t__> pars_flat__;
        current_statement__ = 1;
        pars_flat__ = context__.vals_r("pars");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= Npars; ++sym1__) {
          stan::model::assign(pars, pars_flat__[(pos__ - 1)],
            "assigning variable pars", stan::model::index_uni(sym1__));
          pos__ = (pos__ + 1);
        }
      }
      out__.write(pars);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  inline void
  get_param_names(std::vector<std::string>& names__, const bool
                  emit_transformed_parameters__ = true, const bool
                  emit_generated_quantities__ = true) const {
    names__ = std::vector<std::string>{"pars"};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  get_dims(std::vector<std::vector<size_t>>& dimss__, const bool
           emit_transformed_parameters__ = true, const bool
           emit_generated_quantities__ = true) const {
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{static_cast<
                                                                    size_t>(
                                                                    Npars)}};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  constrained_param_names(std::vector<std::string>& param_names__, bool
                          emit_transformed_parameters__ = true, bool
                          emit_generated_quantities__ = true) const final {
    for (int sym1__ = 1; sym1__ <= Npars; ++sym1__) {
      param_names__.emplace_back(std::string() + "pars" + '.' +
        std::to_string(sym1__));
    }
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  unconstrained_param_names(std::vector<std::string>& param_names__, bool
                            emit_transformed_parameters__ = true, bool
                            emit_generated_quantities__ = true) const final {
    for (int sym1__ = 1; sym1__ <= Npars; ++sym1__) {
      param_names__.emplace_back(std::string() + "pars" + '.' +
        std::to_string(sym1__));
    }
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline std::string get_constrained_sizedtypes() const {
    return std::string("[{\"name\":\"pars\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(Npars) + "},\"block\":\"parameters\"}]");
  }
  inline std::string get_unconstrained_sizedtypes() const {
    return std::string("[{\"name\":\"pars\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(Npars) + "},\"block\":\"parameters\"}]");
  }
  // Begin method overload boilerplate
  template <typename RNG> inline void
  write_array(RNG& base_rng, Eigen::Matrix<double,-1,1>& params_r,
              Eigen::Matrix<double,-1,1>& vars, const bool
              emit_transformed_parameters = true, const bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = Npars;
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (0);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    std::vector<int> params_i;
    vars = Eigen::Matrix<double,-1,1>::Constant(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <typename RNG> inline void
  write_array(RNG& base_rng, std::vector<double>& params_r, std::vector<int>&
              params_i, std::vector<double>& vars, bool
              emit_transformed_parameters = true, bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = Npars;
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (0);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    vars = std::vector<double>(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(Eigen::Matrix<T_,-1,1>& params_r, std::ostream* pstream = nullptr) const {
    Eigen::Matrix<int,-1,1> params_i;
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(std::vector<T_>& params_r, std::vector<int>& params_i,
           std::ostream* pstream = nullptr) const {
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  inline void
  transform_inits(const stan::io::var_context& context,
                  Eigen::Matrix<double,-1,1>& params_r, std::ostream*
                  pstream = nullptr) const final {
    std::vector<double> params_r_vec(params_r.size());
    std::vector<int> params_i;
    transform_inits(context, params_i, params_r_vec, pstream);
    params_r = Eigen::Map<Eigen::Matrix<double,-1,1>>(params_r_vec.data(),
                 params_r_vec.size());
  }
  inline void
  transform_inits(const stan::io::var_context& context, std::vector<int>&
                  params_i, std::vector<double>& vars, std::ostream*
                  pstream__ = nullptr) const {
    vars.resize(num_params_r__);
    transform_inits_impl(context, vars, pstream__);
  }
  inline void
  unconstrain_array(const std::vector<double>& params_constrained,
                    std::vector<double>& params_unconstrained, std::ostream*
                    pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = std::vector<double>(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
  inline void
  unconstrain_array(const Eigen::Matrix<double,-1,1>& params_constrained,
                    Eigen::Matrix<double,-1,1>& params_unconstrained,
                    std::ostream* pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = Eigen::Matrix<double,-1,1>::Constant(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
};
}
using stan_model = estimator_model_namespace::estimator_model;
#ifndef USING_R
// Boilerplate
stan::model::model_base&
new_model(stan::io::var_context& data_context, unsigned int seed,
          std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}
stan::math::profile_map& get_stan_profile_data() {
  return estimator_model_namespace::profiles__;
}
#endif