#include <stan/math.hpp>
#include <stan/model/indexing.hpp>

template <bool propto = FALSE, typename T, typename Tlower, typename Tupper, typename Tlp, typename Tlp_accum>
stan::plain_type_t<T> constrain_pars_lp(const T& uncons_pars,
                                       const std::vector<int>& bound_inds,
                                       const Tlower& lower,
                                       const Tupper& upper,
                                       Tlp& lp__, Tlp_accum& lp_accum__,
                                       std::ostream* pstream__) {
  stan::plain_type_t<T> pars = uncons_pars;

  stan::model::assign(
    pars,
    stan::math::lub_constrain(
      stan::model::rvalue(pars, "pars", stan::model::index_multi(bound_inds)),
      lower, upper, lp__
    ),
    "pars",
    stan::model::index_multi(bound_inds)
  );
  return pars;
}

template <bool propto = FALSE, typename T, typename Tlower, typename Tupper>
stan::plain_type_t<T> constrain_pars_lp(const T& uncons_pars,
                                       const std::vector<int>& bound_inds,
                                       const Tlower& lower,
                                       const Tupper& upper) {
  stan::plain_type_t<T> pars = uncons_pars;

  stan::model::assign(
    pars,
    stan::math::lub_constrain(
      stan::model::rvalue(pars, "pars", stan::model::index_multi(bound_inds)),
      lower, upper
    ),
    "pars",
    stan::model::index_multi(bound_inds)
  );
  return pars;
}
