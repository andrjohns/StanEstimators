functions {
  real r_function(vector x, int finite_diff);
  vector constrain_pars_lp(vector x, array[] int bound_inds,
                           vector lbounds, vector ubounds);
}

data {
  int Npars;
  int finite_diff;
  int Nbounds;
  array[Nbounds] int bound_inds;
  vector[Nbounds] lower_bounds;
  vector[Nbounds] upper_bounds;
}

parameters {
  vector[Npars] pars;
}

transformed parameters {
  vector[Npars] constrained_pars = constrain_pars_lp(pars, bound_inds, lower_bounds, upper_bounds);
}

model {
  target += r_function(constrained_pars, finite_diff);
}
