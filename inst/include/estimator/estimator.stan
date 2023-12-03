functions {
  real r_function(vector x, int finite_diff, vector lower_bounds, vector upper_bounds);
}

data {
  int Npars;
  int finite_diff;
  vector[Npars] lower_bounds;
  vector[Npars] upper_bounds;
}

parameters {
  vector<lower=lower_bounds, upper=upper_bounds>[Npars] pars;
}

model {
  target += r_function(pars, finite_diff, lower_bounds, upper_bounds);
}
