functions {
  real r_function(vector x, int finite_diff, int no_bounds, array[] int bounds_types, vector lower_bounds, vector upper_bounds);
}

data {
  int Npars;
  int finite_diff;
  int no_bounds;
  array[Npars] int bounds_types;
  vector[Npars] lower_bounds;
  vector[Npars] upper_bounds;
}

parameters {
  vector[Npars] pars;
}

model {
  target += r_function(pars, finite_diff, no_bounds, bounds_types, lower_bounds, upper_bounds);
}
