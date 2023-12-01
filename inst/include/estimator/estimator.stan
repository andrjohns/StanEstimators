functions {
  real r_function(vector x, int finite_diff);
}

data {
  int Npars;
  int finite_diff;
}

parameters {
  vector[Npars] pars;
}

model {
  target += r_function(pars, finite_diff);
}
