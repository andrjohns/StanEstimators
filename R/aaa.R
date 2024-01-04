setClass(
  "StanBase",
  slots = c(
    metadata = "list",
    lower_bounds = "numeric",
    upper_bounds = "numeric",
    model_methods = "list"
  ))
