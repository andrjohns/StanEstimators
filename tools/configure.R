if (R.version$os == "emscripten") {
  tmp_folder <- tempdir()
  rcppparallel_tar <- file.path(tmp_folder, "rcppparallel.tgz")
  download.file(
    "https://rcppcore.r-universe.dev/bin/emscripten/contrib/4.5/RcppParallel_5.1.10.9000.tgz",
    rcppparallel_tar,
    mode = "wb"
  )
  untar(rcppparallel_tar)
  Sys.setenv("TBB_INC"=file.path(tmp_folder, "RcppParallel", "include"))
  Sys.setenv("TBB_LIB"=file.path(tmp_folder, "RcppParallel", "lib"))
  if (Sys.getenv("GITHUB_ENV") != "") {
    cat(
      paste0("TBB_INC=",file.path(tmp_folder, "RcppParallel", "include")),
      sep = "\n",
      append = TRUE
    )
    cat(
      paste0("TBB_LIB=",file.path(tmp_folder, "RcppParallel", "lib")),
      sep = "\n",
      append = TRUE
    )
  }
}
