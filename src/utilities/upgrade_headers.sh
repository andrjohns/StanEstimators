CMDSTAN_VER="2.37.0"

wget https://github.com/stan-dev/cmdstan/releases/download/v$CMDSTAN_VER/cmdstan-$CMDSTAN_VER.tar.gz
tar -xf cmdstan-$CMDSTAN_VER.tar.gz
mv include/estimator estimator
rm -r include && mkdir include

cp -r cmdstan-$CMDSTAN_VER/src/cmdstan include/cmdstan
cp -r cmdstan-$CMDSTAN_VER/stan/src/stan include/stan
cp -r cmdstan-$CMDSTAN_VER/stan/lib/stan_math/stan/math include/stan/math
cp -r cmdstan-$CMDSTAN_VER/stan/lib/stan_math/lib/sundials_*/include .
cp -r cmdstan-$CMDSTAN_VER/stan/lib/stan_math/stan/math.hpp include/stan/math.hpp

# Shorten filename to avoid 100-byte path limit for R CMD CHECK
mv include/stan/math/opencl/kernel_generator/get_kernel_source_for_evaluating_into.hpp include/stan/math/opencl/kernel_generator/get_kernel_source_for_evaluating.hpp
sed -i -e 's/get_kernel_source_for_evaluating_into/get_kernel_source_for_evaluating/g' include/stan/math/opencl/kernel_generator.hpp
mv estimator include

chmod +x cmdstan-$CMDSTAN_VER/bin/mac-stanc
cmdstan-$CMDSTAN_VER/bin/mac-stanc include/estimator/estimator.stan --O1 --allow-undefined

sed -i -e 's/std::cout/Rcpp::Rcout/g' include/cmdstan/command.hpp
sed -i -e 's/std::cerr/Rcpp::Rcerr/g' include/cmdstan/command.hpp
sed -i -e 's/parser.print(info)/\/\/parser.print(info)/g' include/cmdstan/command.hpp
sed -i -e 's/std::cerr/Rcpp::Rcerr/g' include/cmdstan/command_helper.hpp
sed -i -e 's/r_function/r_function<jacobian__>/g' include/estimator/estimator.hpp

rm include/estimator/estimator.hpp-e
rm include/cmdstan/command.hpp-e
rm include/cmdstan/command_helper.hpp-e
rm include/stan/math/opencl/kernel_generator.hpp-e

rm cmdstan-$CMDSTAN_VER.tar.gz
rm -rf cmdstan-$CMDSTAN_VER
