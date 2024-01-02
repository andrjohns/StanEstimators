/**
 * TBB Machine detection fails under emscripten when processing included
 * headers, so we force the compiler to treat the machine as generic linux
 * during configuration.
 *
 * No TBB parallelism is used in the package, so this should not be a problem.
 *
*/
#ifdef __EMSCRIPTEN__
    #define __linux__ 1
#endif
#include <tbb/tbb_machine.h>
#ifdef __EMSCRIPTEN__
    #undef __linux__
#endif
