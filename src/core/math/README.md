# krr-math

This is just a simple wrapper (still header-only) upon [Eigen](http://eigen.tuxfamily.org/) for my own purpose. I implemented some interfaces on Eigen to make it more simpler to use within rendering applications.

### Usage

You can now use unary and binary operators like `length()`, `cross()` and `dot()` on vectors.

You can implement new methods on `Vector`, `Array` or other classes by simply extending the class defined in its own header file (e.g., [vector.h](include/krrmath/vector.h)).

`Vector` and `Array` supports serialization/deserialization from/to nlohmann::json (needs to include *json.hpp* first).

Some clip-space matrix arithmetic (e.g., `perspective()`, `look_at()`, all in left-handed coordinate) that is missing in Eigen, might implement more when I need them.

The wrapper classes and their methods are also available in CUDA device code, as in Eigen.

To use this, just include this directory with `add_subdirectory()` in CMake, and link `krr_math`. All classes and functions are in `namespace krr`.