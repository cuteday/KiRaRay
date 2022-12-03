# krr-math

This is just a simple wrapper (still header-only) upon [Eigen](http://eigen.tuxfamily.org/) for my own purpose. I implemented some interfaces on Eigen to make it more simpler to use for a rendering application.

### Usage

You can now use binary operators like `length()`, `cross()` and `dot()` between two vectors.

You can implement new methods on `Vector`, `Array` or other classes, by simply extending the class defined in its own header file (e.g., [vector.h](include/krrmath/vector.h)).

`Vector` and `Array` supports serialization/deserialization from/to nlohmann::json (needs to include *json.hpp* first).

The wrapper classes and its methods are also available in CUDA device code, as in Eigen.

To use this, just include this directory with `add_subdirectory()` in CMake, and link to `krr_math`.