# Krr-Math

This is just a simple wrapper (still header-only) upon [Eigen](http://eigen.tuxfamily.org/) for my own purposes. I implemented some interfaces on Eigen to make it more simpler to use within rendering applications (especially for raytracing on GPU).

### Usage

To use this, just include this directory with `add_subdirectory()` in CMake, and link the interface library `krr_math`. A single `target_link_libraries` command then you are all set. All classes and functions are in `namespace krr`.

### Features

- Unary and binary operators like `length()`, `cross()` and `dot()` on vectors.
- `Vector`, `Array`, `Matrix` and `Quaternion`, etc. supports serialization/deserialization from/to nlohmann::json. To enable this, you need to include *json.hpp* before include the headers here.
- Some clip-space matrix arithmetic (e.g., `perspective()`, `look_at()`, `orthogonal()`, in left or right-handed coordinates) that is missing in Eigen, might implement more when I need them.
- The wrapper classes and their methods are also available in CUDA device code, as in Eigen.

You can also implement new methods on `Vector`, `Array`, `Matrix` or other classes by simply extending the class defined in its own header file (e.g., [vector.h](include/krrmath/vector.h)).

### Caveats

#### Different behavior of accessing matrix elements

Note that Eigen use the `mat(i, j)` styled matrix accessor to index the element in i-th row and j-th column. This, however, is different from GLSL and GLM, where `mat[i]` retrieves the i-th column of the matrix.  