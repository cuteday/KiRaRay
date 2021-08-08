#pragma once

#include <iostream>

#include "assimp/Importer.hpp"
#include "enoki/array.h"

typedef float Float;

typedef enoki::Array<Float, 3> Vec3f;
typedef enoki::Matrix<Float, 3> Mat3f;