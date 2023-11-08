#pragma once

#include <array>
#include <cmath>

// evaluate the homogeneous harmonic polynomial representing
// the (n,m) spherical harmonic at point pos
template<typename T> T evaluateSphericalHarmonic(int n, int m, const std::array<T, 3> &pos);
template<typename T>
double evaluateSphericalHarmonicDouble(int n, int m, const std::array<T, 3> &pos);

// return the minimum value that the (n, m) spherical harmonic takes on the unit sphere
inline double sphericalHarmonicMinValue(int n, int m);

#include "spherical_harmonics.ipp"
