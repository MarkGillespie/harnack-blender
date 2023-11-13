#pragma once

#include <array>
#include <cmath>

// evaluate the homogeneous harmonic polynomial representing
// the (l, m) spherical harmonic at point pos
template<typename T> T evaluateSphericalHarmonic(int l, int m, const std::array<T, 3> &pos);
template<typename T>
double evaluateSphericalHarmonicDouble(int n, int l, const std::array<T, 3> &pos);

// return a bound on the absolute value of the (l, m) spherical harmonic on the unit sphere
inline double sphericalHarmonicBound(int l, int m);

#include "spherical_harmonics.ipp"
