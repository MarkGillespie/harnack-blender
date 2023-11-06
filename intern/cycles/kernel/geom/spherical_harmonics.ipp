// evaluate the homogeneous harmonic polynomial representing
// the (n,m) spherical harmonic at point pos
template<typename T> T evaluateSphericalHarmonic(int n, int m, const std::array<T, 3> &pos)
{
  return static_cast<T>(evaluateSphericalHarmonicDouble(n, m, pos));
}

template<typename T> T evaluateSphericalHarmonicDouble(int n, int m, const std::array<T, 3> &pos)
{
  T x = pos[0];
  T y = pos[1];
  T z = pos[2];
  if (n == 1 && m == -1) {
    return -0.5 * (std::sqrt(3 / M_PI) * y);
  }
  else if (n == 1 && m == 0) {
    return std::sqrt(3 / (2. * M_PI)) * z;
  }
  else if (n == 1 && m == 1) {
    return -0.5 * (std::sqrt(3 / M_PI) * x);
  }
  else if (n == 2 && m == -2) {
    return (std::sqrt(15 / M_PI) * x * y) / 2.;
  }
  else if (n == 2 && m == -1) {
    return -0.5 * (std::sqrt(15 / M_PI) * y * z);
  }
  else if (n == 2 && m == 0) {
    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    return (-0.5 * x2 - y2 / 2. + z2) * std::sqrt(5 / (2. * M_PI));
  }
  else if (n == 2 && m == 1) {
    return -0.5 * (std::sqrt(15 / M_PI) * x * z);
  }
  else if (n == 2 && m == 2) {
    T x2 = x * x;
    T y2 = y * y;
    return ((x2 - y2) * std::sqrt(15 / M_PI)) / 4.;
  }
  else if (n == 3 && m == -3) {
    T x2 = x * x;
    T y3 = y * y * y;
    return (std::sqrt(35 / (2. * M_PI)) * (y3 - 3 * x2 * y)) / 4.;
  }
  else if (n == 3 && m == -2) {
    return (std::sqrt(105 / M_PI) * x * y * z) / 2.;
  }
  else if (n == 3 && m == -1) {
    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    return (std::sqrt(21 / (2. * M_PI)) * (x2 * y + (y2 - 4 * z2) * y)) / 4.;
  }
  else if (n == 3 && m == 0) {
    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    return std::sqrt(7 / (2. * M_PI)) * ((-3 * x2 * z) / 2. + ((-3 * y2) / 2. + z2) * z);
  }
  else if (n == 3 && m == 1) {
    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    return ((x2 + y2 - 4 * z2) * std::sqrt(21 / (2. * M_PI)) * x) / 4.;
  }
  else if (n == 3 && m == 2) {
    T x2 = x * x;
    T y2 = y * y;
    return (std::sqrt(105 / M_PI) * (x2 * z - y2 * z)) / 4.;
  }
  else if (n == 3 && m == 3) {
    T x2 = x * x;
    T y2 = y * y;
    return -0.25 * ((x2 - 3 * y2) * std::sqrt(35 / (2. * M_PI)) * x);
  }
  else if (n == 4 && m == -4) {
    T x2 = x * x;
    T y3 = y * y * y;
    return (3 * std::sqrt(35 / M_PI) * x * (-y3 + x2 * y)) / 4.;
  }
  else if (n == 4 && m == -3) {
    T x2 = x * x;
    T y3 = y * y * y;
    return (-3 * std::sqrt(35 / (2. * M_PI)) * (-(y3 * z) + 3 * x2 * y * z)) / 4.;
  }
  else if (n == 4 && m == -2) {
    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    return (-3 * std::sqrt(5 / M_PI) * x * (x2 * y + (y2 - 6 * z2) * y)) / 4.;
  }
  else if (n == 4 && m == -1) {
    T x2 = x * x;
    T y2 = y * y;
    T z3 = z * z * z;
    return (3 * std::sqrt(5 / (2. * M_PI)) * (3 * x2 * y * z + y * (-4 * z3 + 3 * y2 * z))) / 4.;
  }
  else if (n == 4 && m == 0) {
    T x2 = x * x;
    T y2 = y * y;
    T z4 = z * z * z * z;
    T z2 = z * z;
    return (3 *
            (y2 * ((3 * y2) / 8. - 3 * z2) + x2 * ((3 * x2) / 8. + (3 * y2) / 4. - 3 * z2) + z4)) /
           std::sqrt(2 * M_PI);
  }
  else if (n == 4 && m == 1) {
    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    return (3 * std::sqrt(5 / (2. * M_PI)) * x * (3 * x2 * z + (3 * y2 - 4 * z2) * z)) / 4.;
  }
  else if (n == 4 && m == 2) {
    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    return (-3 * (x2 * (x2 - 6 * z2) + y2 * (-y2 + 6 * z2)) * std::sqrt(5 / M_PI)) / 8.;
  }
  else if (n == 4 && m == 3) {
    T x2 = x * x;
    T y2 = y * y;
    return (-3 * std::sqrt(35 / (2. * M_PI)) * x * (x2 * z - 3 * y2 * z)) / 4.;
  }
  else if (n == 4 && m == 4) {
    T x2 = x * x;
    T y4 = y * y * y * y;
    T y2 = y * y;
    return (3 * (x2 * (x2 - 6 * y2) + y4) * std::sqrt(35 / M_PI)) / 16.;
  }
  return 0;
}
// return the minimum value that the (n, m) spherical harmonic takes on the unit sphere
double sphericalHarmonicMinValue(int n, int m)
{
  if (n == 1 && m == -1) {
    return 0.488601;
  }
  else if (n == 1 && m == 0) {
    return 0.690989;
  }
  else if (n == 1 && m == 1) {
    return 0.488601;
  }
  else if (n == 2 && m == -2) {
    return 0.546274;
  }
  else if (n == 2 && m == -1) {
    return 0.546274;
  }
  else if (n == 2 && m == 0) {
    return 0.446031;
  }
  else if (n == 2 && m == 1) {
    return 0.546274;
  }
  else if (n == 2 && m == 2) {
    return 0.546274;
  }
  else if (n == 3 && m == -3) {
    return 0.590044;
  }
  else if (n == 3 && m == -2) {
    return 0.556298;
  }
  else if (n == 3 && m == -1) {
    return 0.62938;
  }
  else if (n == 3 && m == 0) {
    return 1.0555;
  }
  else if (n == 3 && m == 1) {
    return 0.62938;
  }
  else if (n == 3 && m == 2) {
    return 0.556298;
  }
  else if (n == 3 && m == 3) {
    return 0.590044;
  }
  else if (n == 4 && m == -4) {
    return 0.625836;
  }
  else if (n == 4 && m == -3) {
    return 0.574867;
  }
  else if (n == 4 && m == -2) {
    return 0.608255;
  }
  else if (n == 4 && m == -1) {
    return 0.706531;
  }
  else if (n == 4 && m == 0) {
    return 0.512926;
  }
  else if (n == 4 && m == 1) {
    return 0.706531;
  }
  else if (n == 4 && m == 2) {
    return 0.473087;
  }
  else if (n == 4 && m == 3) {
    return 0.574867;
  }
  else if (n == 4 && m == 4) {
    return 0.625836;
  }
  else if (n == 5 && m == -5) {
    return 0.656382;
  }
  else if (n == 5 && m == -4) {
    return 0.594089;
  }
  return 0;
}
