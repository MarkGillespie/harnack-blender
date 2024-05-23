// TODO: move back to math utils?
// int solid_angle_formula (from DNA_modifier_types.h):
//    MOD_HARNACK_TRIANGULATE = 0,
//    MOD_HARNACK_PREQUANTUM = 1,
//    MOD_HARNACK_GAUSS_BONNET = 2,

// find the two times at which a ray intersects a sphere
template<typename T>
bool intersect_sphere(
    const std::array<T, 3> &ro, const std::array<T, 3> &rd, T radius, T *t0, T *t1)
{
  // solve | ro + t rd  |^2 == radius^2
  T a = dot(rd, rd);
  T b = static_cast<T>(2) * dot(rd, ro);
  T c = dot(ro, ro) - radius * radius;
  T discr = b * b - static_cast<T>(4) * a * c;
  if (discr < 0) {
    return false;
  }
  else {
    T sqrt_d = std::sqrt(discr);
    T q = (b > 0) ? static_cast<T>(-0.5) * (b + sqrt_d) : static_cast<T>(-0.5) * (b - sqrt_d);
    *t0 = q / a;
    *t1 = c / q;
    if (*t1 < *t0)
      std::swap(*t0, *t1);
    return true;
  }
}

typedef struct spherical_harmonic_intersection_params {
  float3 ray_P;
  float3 ray_D;
  float ray_tmin;
  float ray_tmax;
  float R;
  uint l;
  int m;
  float epsilon;
  float levelset;
  float frequency;
  bool use_grad_termination;
  int max_iterations;
} spherical_harmonic_intersection_params;

template<typename T>
ccl_device bool ray_spherical_harmonic_intersect_T(
    const spherical_harmonic_intersection_params &params,
    ccl_private float *isect_u,
    ccl_private float *isect_v,
    ccl_private float *isect_t)
{
  using T3 = std::array<T, 3>;

  T epsilon = static_cast<T>(params.epsilon);
  T frequency = static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);

  T3 ray_P = from_float3<T>(params.ray_P);
  T3 ray_D = from_float3<T>(params.ray_D);

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);

  T radius = static_cast<T>(params.R);
  T outerRadius = static_cast<T>(1.25) * radius;  // TODO: make configurable?

  T unitBound = sphericalHarmonicBound(params.l, params.m);
  // since the spherical harmonic is homogeneous of degree l, we can bound it
  // using a bound on the unit sphere and our sphere radius
  T shift = unitBound * static_cast<T>(std::pow(outerRadius, params.l));

  auto distance_to_boundary = [&](const T3 &x) -> T { return outerRadius - len(x); };

  // Return the value of this harmonic polynomial at an evaluation point p,
  auto evaluatePolynomial = [&](const T3 &p) -> T {
    return evaluateSphericalHarmonic(params.l, params.m, p);
  };

  // TODO: take in analytic expressions?
  // finite difference gradient from
  // https://iquilezles.org/articles/normalsSDF
  auto calculateGradient = [&](const T3 &p) -> T3 {
    const double eps = 0.005;
    T q0 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{1, -1, -1}));
    T q1 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{-1, -1, 1}));
    T q2 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{-1, 1, -1}));
    T q3 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{1, 1, 1}));
    return T3{static_cast<T>(.5773) * (q3 + q0 - q1 - q2),
              static_cast<T>(.5773) * (q3 - q0 - q1 + q2),
              static_cast<T>(.5773) * (q3 - q0 + q1 - q2)};
  };

  // Within the space of all values `levelset + 2π * frequency * k`, find the
  // two bracketing the current value of f lower_levelset is set to the
  // smaller of the two, and upper_levelset is set to the larger
  auto findEnclosingLevelsets =
      [](T f, T levelset, T frequency, T *lower_levelset, T *upper_levelset) {
        // 𝑓0 ← (𝑓(r𝑡)−𝜙)/(2𝜋𝜔)
        // 𝑓− ← (2𝜋𝜔)⌊𝑓0⌋ + 𝜙
        // 𝑓+ ← (2𝜋𝜔)⌈𝑓0⌉ + 𝜙
        T f0 = (f - levelset) / (static_cast<T>(2 * M_PI) * frequency);
        *lower_levelset = static_cast<T>(2 * M_PI) * frequency * std::floor(f0) + levelset;
        *upper_levelset = static_cast<T>(2 * M_PI) * frequency * std::ceil(f0) + levelset;
      };
  // Computes a conservative step size via Harnack bounds, using the value
  // fx at the current point, the radius R of a ball over which the function
  // is harmonic, the values lo_bound and up_bound of the closest level sets
  // above/ below the current value, and a shift that makes the harmonic
  // function positive within this ball.
  auto getMaxStep = [](T fx, T R, T lo_bound, T up_bound, T shift) -> T {
    T w = (fx + shift) / (up_bound + shift);
    T v = (fx + shift) / (lo_bound + shift);
    T lo_r = -R / 2 * (v + 2 - std::sqrt(v * v + 8 * v));
    T up_r = R / 2 * (w + 2 - std::sqrt(w * w + 8 * w));

    return std::min(lo_r, up_r);
  };
  auto getMaxStep0 = [](T fx, T R, T bound, T shift) -> T {
    T a = (fx + shift) / (bound + shift);
    return R / 2 * std::abs(a + 2 - std::sqrt(a * a + 8 * a));
  };

  auto distanceToLevelset = [&](T f, T levelset, const T3 &grad) -> T {
    T scaling = params.use_grad_termination ? fmax(len(grad), epsilon) : 1;
    return std::abs(f - levelset) / scaling;
  };

  auto distanceToBothLevelsets =
      [&](T f, T lower_levelset, T upper_levelset, const T3 &grad) -> T {
    T scaling = params.use_grad_termination ? fmax(len(grad), epsilon) : 1;
    return fmin(f - lower_levelset, upper_levelset - f) / scaling;
  };

  // check if ray intersects sphere. If so, store two intersection times (t0 <
  // t1)
  T t0, t1;
  bool hit_sphere = intersect_sphere(ray_P, ray_D, radius, &t0, &t1);

  // if we miss the sphere, there cannot be an intersection with the levelset
  // within the sphere
  if (!hit_sphere)
    return false;

  T t = fmax(t0,
             ray_tmin);         // start at first sphere intersection if it is ahead of us
  T tMax = fmin(ray_tmax, t1);  // only trace until second sphere intersection
  T ld = len(ray_D);

  // If we're in the periodic case, identify the two levelsets bracketing our
  // starting position Note that these should remain fixed as we step, since
  // we never want to cross the levelsets
  T lower_levelset, upper_levelset;
  if (params.frequency > 0) {
    T3 pos = fma(ray_P, t, ray_D);
    findEnclosingLevelsets(
        evaluatePolynomial(pos), levelset, frequency, &lower_levelset, &upper_levelset);
  }

  int iter = 0;

  static bool exceeded_max = false;

  // Until we reach the maximum ray distance
  while (t < tMax) {
    // If we've exceeded the maximum number of iterations, print a warning
    if (iter >= params.max_iterations) {
      if (!exceeded_max) {
        exceeded_max = true;
        printf(
            "Warning: exceeded maximum number of Harnack "
            "iterations.\n");
      }
    }

    T3 pos = fma(ray_P, t, ray_D);
    T f = evaluatePolynomial(pos);
    T3 grad;
    if (params.use_grad_termination)
      grad = calculateGradient(pos);

    // compute the distance to the level set.
    // if we're working periodically, this involves computing the closest
    // levelsets above and below us, which are also used later to compute a
    // safe step size
    T dist;
    if (params.frequency < 0) {  // just looking for a single level set
      dist = distanceToLevelset(f, levelset, grad);
    }
    else {  // looking for periodic level sets
      dist = distanceToBothLevelsets(f, lower_levelset, upper_levelset, grad);
    }

    // If we're close enough to the level set, return a hit.
    if (dist < epsilon) {
      // TODO: add to normal code
      /*
          T3 grad = calculateGradient(pos);
          // shade points on the boundary sphere using the sphere's normal
          // ( for some reason, these points occasionally cause problems
         otherwise ) if (1 - LengthSquared(Vector3d(pos)) < epsilon) grad
         = Vector3d(pos); return constructIntersection(t, pos, grad, rd,
         f);
      */
      *isect_t = t;
      *isect_u = f;
      *isect_v = ((T)iter) / ((T)params.max_iterations);
      return true;
    }

    T R = distance_to_boundary(pos);

    T step;
    if (params.frequency < 0) {  // just looking for a single level set
      step = getMaxStep0(f, R, levelset, shift);
    }
    else {  // looking for periodic level sets
      step = getMaxStep(f, R, lower_levelset, upper_levelset, shift);
    }

    t += step / ld;
    iter++;
  }

  return false;  // no intersection
}

// Fractional mod function that behaves the same as in GLSL
template<typename T> T glsl_mod(T x, T y)
{
  return x - y * std::floor(x / y);
}

template<typename T>
bool close_to_zero(
    T ang, T lo_bound, T hi_bound, T tol, const std::array<T, 3> &grad, bool use_grad_termination)
{
  // Check if an angle is within epsilon of 0 or 4π
  T dis = fmin(ang - lo_bound, hi_bound - ang);
  T tolScaling = use_grad_termination ? len(grad) : 1;
  return dis < tol * tolScaling;
};

// compute solid angle of loop starting at pts[iStart] with length N,
// evaluated at point x, and adds the gradient to *grad, computed as follows:
// grad_mode: 0 - nicole formula
//            1 - Adiels formula 10
//            2 - Adiels formula 8
//            3 - none

// (if grad == nullptr, gradient is not computed anyway)
template<typename T>
T triangulated_loop_solid_angle(const packed_float3 *pts,
                                uint iStart,
                                uint N,
                                const std::array<T, 3> &x,
                                std::array<T, 3> *grad,
                                int grad_mode = 0,
                                bool use_quick_triangulation = false)
{
  using T3 = std::array<T, 3>;

  // compute the vectors xp from the evaluation point x
  // to all the polygon vertices, and their lengths Lp
  std::vector<T3> xp;
  xp.reserve(N + 1);
  std::vector<T> Lp;
  Lp.reserve(N + 1);
  for (uint i = 0; i < N + 1; i++) {  // center = pts[N]
    xp.push_back(diff_f(pts[iStart + i], x));
    Lp.push_back(len(xp[i]));
  }

  // Iterate over triangles used to triangulate the polygon
  std::complex<T> running_angle{1., 0.};
  uint start = use_quick_triangulation ? 2 : 0;
  for (uint i = start; i < N; i++) {
    int a = i;
    int b = (i + 1) % N;
    int c = N;

    T3 n = cross(xp[a], xp[b]);

    // Add the solid angle of this triangle to the total
    std::complex<T> tri_angle{Lp[a] * Lp[b] * Lp[c] + dot(xp[a], xp[b]) * Lp[c] +
                                  dot(xp[b], xp[c]) * Lp[a] + dot(xp[a], xp[c]) * Lp[b],
                              dot(xp[c], n)};
    running_angle *= tri_angle;

    // normalize complex number every so often
    if (i % 5 == 0)
      running_angle /= std::abs(running_angle);

    //== compute gradient
    if (grad) {
      const T3 &g0 = xp[a];
      const T3 &g1 = xp[b];
      if (grad_mode == 0) {
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
      else if (grad_mode == 1) {
        T lv = len(g0);
        T lw = len(g1);
        T scale = (lv + lw) / (lv * lw * (lv * lw + dot(g0, g1)));
        (*grad)[0] += n[0] * scale;
        (*grad)[1] += n[1] * scale;
        (*grad)[2] += n[2] * scale;
      }
      else if (grad_mode == 2) {
        T n2 = len_squared(n);
        T scale = dot(g0 - g1, normalized(g0) - normalized(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
    }
  }

  return 2 * std::arg(running_angle);
}

// compute solid angle all loops in list polygonLoops evaluated at point x,
// and adds the gradient to *grad, computed as follows:
// grad_mode: 0 - nicole formula
//            1 - Adiels formula 10
//            2 - Adiels formula 8
//            3 - none
// (if grad == nullptr, gradient is not computed anyway)
template<typename T>
T triangulated_solid_angle(const packed_float3 *pts,
                           const packed_uint3 *loops,
                           uint globalStart,
                           const std::vector<uint> &polygonLoops,
                           const std::array<T, 3> &x,
                           std::array<T, 3> *grad,
                           int grad_mode = 0,
                           bool use_quick_triangulation = false)
{
  T omega = 0;
  for (uint iL : polygonLoops) {
    uint iStart = loops[iL].x - globalStart;
    uint N = loops[iL].y;

    omega += triangulated_loop_solid_angle<T>(
        pts, iStart, N, x, grad, grad_mode, use_quick_triangulation);
  }

  return omega;
}

// compute solid angle of loop starting at pts[iStart] with length N,
// evaluated at point x, using the prequantum formula and adds the gradient to
// *grad, computed as follows:
// grad_mode: 0 - nicole formula
//            1 - Adiels formula 10
//            2 - Adiels formula 8
//            3 - none
// (if grad == nullptr, gradient is not computed anyway)
template<typename T>
T prequantum_loop_solid_angle(const packed_float3 *pts,
                              uint iStart,
                              uint N,
                              const std::array<T, 3> &x,
                              std::array<T, 3> *grad,
                              int grad_mode = 0)
{
  using T3 = std::array<T, 3>;
  using T4 = std::array<T, 4>;

  // compute the vectors xp from the evaluation point x
  // to all the polygon vertices, and their lengths Lp
  std::vector<T3> xp;
  xp.reserve(N);
  std::vector<T> Lp;
  Lp.reserve(N);
  for (uint i = 0; i < N; i++) {
    xp.push_back(diff_f(pts[iStart + i], x));
    Lp.push_back(len(xp[i]));
  }

  int start = 0;
  T4 q0 = dihedral(T3{1., 0., 0.},
                   xp[start]);  // point in fiber of points[start]-x
  T4 qi = q0;
  for (uint i = 0; i < N; i++) {
    int a = i;
    int b = (i + 1) % N;
    T4 d = dihedral(xp[a], xp[b]);
    qi = q_mul(d, qi);

    //== compute gradient
    if (grad) {
      const T3 &g0 = xp[a];
      const T3 &g1 = xp[b];
      const T3 &n = cross(g0, g1);
      if (grad_mode == 0) {
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
      else if (grad_mode == 1) {
        T lv = len(g0);
        T lw = len(g1);
        T scale = (lv + lw) / (lv * lw * (lv * lw + dot(g0, g1)));
        (*grad)[0] += n[0] * scale;
        (*grad)[1] += n[1] * scale;
        (*grad)[2] += n[2] * scale;
      }
      else if (grad_mode == 2) {
        T n2 = len_squared(n);
        T scale = dot(g0 - g1, normalized(g0) - normalized(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
    }
  }

  return static_cast<T>(-2.) * fiberArg(q0, qi);
}

// compute solid angle via prequantum formula for all loops in list polygonLoops
// evaluated at point x, and adds the gradient to *grad, computed as follows:
// grad_mode: 0 - nicole formula
//            1 - Adiels formula 10
//            2 - Adiels formula 8
//            3 - none
// (if grad == nullptr, gradient is not computed anyway)
template<typename T>
T prequantum_solid_angle(const packed_float3 *pts,
                         const packed_uint3 *loops,
                         uint globalStart,
                         const std::vector<uint> &polygonLoops,
                         const std::array<T, 3> &x,
                         std::array<T, 3> *grad,
                         int grad_mode = 0)
{
  T omega = 0;
  for (uint iL : polygonLoops) {
    uint iStart = loops[iL].x - globalStart;
    uint N = loops[iL].y;

    omega += prequantum_loop_solid_angle<T>(pts, iStart, N, x, grad, grad_mode);
  }

  return omega;
}

#ifdef HAS_POLYSCOPE
#  include "polyscope/curve_network.h"
#endif

template<typename T>
// take in a curve on the sphere, stored as a list of unnormalized points `xp`
T loop_rotation_index(const std::vector<std::array<T, 3>> &xp, int verbosity = 0)
{
  using T3 = std::array<T, 3>;
  using T2 = std::array<T, 2>;

  size_t nS = 10;      // number of substeps to take along curve
  std::vector<T2> ps;  // stereographic projections to plane
  ps.reserve(nS * xp.size());
  for (size_t iP = 0; iP < xp.size(); iP++) {
    const T3 &q0 = xp[iP];
    const T3 &q1 = xp[(iP + 1) % xp.size()];

    for (size_t iS = 0; iS < nS; iS++) {
      T t = static_cast<T>(iS) / static_cast<T>(nS);

      // use t to interpolate between q0 and q1
      T3 q = {
          (1 - t) * q0[0] + t * q1[0], (1 - t) * q0[1] + t * q1[1], (1 - t) * q0[2] + t * q1[2]};

      // stereographic projection to plane
      T l = len(q);
      ps.push_back({q[0] / (l - q[2]), q[1] / (l - q[2])});
    }

    if (verbosity >= 2) {
      std::cout << "    >> [rotation_index] ps[" << iP << "]: (" << ps.back()[0] << ", "
                << ps.back()[1] << ")" << std::endl;
    }
  }

  auto dot = [](const T2 &v, const T2 &w) { return v[0] * w[0] + v[1] * w[1]; };
  auto cross = [](const T2 &v, const T2 &w) { return v[0] * w[1] - v[1] * w[0]; };

  // sum up total turning angle
  T total_angle = 0;
  size_t N = ps.size();
  for (uint i = 0; i < N; i++) {
    T2 a = ps[(i + N - 1) % N];
    T2 b = ps[i];
    T2 c = ps[(i + 1) % N];

    T2 ab = {b[0] - a[0], b[1] - a[1]};
    T2 bc = {c[0] - b[0], c[1] - b[1]};

    T turn_angle = atan2(cross(ab, bc), dot(ab, bc));

    total_angle += turn_angle;
    if (verbosity >= 2) {
      T turn_degrees = turn_angle / M_PI * 180.;
      std::cout << "    >> [rotation_index] turn[" << i << "]: " << turn_degrees << " degrees"
                << std::endl;
    }
  }
  if (verbosity >= 1) {
    std::cout << "    >> [rotation_index] total angle: " << total_angle << std::endl;
  }

#ifdef HAS_POLYSCOPE
  if (verbosity >= 2) {
    polyscope::registerCurveNetworkLoop2D("ps", ps);
    polyscope::show();
  }
#endif

  return total_angle / (2. * M_PI);
}

// compute solid angle of loop starting at pts[iStart] with length N,
// evaluated at point x, using the gauss-bonnet formula and adds the gradient to
// *grad, computed as follows:
// grad_mode: 0 - nicole formula
//            1 - Adiels formula 10
//            2 - Adiels formula 8
//            3 - none
// (if grad == nullptr, gradient is not computed anyway)
template<typename T>
T gauss_bonnet_loop_solid_angle(const packed_float3 *pts,
                                uint iStart,
                                uint N,
                                const std::array<T, 3> &x,
                                std::array<T, 3> *grad,
                                int grad_mode = 0,
                                int verbosity = 0)
{
  using T3 = std::array<T, 3>;

  // compute the vectors xp from the evaluation point x
  // to all the polygon vertices, and their lengths Lp
  std::vector<T3> xp;
  xp.reserve(N);
  std::vector<T> Lp;
  Lp.reserve(N);
  for (uint i = 0; i < N; i++) {
    xp.push_back(diff_f(pts[iStart + i], x));
    Lp.push_back(len(xp[i]));
  }

  // Iterate over polygon corners
  T total_angle = 0.;
  for (uint i = 0; i < N; i++) {
    int a = (i + N - 1) % N;
    int b = i;
    int c = (i + 1) % N;
    T3 n_prev = cross(xp[a], xp[b]);
    T3 n_next = cross(xp[b], xp[c]);

    T corner_angle = atan2(dot(xp[b], cross(n_prev, n_next)), Lp[b] * dot(n_prev, n_next));

    total_angle += corner_angle;

    if (verbosity >= 1) {
      std::cout << "  corner " << i << " has angle " << corner_angle << std::endl;
    }

    //== compute gradient
    if (grad) {
      const T3 &g0 = xp[a];
      const T3 &g1 = xp[b];
      const T3 &n = n_prev;
      if (grad_mode == 0) {
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
      else if (grad_mode == 1) {
        T lv = len(g0);
        T lw = len(g1);
        T scale = (lv + lw) / (lv * lw * (lv * lw + dot(g0, g1)));
        (*grad)[0] += n[0] * scale;
        (*grad)[1] += n[1] * scale;
        (*grad)[2] += n[2] * scale;
      }
      else if (grad_mode == 2) {
        T n2 = len_squared(n);
        T scale = dot(g0 - g1, normalized(g0) - normalized(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
    }
  }
  T rho = loop_rotation_index(xp, verbosity);
  if (verbosity >= 1) {
    std::cout << "      rho " << rho << std::endl;
  }
  return static_cast<T>(2. * M_PI) * rho - total_angle;
}

// compute solid angle via gauss-bonnet for all loops in list polygonLoops
// evaluated at point x, and adds the gradient to *grad, computed as follows:
// grad_mode: 0 - nicole formula
//            1 - Adiels formula 10
//            2 - Adiels formula 8
//            3 - none
// (if grad == nullptr, gradient is not computed anyway)
template<typename T>
T gauss_bonnet_solid_angle(const packed_float3 *pts,
                           const packed_uint3 *loops,
                           uint globalStart,
                           const std::vector<uint> &polygonLoops,
                           const std::array<T, 3> &x,
                           std::array<T, 3> *grad,
                           int grad_mode = 0,
                           int verbosity = 0)
{
  T omega = 0;
  for (uint iL : polygonLoops) {
    uint iStart = loops[iL].x - globalStart;
    uint N = loops[iL].y;

    omega += gauss_bonnet_loop_solid_angle<T>(pts, iStart, N, x, grad, grad_mode, verbosity);
  }

  return omega;
}

// compute solid angle bonnet for all loops in list polygonLoops
// evaluated at point x,computed as follows:
// solid_angle_mode: 0 - triangulated
//                   1 - prequantum
//                   2 - gauss-bonnet
// and adds the gradient to *grad, computed as follows:
// grad_mode: 0 - nicole formula
//            1 - Adiels formula 10
//            2 - Adiels formula 8
//            3 - none
// (if grad == nullptr, gradient is not computed anyway)
template<typename T>
T polygon_solid_angle(const packed_float3 *pts,
                      const packed_uint3 *loops,
                      uint globalStart,
                      const std::vector<uint> &polygonLoops,
                      const std::array<T, 3> &x,
                      int solid_angle_mode,
                      std::array<T, 3> *grad,
                      int grad_mode = 0,
                      bool use_quick_triangulation = false,
                      int verbosity = 0)
{
  switch (solid_angle_mode) {
    case 0:
      return triangulated_solid_angle(
          pts, loops, globalStart, polygonLoops, x, grad, grad_mode, use_quick_triangulation);
    case 1:
      return prequantum_solid_angle(pts, loops, globalStart, polygonLoops, x, grad, grad_mode);
    case 2:
      return gauss_bonnet_solid_angle(
          pts, loops, globalStart, polygonLoops, x, grad, grad_mode, verbosity);
    default:
      throw std::runtime_error("Unrecognized solid angle mode: '" +
                               std::to_string(solid_angle_mode) + "'");
      return 0;
  }
}

// compute solid angle of given disk,
// evaluated at point x. If grad is not null, adds the gradient to *grad
template<typename T>
T disk_solid_angle(const std::array<T, 3> &x,
                   std::array<T, 3> *grad,
                   const std::array<T, 3> &disk_center,
                   const std::array<T, 3> &disk_normal,
                   T disk_radius)
{
  using T3 = std::array<T, 3>;
  //     Input, double ERRTOL, the error tolerance.
  //     Relative error due to truncation is less than
  //       ERRTOL ^ 6 / (4 * (1 - ERRTOL)).
  //     Sample choices:
  //       ERRTOL   Relative truncation error less than
  //       1.D-3    3.D-19
  //       3.D-3    2.D-16
  //       1.D-2    3.D-13
  //       3.D-2    2.D-10
  //       1.D-1    3.D-7
  // double elliptic_errtol = 0.75;
  double elliptic_errtol = 0.01;

  // Formulas from Solid Angle Calculation for a Circular Disk by Paxton
  // 1959
  T3 displacement = diff(x, disk_center);  // might need to switch to diff_f
  const T3 &zHat = disk_normal;
  T L = dot(displacement, zHat);
  T3 inPlane = fma(displacement, -L, zHat);
  T r0 = len(inPlane);
  T sign = copysign(1, L);
  T aL = std::abs(L);  // use absolute value of length when evaluating solid
                       // angle to shift branch cut to the horizontal plane
  T rm = disk_radius;
  T alphaSq = 4. * r0 * rm / std::pow(r0 + rm, 2);
  T R1Sq = std::pow(L, 2) + std::pow(r0 - rm, 2);
  T Rmax = std::sqrt(std::pow(L, 2) + std::pow(r0 + rm, 2));
  T kSq = 1 - R1Sq / (Rmax * Rmax);
  T3 rHat = over(inPlane, r0);

  T F = elliptic_fm(kSq, elliptic_errtol);

  // TODO: adjust tolerance
  // ( When r0 and rm are too close, alphaSq goes to 1, which leads to a
  // singularity in elliptic_pim )
  T omega;
  if (std::abs(r0 - rm) < 1e-3) {
    omega = sign * (M_PI - 2 * aL / Rmax * F);
  }
  else if (r0 < rm) {
    omega = sign *
            (2 * M_PI - 2 * aL / Rmax * F +
             2 * aL / Rmax * (r0 - rm) / (r0 + rm) * elliptic_pim(alphaSq, kSq, elliptic_errtol));
  }
  else if (r0 > rm) {
    omega = sign * (-2 * aL / Rmax * F + 2 * aL / Rmax * (r0 - rm) / (r0 + rm) *
                                             elliptic_pim(alphaSq, kSq, elliptic_errtol));
  }

  if (grad) {
    T E = elliptic_em(kSq, elliptic_errtol);
    T Br = -2 * aL / (r0 * Rmax) * (-F + (rm * rm + r0 * r0 + aL * aL) / R1Sq * E);
    T Bz = -2 / (Rmax) * (F + (rm * rm - r0 * r0 - aL * aL) / R1Sq * E);

    for (size_t i = 0; i < 3; i++)
      (*grad)[i] += sign * Br * rHat[i] + Bz * zHat[i];
  }
  return omega;
}

// classify each loop as a polygon or as an disk
// HACK: count a polygon as a disk if it has five sides, and is
// disk-inscribed
template<typename T>
void classify_loops(const packed_float3 *const pts,
                    const packed_uint3 *const loops,
                    uint n_loops,
                    uint global_start,
                    std::vector<uint> *polygon_loops,
                    std::vector<std::array<T, 3>> *disk_centers,
                    std::vector<std::array<T, 3>> *disk_normals,
                    std::vector<T> *disk_radii)
{
  using T3 = std::array<T, 3>;

  for (uint iL = 0; iL < n_loops; iL++) {
    uint iStart = loops[iL].x - global_start;
    uint N = loops[iL].y;

    //== Test if loop is "circular", i.e. 5-sided, equidistant from center
    // point, and planar
    //     if so, store the center, normal, and radius
    T3 center, normal;
    T radius;
    bool is_circular = true;
    if (N != 5)
      is_circular = false;

    if (is_circular) {
      center = from_float3<T>(pts[iStart + N]);

      // Store displacements from center
      std::vector<T3> rx;
      rx.reserve(N);
      for (uint i = 0; i < N; i++)
        rx.push_back(diff_f(pts[iStart + i], center));

      // Test if points are equidistant from center, and store radius
      radius = len(rx[0]);
      for (uint i = 1; i < N; i++) {
        T radius_i = len(rx[i]);
        if (std::abs(radius_i - radius) >= 0.01) {
          is_circular = false;
          break;
        }
      }

      if (is_circular) {
        // Test if points are coplanar (and store normal)
        normal = cross(rx[0], rx[1]);
        normalize(normal);
        for (uint i = 1; i < N; i++) {
          int j = (i + 1) % N;
          T3 normal_i = cross(rx[i], rx[j]);

          // Check whether or not normal is parallel to normal_i
          if (std::abs(dot(normal, normal_i) - len(normal) * len(normal_i)) >= 0.01) {
            is_circular = false;
            break;
          }
        }
      }
    }

    if (is_circular) {
      // if we got here, the loop is circular. Store it as a disk
      if (disk_centers)
        disk_centers->push_back(center);
      if (disk_normals)
        disk_normals->push_back(normal);
      if (disk_radii)
        disk_radii->push_back(radius);
    }
    else {
      // if loop was not determined to be circular, record it as a polygon
      if (polygon_loops)
        polygon_loops->push_back(iL);
    }
  }
}

typedef struct solid_angle_intersection_params {
  float3 ray_P;
  float3 ray_D;
  float ray_tmin;
  float ray_tmax;
  const packed_uint3 *loops;
  const packed_float3 *pts;
  uint n_loops;
  float epsilon;
  float levelset;
  float frequency;
  int solid_angle_formula;
  bool use_grad_termination;
  int max_iterations;
  bool clip_y;
  bool capture_misses;
  bool use_overstepping;
  bool use_extrapolation;
  bool use_newton;
  bool use_quick_triangulation;
  float epsilon_loose;
  bool fixed_step_count;
} solid_angle_intersection_params;

typedef struct acceleration_stats {
  // general
  int total_iterations = 0;
  std::vector<float> ts, vals;
  // oversteps
  int successful_oversteps = 0;
  int failed_oversteps = 0;
  std::vector<float> times, omegas, Rs, rs;
  // extrapolation
  int successful_extrapolations = 0;
  std::vector<float> extrapolation_times, extrapolation_values, true_values, as, bs;
  // newton
  int n_newton_steps = 0;
  int n_steps_after_eps = 0;
  int newton_found_intersection = 0;
  std::vector<float> newton_dts, newton_dfs, newton_ts, newton_vals;
} acceleration_stats;

template<typename T>
ccl_device bool ray_nonplanar_polygon_intersect_T(const solid_angle_intersection_params &params,
                                                  ccl_private float *isect_u,
                                                  ccl_private float *isect_v,
                                                  ccl_private float *isect_t,
                                                  acceleration_stats *stats = nullptr)
{
  using T3 = std::array<T, 3>;

  T epsilon = static_cast<T>(params.epsilon);
  T frequency = static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);
  T shift = static_cast<T>(4. * M_PI);

  T3 ray_P = from_float3<T>(params.ray_P);
  T3 ray_D = from_float3<T>(params.ray_D);
  uint globalStart = params.loops[0].x;

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);

  if (params.clip_y) {
    if (ray_P[1] <= 0 && ray_D[1] <= 0) {  // moving away from clipping plane
      return false;
    }
    else if (ray_P[1] <= 0) {  // moving towards clipping plane from far side
      ray_tmin = fmax(ray_tmin, -ray_P[1] / ray_D[1]);  // p + t * d = 0
    }
    else if (ray_D[1] <= 0) {  // moving towards clipping plane from near side
      ray_tmax = fmin(ray_tmax, -ray_P[1] / ray_D[1]);
    }
  }

  std::vector<uint> polygonLoops;
  std::vector<T3> diskCenters, diskNormals;
  std::vector<T> diskRadii;

  classify_loops(params.pts,
                 params.loops,
                 params.n_loops,
                 globalStart,
                 &polygonLoops,
                 &diskCenters,
                 &diskNormals,
                 &diskRadii);

  // update min_d2, closest_point, and tangent if any closer point is present
  // on loop iL. closest_point and tangent may be null, but min_d2 must not be
  auto squared_distance_to_polygon_boundary =
      [&](uint iL, const T3 &x, T *min_d2, T3 *closest_point, T3 *tangent) {
        uint iStart = params.loops[iL].x - globalStart;
        uint N = params.loops[iL].y;

        // compute closest distance to each polygon line segment
        for (uint i = 0; i < N; i++) {
          T3 p1 = from_float3<T>(params.pts[iStart + i]);
          T3 p2 = from_float3<T>(params.pts[iStart + (i + 1) % N]);
          T3 m = diff(p2, p1);
          T3 v = diff(x, p1);
          // dot = |a|*|b|cos(theta) * n, isolating |a|sin(theta)
          T t = fmin(fmax(dot(m, v) / dot(m, m), (T)0.), (T)1.);
          T d2 = len_squared(fma(v, -t, m));
          // if closestPoint is not null, update it to track closest point
          if (closest_point && d2 < *min_d2)
            *closest_point = fma(p1, t, m);
          if (tangent && d2 < *min_d2)
            *tangent = diff(p2, p1);
          *min_d2 = fmin(*min_d2, d2);
        }
        if (tangent)
          normalize(*tangent);
      };

  // update min_d2, closest_point, and tangent if any closer point is present
  // on disk iD. closest_point and tangent may be null, but min_d2 must not be
  auto squared_distance_to_disk_boundary =
      [&](uint iD, const T3 &x, T *min_d2, T3 *closest_point, T3 *tangent) {
        T3 displacement = diff(x, diskCenters[iD]);
        T L = dot(displacement, diskNormals[iD]);
        T r0 = len(fma(displacement, -L, diskNormals[iD]));
        T rm = diskRadii[iD];
        T d2 = static_cast<T>(std::pow(rm - r0, 2) + std::pow(L, 2));
        *min_d2 = fmin(*min_d2, d2);
        if (d2 < *min_d2 && closest_point) {
          T3 rHat = fma(displacement, -L, diskNormals[iD]);
          *closest_point = fma(diskCenters[iD], rm / r0, rHat);
        }
        if (d2 < *min_d2 && tangent) {
          *tangent = cross(diskNormals[iD], displacement);
          normalize(*tangent);
        }
      };

  auto distance_to_boundary = [&](const T3 &x, T3 *closest_point, T3 *tangent) -> T {
    const T infinity = 100000.;
    T min_d2 = infinity;
    // TODO: TKTKTKT closest point logic is broken
    for (uint iL : polygonLoops)
      squared_distance_to_polygon_boundary(iL, x, &min_d2, closest_point, tangent);
    for (uint iD = 0; iD < diskCenters.size(); iD++)
      squared_distance_to_disk_boundary(iD, x, &min_d2, closest_point, tangent);
    return std::sqrt(min_d2);
  };

  // Computes a conservative step size via Harnack bounds, using the value
  // fx at the current point, the radius R of a ball over which the function
  // is harmonic, the values lo_bound and up_bound of the closest level sets
  // above/ below the current value, and a shift that makes the harmonic
  // function positive within this ball.
  auto get_max_step = [&](T fx, T R, T lo_bound, T up_bound, T shift) -> T {
    T w = (fx + shift) / (up_bound + shift);
    T v = (fx + shift) / (lo_bound + shift);
    T lo_r = -R / 2 * (v + 2 - std::sqrt(v * v + 8 * v));
    T up_r = R / 2 * (w + 2 - std::sqrt(w * w + 8 * w));

    return std::min(lo_r, up_r);
  };

  // sums solid angle of polygon loops and disks and adds gradient to grad
  auto total_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    int grad_mode = params.use_grad_termination ? 0 : 3;
    grad = T3{0, 0, 0};  // zero out gradient before computing
    T omega = polygon_solid_angle(params.pts,
                                  params.loops,
                                  globalStart,
                                  polygonLoops,
                                  x,
                                  params.solid_angle_formula,
                                  &grad,
                                  grad_mode,
                                  params.use_quick_triangulation);

    for (uint iD = 0; iD < diskCenters.size(); iD++)
      omega += disk_solid_angle(x, &grad, diskCenters[iD], diskNormals[iD], diskRadii[iD]);

    return omega;
  };

  /* Find intersection with Harnack tracing */

  T t = ray_tmin;
  int iter = 0;
  T lo_bound = 0;
  T hi_bound = static_cast<T>(4. * M_PI);

  T ld = len(ray_D);

  auto report_stats = [&]() {
    if (stats) {
      stats->total_iterations = iter;
    }
  };

  T t_overstep = 0;
  T t_prev = -1, val_prev = 0;
  static bool exceeded_max = false;
  bool inside_loose_shell = false;
  while (t < ray_tmax) {
    T3 pos = fma(ray_P, t + t_overstep, ray_D);

    // If we've exceeded the maximum number of iterations,
    // print a warning
    if (iter > params.max_iterations) {
      if (!exceeded_max) {
        exceeded_max = true;
        printf(
            "Warning: exceeded maximum number of Harnack "
            "iterations.\n");
      }

      *isect_t = static_cast<float>(t + t_overstep);
      *isect_v = static_cast<float>(((T)iter) / ((T)params.max_iterations));
      report_stats();
      if (params.capture_misses) {
        T3 pos = fma(ray_P, t, ray_D);
        T3 grad{0, 0, 0};
        T omega = total_solid_angle(pos, grad);
        *isect_u = static_cast<float>(omega / static_cast<T>(4. * M_PI));
        return true;
      }
      else {
        return false;
      }
    }

    T3 grad{0, 0, 0};
    T omega = total_solid_angle(pos, grad);

    // To get the most aggressive Harnack bound, we first find a
    // representative of the solid angle, shifted by the target level
    // set value, within the range [0,4π).  Only then do we apply the
    // shift.
    T val = glsl_mod(omega - levelset, static_cast<T>(4. * M_PI));

    if (stats) {
      stats->ts.push_back(static_cast<float>(t + t_overstep));
      stats->vals.push_back(static_cast<float>(val));
    }

    if (frequency > 0) {
      lo_bound = 0;
      hi_bound = static_cast<T>(4. * M_PI);
      // add four additional level sets
      if (val < frequency * hi_bound) {
        hi_bound = frequency * hi_bound;
      }
      else if (val < 2 * frequency * hi_bound) {
        lo_bound = frequency * hi_bound;
        hi_bound = 2 * frequency * hi_bound;
      }
      else if (val < hi_bound - 2 * frequency * hi_bound) {
        lo_bound = 2 * frequency * hi_bound;
        hi_bound = hi_bound - 2 * frequency * hi_bound;
      }
      else if (val < hi_bound - frequency * hi_bound) {
        lo_bound = hi_bound - 2 * frequency * hi_bound;
        hi_bound = hi_bound - frequency * hi_bound;
      }
      else {
        lo_bound = hi_bound - frequency * hi_bound;
      }
    }

    // Calculate the radius of a ball around the current point over
    // which we know the function is harmonic.  An easy way to identify
    // such a ball is to restrict to the sphere touching the closest
    // point on the polygon boundary.
    T3 closestPoint;
    T R = distance_to_boundary(pos, &closestPoint, nullptr);

    // Compute a conservative step size based on the Harnack bound.
    T r = get_max_step(val, R, lo_bound, hi_bound, shift) / ld;

    if (stats) {
      stats->times.push_back(static_cast<float>(t + t_overstep));
      stats->omegas.push_back(static_cast<float>(val));
      stats->Rs.push_back(static_cast<float>(R));
      stats->rs.push_back(static_cast<float>(r));
    }

    if (r >= t_overstep) {  // commit to step
      // If we're close enough to the level set, or we've exceeded the
      // maximum number of iterations, assume there's a hit.
      if (!params.fixed_step_count &&
          (close_to_zero(val, lo_bound, hi_bound, epsilon, grad, params.use_grad_termination) ||
           R < epsilon || iter > params.max_iterations))
      {
        // if (R < epsilon)   grad = pos - closestPoint; // TODO:
        // this?
        *isect_t = static_cast<float>(t + t_overstep);
        *isect_u = static_cast<float>(omega / static_cast<T>(4. * M_PI));
        *isect_v = static_cast<float>(((T)iter) / ((T)params.max_iterations));
        report_stats();
        return true;
      }

      if (!params.fixed_step_count &&
          close_to_zero(
              val, lo_bound, hi_bound, (T)params.epsilon_loose, grad, params.use_grad_termination))
      {
        // try newton's method when you first enter the
        // epsilon_loose shell
        if (params.use_newton && !inside_loose_shell) {
          auto f = [&](T t) -> T {  // also updates grad
            T3 pos = fma(ray_P, t + t_overstep, ray_D);
            T omega = total_solid_angle(pos, grad);
            return glsl_mod(omega - levelset, static_cast<T>(4. * M_PI));

          };

          // if you're too close to the curve, do a weird
          // approximation thing
          if (R < 0.25) {
            T t_radial = t + t_overstep;
            T3 tangent;
            T R = distance_to_boundary(pos, nullptr, &tangent);
            T3 rd_planar = fma(ray_D, -dot(ray_D, tangent), tangent);

            for (int i = 0; i < 5; i++) {
              T val_err = (val < static_cast<T>(2. * M_PI) ? -val : static_cast<T>(4. * M_PI) - val);
              T alpha = -atan2(dot(tangent, cross(rd_planar, grad)), dot(rd_planar, grad));
              T dt = R * sin(val_err) / cos(val_err - alpha);
              // todo: clever trig to use tan(val) somehow?
              T old_val = val;
              val = f(t_radial + dt);
              T new_val_err = (val < static_cast<T>(2. * M_PI) ? -val : static_cast<T>(4. * M_PI) - val);
              while (val_err * new_val_err < 0 && dt > 1e-8) {
                dt /= 2;
                val = f(t_radial + dt);
                new_val_err = (val < static_cast<T>(2. * M_PI) ? -val : static_cast<T>(4. * M_PI) - val);
                if (stats) {
                  stats->n_newton_steps++;
                }
              }
              t_radial += dt;
              bool close = close_to_zero(
                  val, lo_bound, hi_bound, epsilon, grad, params.use_grad_termination);

              // if (i == 0) {
              //     std::cout << "radial approx: ";
              // } else {
              //     std::cout << "             : ";
              // }
              // std::cout << std::fixed;
              // std::cout.precision(5);
              // //=== newton
              // std::cout
              //     << "| t  = " << std::setw(8) << t +
              //     t_overstep
              //     << "  f  = " << std::setw(8) << old_val
              //     << "  4π = " << std::setw(8) << 4. * M_PI
              //     << "  dt = " << std::setw(8) << dt
              //     << "  f' = " << std::setw(8) << val
              //     << " close: " << (close ? "true" :
              //     "false");
              // std::cout << std::endl;
              if (close) {
                *isect_t = static_cast<float>(t_radial);
                *isect_u = static_cast<float>(omega / static_cast<T>(4. * M_PI));
                *isect_v = static_cast<float>(((T)iter) / ((T)params.max_iterations));
                report_stats();
                return true;
              }
            }
            // t = t_radial;
          }
          else {  // otherwise, run a few rounds of Newton's
                  // method
            T t_newton = t + t_overstep;
            for (int i = 0; i < 8; i++) {
              T df = dot(ray_D, grad);
              T dt = -(val < static_cast<T>(2. * M_PI) ? val : val - static_cast<T>(4. * M_PI)) / df;

              t_newton += dt;
              val = f(t_newton);

              if (stats) {
                stats->newton_dfs.push_back(static_cast<float>(df));
                stats->newton_dts.push_back(static_cast<float>(dt));
                stats->newton_ts.push_back(static_cast<float>(t_newton));
                stats->newton_vals.push_back(static_cast<float>(val));
                stats->n_newton_steps++;
              }
              if (close_to_zero(
                      val, lo_bound, hi_bound, epsilon, grad, params.use_grad_termination))
              {
                *isect_t = static_cast<float>(t_newton);
                *isect_u = static_cast<float>(omega / static_cast<T>(4. * M_PI));
                *isect_v = static_cast<float>(((T)iter) / ((T)params.max_iterations));
                report_stats();
                return true;
              }
            }
          }
        }
        inside_loose_shell = true;
      }
      else {
        inside_loose_shell = false;
      }

      if (!params.fixed_step_count && params.use_extrapolation) {
        // model val(t) = a t + b
        float a = (val - val_prev) / (t + t_overstep - t_prev);
        float b = val - a * t;
        float t_low = (lo_bound - b) / a;
        float t_high = (hi_bound - b) / a;
        float t_test = -1.;
        if (t_low > 0. && t_high > 0.) {
          t_test = fmin(t_low, t_high);
        }
        else if (t_low > 0.) {
          t_test = t_low;
        }
        else if (t_high > 0.) {
          t_test = t_high;
        }

        T3 e_pos = fma(ray_P, t_test, ray_D);
        T3 e_grad;
        T e_omega = total_solid_angle(e_pos, e_grad);
        T e_val = glsl_mod(omega - levelset, static_cast<T>(4. * M_PI));

        if (stats) {
          stats->as.push_back(a);
          stats->bs.push_back(b);
          stats->extrapolation_times.push_back(t_test);
          stats->extrapolation_values.push_back(a * t_test + b);
          stats->true_values.push_back(e_val);
        }
        if (0. <= t_test && t_test <= 4. * (t + t_overstep - t_prev)) {
          if (close_to_zero(
                  e_val, lo_bound, hi_bound, epsilon, e_grad, params.use_grad_termination))
          {
            if (stats)
              stats->successful_extrapolations++;
            return true;
          }
        }

        t_prev = t + t_overstep;
        val_prev = val;
      }

      t += t_overstep + r;
      if (params.use_overstepping)
        t_overstep = r * static_cast<T>(.75);
      if (params.use_overstepping && stats)
        stats->successful_oversteps++;
    }
    else {  // step back and try again
      t_overstep = 0;
      if (stats)
        stats->failed_oversteps++;
    }
    iter++;
    if (inside_loose_shell && stats)
      stats->n_steps_after_eps++;
  }

  *isect_t = static_cast<float>(t);
  *isect_v = static_cast<float>(((T)iter) / ((T)params.max_iterations));
  report_stats();
  if (params.capture_misses) {
    T3 pos = fma(ray_P, t, ray_D);
    T3 grad{0, 0, 0};
    T omega = total_solid_angle(pos, grad);
    *isect_u = static_cast<float>(omega / static_cast<T>(4. * M_PI));
    return true;
  }
  else {
    return false;
  }
}

template<typename T>
ccl_device bool newton_intersect_T(const solid_angle_intersection_params &params,
                                   ccl_private float *isect_u,
                                   ccl_private float *isect_v,
                                   ccl_private float *isect_t,
                                   ccl_private float *t_start = nullptr,
                                   acceleration_stats *stats = nullptr,
                                   int verbosity = 0)
{

  using T3 = std::array<T, 3>;

  T epsilon = static_cast<T>(params.epsilon);
  T frequency = static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);
  T shift = static_cast<T>(4. * M_PI);

  T3 ray_P = from_float3<T>(params.ray_P);
  T3 ray_D = from_float3<T>(params.ray_D);
  uint globalStart = params.loops[0].x;

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);

  T lo_bound = 0;
  T hi_bound = static_cast<T>(4. * M_PI);

  std::vector<uint> polygonLoops;
  std::vector<T3> diskCenters, diskNormals;
  std::vector<T> diskRadii;

  classify_loops(params.pts,
                 params.loops,
                 params.n_loops,
                 globalStart,
                 &polygonLoops,
                 &diskCenters,
                 &diskNormals,
                 &diskRadii);

  // sums solid angle of polygon loops and disks and adds gradient to grad
  auto total_solid_angle = [&](const T3 &x, T3 *grad) -> T {
    if (grad)
      *grad = T3{0, 0, 0};  // zero out gradient before computing
    int grad_mode = 0;
    T omega = polygon_solid_angle(params.pts,
                                  params.loops,
                                  globalStart,
                                  polygonLoops,
                                  x,
                                  params.solid_angle_formula,
                                  grad,
                                  grad_mode,
                                  params.use_quick_triangulation);

    for (uint iD = 0; iD < diskCenters.size(); iD++)
      omega += disk_solid_angle(x, grad, diskCenters[iD], diskNormals[iD], diskRadii[iD]);

    return omega;
  };

  T3 grad_f;
  auto f = [&](T t, T3 *grad_f) -> T {
    T3 pos = fma(ray_P, t, ray_D);
    T omega = total_solid_angle(pos, grad_f);
    return glsl_mod(omega - levelset, static_cast<T>(4. * M_PI));
  };

  int iter = 0;
  T t = t_start ? *t_start : 0;  // TODO: random starting time?
  T val = f(t, &grad_f);

  if (verbosity >= 1)
    std::cout << ">>> initial grad: " << grad_f << std::flush << std::endl;

  auto report_stats = [&]() {
    if (stats) {
      stats->total_iterations = iter;
    }
  };

  for (int iN = 0; iN < 8; iN++) {
    T df = dot(ray_D, grad_f);
    T f_err = (val < static_cast<T>(2. * M_PI) ? val : val - static_cast<T>(4. * M_PI));
    T dt = -f_err / df;
    dt = fmin(fmax(dt, -2.), 2.);  // clamp to [-2, 2]

    if (verbosity >= 1) {
      auto pr = std::setprecision(4);
      double fpi = 4. * M_PI;
      std::cout << std::setfill(' ') << std::setw(3) << iN << "| t = " << std::setw(8)
                << std::fixed << pr << t << "  f = " << std::setw(8) << std::fixed << pr << val
                << " 4π = " << std::setw(8) << std::fixed << pr << fpi << " dt = " << std::setw(8)
                << std::fixed << pr << dt << " ferr = " << std::setw(8) << std::fixed << pr
                << f_err << " df = " << std::setw(8) << std::fixed << pr << df
                << " pos = " << fma(ray_P, t, ray_D) << " grad_f = " << grad_f << std::endl;
    }

    t += dt;
    val = f(t, &grad_f);

    if (stats) {
      stats->newton_dfs.push_back(static_cast<float>(df));
      stats->newton_dts.push_back(static_cast<float>(dt));
      stats->newton_ts.push_back(static_cast<float>(t));
      stats->newton_vals.push_back(static_cast<float>(val));
      stats->n_newton_steps++;
    }
    if (close_to_zero(val, lo_bound, hi_bound, epsilon, grad_f, params.use_grad_termination)) {
      *isect_t = static_cast<float>(t);
      *isect_u = static_cast<float>(val / static_cast<T>(4. * M_PI));
      *isect_v = static_cast<float>(((T)iter) / ((T)params.max_iterations));
      report_stats();
      return true;
    }
  }

  return false;
}

template<typename T>
ccl_device bool bisection_intersect_T(const solid_angle_intersection_params &params,
                                      ccl_private float *isect_u,
                                      ccl_private float *isect_v,
                                      ccl_private float *isect_t,
                                      ccl_private float *t_start = nullptr,
                                      acceleration_stats *stats = nullptr,
                                      int verbosity = 0)
{

  using T3 = std::array<T, 3>;

  T epsilon = static_cast<T>(params.epsilon);
  T frequency = static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);
  T shift = static_cast<T>(4. * M_PI);

  T3 ray_P = from_float3<T>(params.ray_P);
  T3 ray_D = from_float3<T>(params.ray_D);
  uint globalStart = params.loops[0].x;

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);

  T lo_bound = 0;
  T hi_bound = static_cast<T>(4. * M_PI);

  std::vector<uint> polygonLoops;
  std::vector<T3> diskCenters, diskNormals;
  std::vector<T> diskRadii;

  classify_loops(params.pts,
                 params.loops,
                 params.n_loops,
                 globalStart,
                 &polygonLoops,
                 &diskCenters,
                 &diskNormals,
                 &diskRadii);

  // sums solid angle of polygon loops and disks and adds gradient to grad
  auto total_solid_angle = [&](const T3 &x, T3 *grad) -> T {
    if (grad)
      *grad = T3{0, 0, 0};  // zero out gradient before computing
    int grad_mode = 0;
    T omega = polygon_solid_angle(params.pts,
                                  params.loops,
                                  globalStart,
                                  polygonLoops,
                                  x,
                                  params.solid_angle_formula,
                                  grad,
                                  grad_mode,
                                  params.use_quick_triangulation);

    for (uint iD = 0; iD < diskCenters.size(); iD++)
      omega += disk_solid_angle(x, grad, diskCenters[iD], diskNormals[iD], diskRadii[iD]);

    return omega;
  };

  // mod and shift solid angle function so that we're looking for a zero of f
  auto f = [&](T t) -> T {
    T3 pos = fma(ray_P, t, ray_D);
    T omega = total_solid_angle(pos, nullptr);
    return glsl_mod(omega, static_cast<T>(4. * M_PI)) - levelset;
  };

  T ta = t_start ? *t_start : 0;  // TODO: random starting time?
  T fa = f(ta);
  T tb = .125;
  T fb = f(tb);

  size_t i_double = 0;
  while (fa * fb > 0 && i_double < 10) {
    tb *= 2;
    fb = f(tb);
    i_double++;
  }

  if (fa * fb > 0)
    return false;

  if (verbosity >= 1)
    std::cout << ">>> initial bounds: [ " << ta << ", " << tb << " ]" << std::endl;

  int iter = 0;
  auto report_stats = [&]() {
    if (stats) {
      stats->total_iterations = iter;
    }
  };

  for (; iter < 100; iter++) {
    T tc = ta + (tb - ta) / 2.;
    T fc = f(tc);

    if (std::abs(fc) < params.epsilon || (false && (tb - ta < 1e-8))) {
      T3 pos = fma(ray_P, tc, ray_D);
      T omega = total_solid_angle(pos, nullptr);
      T val = glsl_mod(omega, static_cast<T>(4. * M_PI));
      *isect_t = static_cast<float>(tc);
      *isect_u = static_cast<float>(val / static_cast<T>(4. * M_PI));
      *isect_v = static_cast<float>(((T)iter) / ((T)params.max_iterations));
      report_stats();
      return true;
    }

    // if (tb - ta < params.epsilon * params.epsilon) break;

    if (fa * fc < 0) {  // intersection in first interfal
      tb = tc;
      fb = fc;
    }
    else {  // intersection in second interfal
      ta = tc;
      fa = fc;
    }

    if (verbosity >= 1) {
      auto pr = std::setprecision(4);
      double dt = tb - ta;
      std::cout << std::setfill(' ') << std::setw(3) << iter << "| ta = " << std::setw(8)
                << std::fixed << pr << ta << "  tb = " << std::setw(8) << std::fixed << pr << tb
                << "  δt = " << std::setw(8) << std::fixed << pr << dt << "  fa = " << std::setw(8)
                << std::fixed << pr << fa << "  fb = " << std::setw(8) << std::fixed << pr << fb
                << std::endl;
    }
  }

  return false;
}

/* Normal on nonplanar polygon. */
// TODO: move to better-named file
// TODO: deduplicate with version in harnack tracing code?

// int grad_mode; // 0 = nicole formula,
//                   1 = finite diff,
//                   2 = architecture formula

template<typename T>
ccl_device float3 ray_nonplanar_polygon_normal_T(const float3 pf,
                                                 const packed_uint3 *loops,
                                                 const packed_float3 *pts,
                                                 const uint n_loops,
                                                 int grad_mode = 0)
{
  using T3 = std::array<T, 3>;
  /* auto smoothstep = [](T x, T start, T end) -> T {  //
   * https://docs.gl/sl4/smoothstep */
  /*   // normalize and clamp to [0, 1] */
  /*   x = fmin(fmax(((x - start) / (end - start)), 0), 1); */

  /*   return x * x * (3. - 2. * x); */
  /* }; */

  // find solid angle gradient and closest point on boundary curve
  uint globalStart = loops[0].x;
  T3 p = from_float3<T>(pf);

  // classify each loop as a polygon or as an disk
  // HACK: count a polygon as a disk if it has five sides, and is
  // disk-inscribed
  std::vector<uint> polygonLoops;
  std::vector<T3> diskCenters, diskNormals;
  std::vector<T> diskRadii;

  classify_loops(
      pts, loops, n_loops, globalStart, &polygonLoops, &diskCenters, &diskNormals, &diskRadii);

  auto compute_solid_angle = [&](const T3 &x, T3 *grad) -> T {
    // polygon contribution
    T omega = triangulated_solid_angle(
        pts, loops, globalStart, polygonLoops, x, grad, grad_mode, false);
    // disk contribution
    for (uint iD = 0; iD < diskCenters.size(); iD++)
      omega += disk_solid_angle(x, grad, diskCenters[iD], diskNormals[iD], diskRadii[iD]);

    return omega;
  };

  T3 grad{0, 0, 0};
  switch (grad_mode) {
    case 0:  // nicole formula
      compute_solid_angle(p, &grad);
      break;
    case 1:  // finite differences
    {
      T h = 0.000001;
      T omega = compute_solid_angle(p, nullptr);
      T omega_x = compute_solid_angle(T3{p[0] + h, p[1], p[2]}, nullptr);
      T omega_y = compute_solid_angle(T3{p[0], p[1] + h, p[2]}, nullptr);
      T omega_z = compute_solid_angle(T3{p[0], p[1], p[2] + h}, nullptr);

      T fd_df_dx = (omega_x - omega) / h;
      T fd_df_dy = (omega_y - omega) / h;
      T fd_df_dz = (omega_z - omega) / h;

      grad = T3{fd_df_dx, fd_df_dy, fd_df_dz};
      break;
    }
    case 2:  // architecture
    {
      compute_solid_angle(p, &grad);
      break;
    }
  }

  // blend solid angle gradient and offset based on distance to boundary
  /* T s = smoothstep(min_d2, 0, HARNACK_EPS); */
  /* s = 0; */
  /* grad[0] = (1 - s) * offset[0] + s * grad[0]; */
  /* grad[1] = (1 - s) * offset[1] + s * grad[1]; */
  /* grad[2] = (1 - s) * offset[2] + s * grad[2]; */

  T grad_norm = len(grad);
  grad[0] /= grad_norm;
  grad[1] /= grad_norm;
  grad[2] /= grad_norm;

  return make_float3(grad[0], grad[1], grad[2]);
}

template<typename T>
ccl_device float3 ray_spherical_harmonic_normal_T(const float3 pf, uint m, int l)
{
  using T3 = std::array<T, 3>;

  T3 p = from_float3<T>(pf);

  // Return the value of this harmonic polynomial at an evaluation point
  // p,
  auto evaluatePolynomial = [&](const T3 &p) -> T { return evaluateSphericalHarmonic(l, m, p); };

  // TODO: take in analytic expressions?
  // finite difference gradient from
  // https://iquilezles.org/articles/normalsSDF
  auto calculateGradient = [&](const T3 &p) -> T3 {
    const double eps = 0.05;
    T q0 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{1, -1, -1}));
    T q1 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{-1, -1, 1}));
    T q2 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{-1, 1, -1}));
    T q3 = evaluatePolynomial(fma(p, 0.5773 * eps, T3{1, 1, 1}));
    return T3{static_cast<T>(.5773) * (q3 + q0 - q1 - q2),
              static_cast<T>(.5773) * (q3 - q0 - q1 + q2),
              static_cast<T>(.5773) * (q3 - q0 + q1 - q2)};
  };

  T3 grad = calculateGradient(p);
  normalize(grad);

  return make_float3(grad[0], grad[1], grad[2]);
}

typedef struct gyroid_intersection_params {
  float3 ray_P;
  float3 ray_D;
  float ray_tmin;
  float ray_tmax;

  float R;
  float frequency;
  float epsilon;
  float levelset;
  int max_iterations;
  bool use_overstepping;
  bool use_grad_termination;
} gyroid_intersection_params;

// Return the value of the gyroid at an evaluation point s*p,
template<typename T> T evaluateGyroid(const std::array<T, 3> &p, T s)
{
  T x = s * p[0];
  T y = s * p[1];
  T z = s * p[2];
  return sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x);
}

// Return the gradient of gyroid(s*p) with respect to p, evaluated at evaluation
// point s*p,
template<typename T> std::array<T, 3> evaluateGyroidGradient(const std::array<T, 3> &p, T s)
{
  T x = s * p[0];
  T y = s * p[1];
  T z = s * p[2];
  return {s * cos(x) * cos(y) - s * sin(z) * sin(x),
          s * cos(y) * cos(z) - s * sin(x) * sin(y),
          s * cos(z) * cos(x) - s * sin(y) * sin(z)};
}

template<typename T>
ccl_device bool ray_gyroid_intersect_T(const gyroid_intersection_params &params,
                                       ccl_private float *isect_u,
                                       ccl_private float *isect_v,
                                       ccl_private float *isect_t,
                                       acceleration_stats *stats = nullptr)
{
  using T3 = std::array<T, 3>;
  T epsilon = static_cast<T>(params.epsilon);
  T scale = 1. / static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);

  T3 ray_P = from_float3<T>(params.ray_P);
  T3 ray_D = from_float3<T>(params.ray_D);

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);

  T radius = static_cast<T>(params.R);
  T outer_radius = static_cast<T>(1.25) * radius;  // TODO: make configurable?

  T unit_shift = 3.;

  auto distance_to_boundary = [&](const T3 &x) -> T { return outer_radius - len(x); };

  // find safe step size derived from 4D Harnack inequality
  auto getMaxStep4D = [](T fx, T R, T levelset, T shift) -> T {
    T a = (fx + shift) / (levelset + shift);
    T u = std::pow(3. * sqrt(3. * std::pow(a, 3.) + 81. * std::pow(a, 2.)) + 27. * a, 1. / 3.);
    return R * std::abs(u / 3. - a / u - 1.);
  };

  auto distanceToLevelset = [&](T f, T levelset, const T3 &grad) -> T {
    T scaling = params.use_grad_termination ? fmax(len(grad), epsilon) : 1;
    return std::abs(f - levelset) / scaling;
  };

  // check if ray intersects sphere. If so, store intersection times (t0 < t1)
  T t0, t1;
  bool hit_sphere = intersect_sphere(ray_P, ray_D, radius, &t0, &t1);

  // if we miss the sphere, there cannot be an intersection with the levelset
  // within the sphere
  if (!hit_sphere)
    return false;

  // start at first sphere intersection if it is ahead of us
  T t = fmax(t0, ray_tmin);
  T tMax = fmin(ray_tmax, t1);  // only trace until second sphere intersection
  T ld = len(ray_D);

  int iter = 0;
  T t_overstep = 0.;

  static bool exceeded_max = false;

  // Until we reach the maximum ray distance
  while (t < tMax) {
    // If we've exceeded the maximum number of iterations, print a warning
    if (iter >= params.max_iterations) {
      if (!exceeded_max) {
        exceeded_max = true;
        printf(
            "Warning: exceeded maximum number of Harnack "
            "iterations.\n");
      }
    }

    T3 pos = fma(ray_P, t + t_overstep, ray_D);
    T f = evaluateGyroid(pos, scale);
    T3 grad;
    if (params.use_grad_termination)
      grad = evaluateGyroidGradient(pos, scale);

    T R = distance_to_boundary(pos);
    T shift = std::exp(sqrt(2.) * R) * unit_shift;  // scale shift for 4D sphere

    T r = getMaxStep4D(f, R, levelset, shift) / ld;  // safe step size

    if (r >= t_overstep) {  // commit to step

      T dist = distanceToLevelset(f, levelset, grad);
      // If we're close enough to the level set, return a hit.
      if (dist < epsilon) {
        *isect_t = static_cast<float>(t + t_overstep);
        *isect_u = static_cast<float>(f);
        *isect_v = static_cast<float>(((T)iter) / ((T)params.max_iterations));
        return true;
      }

      t += t_overstep + r;
      if (params.use_overstepping)
        t_overstep = r * static_cast<T>(.75);
    }
    else {  // step back and try again
      t_overstep = 0;
    }

    iter++;
  }

  return false;  // no intersection
}

template<typename T>
ccl_device float3 ray_gyroid_normal_T(const float3 pf, float R, float frequency)
{
  using T3 = std::array<T, 3>;

  T3 p = from_float3<T>(pf);
  T radius = static_cast<T>(R);
  T scale = 1. / static_cast<T>(frequency);

  // special case for points on boundary
  if (std::abs(len_squared(p) - radius * radius) < (float)1e-5) {
    normalize(p);
    return to_float3(p);
  }
  else {
    T3 grad = evaluateGyroidGradient(p, scale);
    normalize(grad);
    return to_float3(grad);
  }
}

template<typename T>
ccl_device bool newton_intersect_gyroid_T(const gyroid_intersection_params &params,
                                          ccl_private float *isect_u,
                                          ccl_private float *isect_v,
                                          ccl_private float *isect_t,
                                          ccl_private float *t_start = nullptr,
                                          acceleration_stats *stats = nullptr,
                                          int verbosity = 0)
{

  // TODO: clip to sphere
  using T3 = std::array<T, 3>;
  T epsilon = static_cast<T>(params.epsilon);
  T scale = 1. / static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);

  T3 ray_P = from_float3<T>(params.ray_P);
  T3 ray_D = from_float3<T>(params.ray_D);

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);
  T radius = static_cast<T>(params.R);

  // check if ray intersects sphere. If so, store intersection times (t0 < t1)
  T t0, t1;
  bool hit_sphere = intersect_sphere(ray_P, ray_D, radius, &t0, &t1);

  // if we miss the sphere, there cannot be an intersection with the levelset
  // within the sphere
  if (!hit_sphere)
    return false;

  // start at first sphere intersection if it is ahead of us
  T tInit = fmax(t0, ray_tmin);
  T tMax = fmin(ray_tmax, t1);  // only trace until second sphere intersection

  auto distanceToLevelset = [&](T f, T levelset, const T3 &grad) -> T {
    T scaling = params.use_grad_termination ? fmax(len(grad), epsilon) : 1;
    return std::abs(f - levelset) / scaling;
  };

  T3 grad_f;
  auto f = [&](T t, T3 *grad_f) -> T {
    T3 pos = fma(ray_P, t, ray_D);
    T val = evaluateGyroid(pos, scale);
    if (grad_f)
      *grad_f = evaluateGyroidGradient(pos, scale);
    return val;
  };

  int iter = 0;
  T t = t_start ? *t_start : tInit;  // TODO: random starting time?
  T val = f(t, &grad_f);

  if (verbosity >= 1)
    std::cout << ">>> initial grad: " << grad_f << std::flush << std::endl;

  for (int iN = 0; iN < 8; iN++) {
    T df = dot(ray_D, grad_f);
    T f_err = val - levelset;
    T dt = -f_err / df;
    dt = fmin(fmax(dt, -2.), 2.);  // clamp to [-2, 2]

    if (verbosity >= 1) {
      auto pr = std::setprecision(4);
      double fpi = 4. * M_PI;
      std::cout << std::setfill(' ') << std::setw(3) << iN << "| t = " << std::setw(8)
                << std::fixed << pr << t << "  f = " << std::setw(8) << std::fixed << pr << val
                << " 4π = " << std::setw(8) << std::fixed << pr << fpi << " dt = " << std::setw(8)
                << std::fixed << pr << dt << " ferr = " << std::setw(8) << std::fixed << pr
                << f_err << " df = " << std::setw(8) << std::fixed << pr << df
                << " pos = " << fma(ray_P, t, ray_D) << " grad_f = " << grad_f << std::endl;
    }

    t += dt;
    val = f(t, &grad_f);

    if (distanceToLevelset(val, levelset, grad_f) < epsilon) {
      if (t < tInit || t > tMax)
        return false;  // root is out of bounds
      *isect_t = t;
      *isect_u = val;
      *isect_v = ((T)iter) / ((T)params.max_iterations);
      return true;
    }
  }

  return false;
}

template<typename T>
ccl_device bool bisection_intersect_gyroid_T(const gyroid_intersection_params &params,
                                             ccl_private float *isect_u,
                                             ccl_private float *isect_v,
                                             ccl_private float *isect_t,
                                             ccl_private float *t_start = nullptr,
                                             acceleration_stats *stats = nullptr,
                                             int verbosity = 0)
{
  // TODO: clip to sphere
  using T3 = std::array<T, 3>;
  T epsilon = static_cast<T>(params.epsilon);
  T scale = 1. / static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);

  T3 ray_P = from_float3<T>(params.ray_P);
  T3 ray_D = from_float3<T>(params.ray_D);

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);
  T radius = static_cast<T>(params.R);

  auto distanceToLevelset = [&](T f, T levelset, const T3 &grad) -> T {
    T scaling = params.use_grad_termination ? fmax(len(grad), epsilon) : 1;
    return std::abs(f - levelset) / scaling;
  };

  // check if ray intersects sphere. If so, store intersection times (t0 < t1)
  T t0, t1;
  bool hit_sphere = intersect_sphere(ray_P, ray_D, radius, &t0, &t1);

  // if we miss the sphere, there cannot be an intersection with the levelset
  // within the sphere
  if (!hit_sphere)
    return false;

  // start at first sphere intersection if it is ahead of us
  T t = fmax(t0, ray_tmin);
  T tMax = fmin(ray_tmax, t1);  // only trace until second sphere intersection

  // shift function so that we're looking for a zero of f
  auto f = [&](T t) -> T {
    T3 pos = fma(ray_P, t, ray_D);
    T val = evaluateGyroid(pos, scale);
    return val - levelset;
  };

  T ta = t;
  T fa = f(ta);
  T sb = .125;
  T tb = ta + sb;
  T fb = f(tb);

  size_t i_double = 0;
  while (fa * fb > 0 && tb < tMax) {
    sb *= 2;
    tb = ta + sb;
    fb = f(tb);
  }

  if (fa * fb > 0)
    return false;

  if (verbosity >= 1)
    std::cout << ">>> initial bounds: [ " << ta << ", " << tb << " ]" << std::endl;

  int iter = 0;

  for (; iter < 100; iter++) {
    T tc = ta + (tb - ta) / 2.;
    T fc = f(tc);

    T3 grad_f;
    if (params.use_grad_termination) {
      T3 pos = fma(ray_P, tc, ray_D);
      grad_f = evaluateGyroidGradient(pos, scale);
    }

    if (distanceToLevelset(fc, 0, grad_f) < params.epsilon) {
      T3 pos = fma(ray_P, tc, ray_D);
      T val = evaluateGyroid(pos, scale);
      *isect_t = tc;
      *isect_u = val;
      *isect_v = ((T)iter) / ((T)params.max_iterations);
      return true;
    }

    // if (tb - ta < params.epsilon * params.epsilon) break;

    if (fa * fc < 0) {  // intersection in first interfal
      tb = tc;
      fb = fc;
    }
    else {  // intersection in second interfal
      ta = tc;
      fa = fc;
    }

    if (verbosity >= 1) {
      auto pr = std::setprecision(4);
      double dt = tb - ta;
      std::cout << std::setfill(' ') << std::setw(3) << iter << "| ta = " << std::setw(8)
                << std::fixed << pr << ta << "  tb = " << std::setw(8) << std::fixed << pr << tb
                << "  δt = " << std::setw(8) << std::fixed << pr << dt << "  fa = " << std::setw(8)
                << std::fixed << pr << fa << "  fb = " << std::setw(8) << std::fixed << pr << fb
                << std::endl;
    }
  }

  return false;
}
