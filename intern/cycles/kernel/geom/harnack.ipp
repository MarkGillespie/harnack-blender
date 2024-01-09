// TODO: move back to math utils?
// int solid_angle_formula (from DNA_modifier_types.h):
//    MOD_HARNACK_TRIANGULATE = 0,
//    MOD_HARNACK_PREQUANTUM = 1,
//    MOD_HARNACK_GAUSS_BONNET = 2,

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
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };

  T epsilon = static_cast<T>(params.epsilon);
  T frequency = static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);

  T3 ray_P = from_float3(params.ray_P);
  T3 ray_D = from_float3(params.ray_D);

  T ray_tmin = static_cast<T>(params.ray_tmin);
  T ray_tmax = static_cast<T>(params.ray_tmax);

  T radius = static_cast<T>(params.R);
  T outerRadius = static_cast<T>(1.25) * radius;  // TODO: make configurable?

  T unitBound = sphericalHarmonicBound(params.l, params.m);
  // since the spherical harmonic is homogeneous of degree l, we can bound it
  // using a bound on the unit sphere and our sphere radius
  T shift = unitBound * static_cast<T>(std::pow(outerRadius, params.l));

  // find the two times at which a ray intersects a sphere
  auto intersectSphere = [](const T3 &ro, const T3 &rd, T radius, T *t0, T *t1) -> bool {
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
      if (t1 < t0)
        std::swap(*t0, *t1);
      return true;
    }
  };

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
  bool hitSphere = intersectSphere(ray_P, ray_D, radius, &t0, &t1);

  // if we miss the sphere, there cannot be an intersection with the levelset
  // within the sphere
  if (!hitSphere)
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

// compute solid angle of loop starting at pts[iStart] with length N,
// evaluated at point x, and adds the gradient to *grad, computed as follows:
// grad_mode: 0 - nicole formula
//            1 - none
//            2 - architecture formula
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

    //== compute gradient
    if (grad) {
      if (grad_mode == 0) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
      else {
        const T3 &v = xp[a];
        const T3 &w = xp[b];
        T lv = len(v);
        T lw = len(w);
        T scale = (lv + lw) / (lv * lw + dot(v, w));
        (*grad)[0] += n[0] * scale;
        (*grad)[1] += n[1] * scale;
        (*grad)[2] += n[2] * scale;
      }
    }
  }

  return 2 * std::arg(running_angle);
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
} solid_angle_intersection_params;

typedef struct acceleration_stats {
  // general
  int total_iterations = 0;
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

static bool printed_disk_normal = false;
template<typename T>
ccl_device bool ray_nonplanar_polygon_intersect_T(const solid_angle_intersection_params &params,
                                                  ccl_private float *isect_u,
                                                  ccl_private float *isect_v,
                                                  ccl_private float *isect_t,
                                                  acceleration_stats *stats = nullptr)
{
  using T3 = std::array<T, 3>;
  using T4 = std::array<T, 4>;
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };

  T epsilon = static_cast<T>(params.epsilon);
  T frequency = static_cast<T>(params.frequency);
  T levelset = static_cast<T>(params.levelset);
  T shift = 4. * M_PI;

  T3 ray_P = from_float3(params.ray_P);
  T3 ray_D = from_float3(params.ray_D);
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

  // classify each loop as a polygon or as an disk
  // HACK: count a polygon as a disk if it has five sides, and is
  // disk-inscribed
  std::vector<uint> polygonLoops;
  std::vector<T3> diskCenters, diskNormals;
  std::vector<T> diskRadii;
  for (uint iL = 0; iL < params.n_loops; iL++) {
    uint iStart = params.loops[iL].x - globalStart;
    uint N = params.loops[iL].y;

    //== Test if loop is "circular", i.e. 5-sided, equidistant from center
    // point, and planar
    //     if so, store the center, normal, and radius
    T3 center, normal;
    T radius;
    bool isCircular = true;
    if (N != 5)
      isCircular = false;

    if (isCircular) {
      center = from_float3(params.pts[iStart + N]);

      // Store displacements from center
      std::vector<T3> rx;
      rx.reserve(N);
      for (uint i = 0; i < N; i++)
        rx.push_back(diff_f(params.pts[iStart + i], center));

      // Test if points are equidistant from center, and store radius
      radius = len(rx[0]);
      for (uint i = 1; i < N; i++) {
        T radius_i = len(rx[i]);
        if (std::abs(radius_i - radius) >= 0.01) {
          isCircular = false;
          break;
        }
      }

      if (isCircular) {
        // Test if points are coplanar (and store normal)
        normal = cross(rx[0], rx[1]);
        normalize(normal);
        for (uint i = 1; i < N; i++) {
          int j = (i + 1) % N;
          T3 normal_i = cross(rx[i], rx[j]);

          // Check whether or not normal is parallel to normal_i
          if (std::abs(dot(normal, normal_i) - len(normal) * len(normal_i)) >= 0.01) {
            isCircular = false;
            break;
          }
        }
      }
    }

    if (isCircular) {
      // if we got here, the loop is circular. Store it as a disk
      diskCenters.push_back(center);
      diskNormals.push_back(normal);
      diskRadii.push_back(radius);

      if (!printed_disk_normal) {
        std::cout << ("normal: (" + std::to_string(normal[0]) + ", " + std::to_string(normal[1]) +
                      ", " + std::to_string(normal[2]) + ")")
                  << std::endl;
        printed_disk_normal = true;
      }
    }
    else {
      // if loop was not determined to be circular, record it as a polygon
      polygonLoops.push_back(iL);
    }
  }

  // Fractional mod function that behaves the same as in GLSL
  auto glsl_mod = [](T x, T y) -> T { return x - y * std::floor(x / y); };

  auto close_to_zero = [&](T ang, T lo_bound, T hi_bound, T tol, const T3 &grad) -> bool {
    // Check if an angle is within epsilon of 0 or 4π
    T dis = fmin(ang - lo_bound, hi_bound - ang);
    T tolScaling = params.use_grad_termination ? len(grad) : 1;
    return dis < tol * tolScaling;
  };

  // update min_d2, closest_point, and tangent if any closer point is present
  // on loop iL. closest_point and tangent may be null, but min_d2 must not be
  auto squared_distance_to_polygon_boundary =
      [&](uint iL, const T3 &x, T *min_d2, T3 *closest_point, T3 *tangent) {
        uint iStart = params.loops[iL].x - globalStart;
        uint N = params.loops[iL].y;

        // compute closest distance to each polygon line segment
        for (uint i = 0; i < N; i++) {
          T3 p1 = from_float3(params.pts[iStart + i]);
          T3 p2 = from_float3(params.pts[iStart + (i + 1) % N]);
          T3 m = diff(p2, p1);
          T3 v = diff(x, p1);
          // dot = |a|*|b|cos(theta) * n, isolating |a|sin(theta)
          T t = fmin(fmax(dot(m, v) / dot(m, m), 0.), 1.);
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
        T d2 = std::pow(rm - r0, 2) + std::pow(L, 2);
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

  // sums solid angle of polygon loops and adds gradient to grad
  auto triangulated_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    T omega = 0;
    for (uint iL : polygonLoops) {
      uint iStart = params.loops[iL].x - globalStart;
      uint N = params.loops[iL].y;
      omega += triangulated_loop_solid_angle(
          params.pts, iStart, N, x, &grad, 0, params.use_quick_triangulation);
    }
    return omega;
  };

  // returns solid angle of loop iL and adds gradient to grad
  auto prequantum_loop_solid_angle = [&](uint iL, const T3 &x, T3 &grad) -> T {
    uint iStart = params.loops[iL].x - globalStart;
    uint N = params.loops[iL].y;

    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N);
    std::vector<T> Lp;
    Lp.reserve(N);
    for (uint i = 0; i < N; i++) {
      xp.push_back(diff_f(params.pts[iStart + i], x));
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
      if (params.use_grad_termination) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T3 n = cross(g0, g1);
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        grad[0] -= n[0] / n2 * scale;
        grad[1] -= n[1] / n2 * scale;
        grad[2] -= n[2] / n2 * scale;
      }
    }

    return static_cast<T>(-2.) * fiberArg(q0, qi);
  };

  // sums solid angle of polygon loops and adds gradient to grad
  auto prequantum_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    T omega = 0;
    for (uint iL : polygonLoops)
      omega += prequantum_loop_solid_angle(iL, x, grad);
    return omega;
  };

  // returns solid angle of loop iL and adds gradient to grad
  auto gauss_bonnet_loop_solid_angle = [&](uint iL, const T3 &x, T3 &grad) -> T {
    uint iStart = params.loops[iL].x - globalStart;
    uint N = params.loops[iL].y;

    // compute the vectors xp from the evaluation point x
    // to all the polygon vertices, and their lengths Lp
    std::vector<T3> xp;
    xp.reserve(N);
    std::vector<T> Lp;
    Lp.reserve(N);
    for (uint i = 0; i < N; i++) {
      xp.push_back(diff_f(params.pts[iStart + i], x));
      Lp.push_back(len(xp[i]));
    }

    // Iterate over triangles used to triangulate the polygon
    T total_angle = 0.;
    for (uint i = 0; i < N; i++) {
      int a = (i + N - 1) % N;
      int b = i;
      int c = (i + 1) % N;
      T3 n_prev = cross(xp[a], xp[b]);
      T3 n_next = cross(xp[b], xp[c]);

      total_angle += atan2(dot(xp[b], cross(n_prev, n_next)) / Lp[b], dot(n_prev, n_next));

      //== compute gradient
      if (params.use_grad_termination) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T n2 = len_squared(n_prev);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        grad[0] -= n_prev[0] / n2 * scale;
        grad[1] -= n_prev[1] / n2 * scale;
        grad[2] -= n_prev[2] / n2 * scale;
      }
    }
    return static_cast<T>(2. * M_PI) - total_angle;
  };

  // sums solid angle of polygon loops and adds gradient to grad
  auto gauss_bonnet_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    T omega = 0;
    for (uint iL : polygonLoops)
      omega += gauss_bonnet_loop_solid_angle(iL, x, grad);
    return omega;
  };

  // returns solid angle of disk iD and adds gradient to grad
  auto disk_solid_angle = [&](uint iD, const T3 &x, T3 &grad) -> T {
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
    T3 displacement = diff(x, diskCenters[iD]);  // might need to switch to diff_f
    const T3 &zHat = diskNormals[iD];
    T L = dot(displacement, diskNormals[iD]);
    T3 inPlane = fma(displacement, -L, zHat);
    T r0 = len(inPlane);
    T sign = copysign(1, L);
    T aL = std::abs(L);  // use absolute value of length when evaluating solid
                         // angle to shift branch cut to the horizontal plane
    T rm = diskRadii[iD];
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
      omega = sign * (2 * M_PI - 2 * aL / Rmax * F +
                      2 * aL / Rmax * (r0 - rm) / (r0 + rm) *
                          elliptic_pim(alphaSq, kSq, elliptic_errtol));
    }
    else if (r0 > rm) {
      omega = sign * (-2 * aL / Rmax * F + 2 * aL / Rmax * (r0 - rm) / (r0 + rm) *
                                               elliptic_pim(alphaSq, kSq, elliptic_errtol));
    }

    if (true) {
      T E = elliptic_em(kSq, elliptic_errtol);
      T Br = -2 * aL / (r0 * Rmax) * (-F + (rm * rm + r0 * r0 + aL * aL) / R1Sq * E);
      T Bz = -2 / (Rmax) * (F + (rm * rm - r0 * r0 - aL * aL) / R1Sq * E);

      for (size_t i = 0; i < 3; i++)
        (grad)[i] += sign * Br * rHat[i] + Bz * zHat[i];
    }
    return omega;
  };

  // sums solid angle of polygon loops and disks and adds gradient to grad
  auto total_solid_angle = [&](const T3 &x, T3 &grad) -> T {
    T omega = params.solid_angle_formula == 0 ? triangulated_solid_angle(x, grad) :
              params.solid_angle_formula == 1 ? prequantum_solid_angle(x, grad) :
                                                gauss_bonnet_solid_angle(x, grad);

    for (uint iD = 0; iD < diskCenters.size(); iD++)
      omega += disk_solid_angle(iD, x, grad);

    return omega;
  };

  /* Find intersection with Harnack tracing */

  T t = ray_tmin;
  int iter = 0;
  T lo_bound = 0;
  T hi_bound = 4. * M_PI;

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

      *isect_t = t + t_overstep;
      *isect_v = ((T)iter) / ((T)params.max_iterations);
      report_stats();
      if (params.capture_misses) {
        T3 pos = fma(ray_P, t, ray_D);
        T3 grad{0, 0, 0};
        T omega = total_solid_angle(pos, grad);
        *isect_u = omega / static_cast<T>(4. * M_PI);
        return true;
      }
      else {
        return false;
      }
    }

    T3 grad{0, 0, 0};
    T omega = total_solid_angle(pos, grad);

    // To get the most aggressive Harnack bound, we first find a
    // representative of the solid angle, shifted by the target level set
    // value, within the range [0,4π).  Only then do we apply the shift.
    T val = glsl_mod(omega - levelset, static_cast<T>(4. * M_PI));

    if (frequency > 0) {
      lo_bound = 0;
      hi_bound = 4. * M_PI;
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

    // Calculate the radius of a ball around the current point over which
    // we know the function is harmonic.  An easy way to identify such a
    // ball is to restrict to the sphere touching the closest point on the
    // polygon boundary.
    T3 closestPoint;
    T R = distance_to_boundary(pos, &closestPoint, nullptr);

    // Compute a conservative step size based on the Harnack bound.
    T r = get_max_step(val, R, lo_bound, hi_bound, shift) / ld;

    if (stats) {
      stats->times.push_back(t + t_overstep);
      stats->omegas.push_back(val);
      stats->Rs.push_back(R);
      stats->rs.push_back(r);
    }

    if (r >= t_overstep) {  // commit to step
      // If we're close enough to the level set, or we've exceeded the
      // maximum number of iterations, assume there's a hit.
      if (close_to_zero(val, lo_bound, hi_bound, epsilon, grad) || R < epsilon ||
          iter > params.max_iterations)
      {
        // if (R < epsilon)   grad = pos - closestPoint; // TODO: this?
        *isect_t = t + t_overstep;
        *isect_u = omega / static_cast<T>(4. * M_PI);
        *isect_v = ((T)iter) / ((T)params.max_iterations);
        report_stats();
        return true;
      }

      if (close_to_zero(val, lo_bound, hi_bound, params.epsilon_loose, grad)) {
        // try newton's method when you first enter the epsilon_loose
        // shell
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
              T val_err = (val < 2. * M_PI ? -val : 4. * M_PI - val);
              T alpha = -atan2(dot(tangent, cross(rd_planar, grad)), dot(rd_planar, grad));
              T dt = R * sin(val_err) / cos(val_err - alpha);
              // todo: clever trig to use tan(val) somehow?
              T old_val = val;
              val = f(t_radial + dt);
              T new_val_err = (val < 2. * M_PI ? -val : 4. * M_PI - val);
              while (val_err * new_val_err < 0 && dt > 1e-8) {
                dt /= 2;
                val = f(t_radial + dt);
                new_val_err = (val < 2. * M_PI ? -val : 4. * M_PI - val);
                if (stats) {
                  stats->n_newton_steps++;
                }
              }
              t_radial += dt;
              bool close = close_to_zero(val, lo_bound, hi_bound, epsilon, grad);

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
              //     << " close: " << (close ? "true" : "false");
              // std::cout << std::endl;
              if (close) {
                *isect_t = t_radial;
                *isect_u = omega / static_cast<T>(4. * M_PI);
                *isect_v = ((T)iter) / ((T)params.max_iterations);
                report_stats();
                return true;
              }
            }
            // t = t_radial;
          }
          else {  // otherwise, run a few rounds of Newton's method
            T t_newton = t + t_overstep;
            for (int i = 0; i < 8; i++) {
              T df = dot(ray_D, grad);
              T dt = -(val < 2. * M_PI ? val : val - 4. * M_PI) / df;

              t_newton += dt;
              val = f(t_newton);

              if (stats) {
                stats->newton_dfs.push_back(df);
                stats->newton_dts.push_back(dt);
                stats->newton_ts.push_back(t_newton);
                stats->newton_vals.push_back(val);
                stats->n_newton_steps++;
              }
              if (close_to_zero(val, lo_bound, hi_bound, epsilon, grad)) {
                *isect_t = t_newton;
                *isect_u = omega / static_cast<T>(4. * M_PI);
                *isect_v = ((T)iter) / ((T)params.max_iterations);
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

      if (params.use_extrapolation) {
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
          if (close_to_zero(e_val, lo_bound, hi_bound, epsilon, e_grad)) {
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
        t_overstep = r * .75;
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

  *isect_t = t;
  *isect_v = ((T)iter) / ((T)params.max_iterations);
  report_stats();
  if (params.capture_misses) {
    T3 pos = fma(ray_P, t, ray_D);
    T3 grad{0, 0, 0};
    T omega = total_solid_angle(pos, grad);
    *isect_u = omega / static_cast<T>(4. * M_PI);
    return true;
  }
  else {
    return false;
  }
}
/* Normal on nonplanar polygon. */
// TODO: move to better-named file
// TODO: deduplicate with version in harnack tracing code?

template<typename T>
ccl_device float3 ray_nonplanar_polygon_normal_T(const float3 pf,
                                                 const packed_uint3 *loops,
                                                 const packed_float3 *pts,
                                                 const uint n_loops)
{
  using T3 = std::array<T, 3>;
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };
  /* auto smoothstep = [](T x, T start, T end) -> T {  //
   * https://docs.gl/sl4/smoothstep */
  /*   // normalize and clamp to [0, 1] */
  /*   x = fmin(fmax(((x - start) / (end - start)), 0), 1); */

  /*   return x * x * (3. - 2. * x); */
  /* }; */

  // find solid angle gradient and closest point on boundary curve
  uint globalStart = loops[0].x;
  T3 p = from_float3(pf);

  // classify each loop as a polygon or as an disk
  // HACK: count a polygon as a disk if it has five sides, and is
  // disk-inscribed
  std::vector<uint> polygonLoops;
  std::vector<T3> diskCenters, diskNormals;
  std::vector<T> diskRadii;
  for (uint iL = 0; iL < n_loops; iL++) {
    uint iStart = loops[iL].x - globalStart;
    uint N = loops[iL].y;

    //== Test if loop is "circular", i.e. 5-sided, equidistant from
    // center
    // point, and planar
    //     if so, store the center, normal, and radius
    T3 center, normal;
    T radius;
    bool isCircular = true;
    if (N != 5)
      isCircular = false;

    if (isCircular) {
      center = from_float3(pts[iStart + N]);

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
          isCircular = false;
          break;
        }
      }

      if (isCircular) {
        // Test if points are coplanar (and store normal)
        normal = cross(rx[0], rx[1]);
        normalize(normal);
        for (uint i = 1; i < N; i++) {
          int j = (i + 1) % N;
          T3 normal_i = cross(rx[i], rx[j]);

          // Check whether or not normal is parallel to normal_i
          if (std::abs(dot(normal, normal_i) - len(normal) * len(normal_i)) >= 0.01) {
            isCircular = false;
            break;
          }
        }
      }
    }

    if (isCircular) {
      // if we got here, the loop is circular. Store it as a disk
      diskCenters.push_back(center);
      diskNormals.push_back(normal);
      diskRadii.push_back(radius);
    }
    else {
      // if loop was not determined to be circular, record it as a
      // polygon
      polygonLoops.push_back(iL);
    }
  }

  // returns solid angle of loop iL and adds gradient to grad
  auto triangulated_loop_solid_angle = [&](uint iL, const T3 &x, T3 *grad) -> T {
    uint iStart = loops[iL].x - globalStart;
    uint N = loops[iL].y;

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
    for (uint i = 0; i < N; i++) {
      int a = i;
      int b = (i + 1) % N;
      int c = N;

      T3 n = cross(xp[a], xp[b]);

      // Add the solid angle of this triangle to the total
      std::complex<T> tri_angle{Lp[a] * Lp[b] * Lp[c] + dot(xp[a], xp[b]) * Lp[c] +
                                    dot(xp[b], xp[c]) * Lp[a] + dot(xp[a], xp[c]) * Lp[b],
                                dot(xp[c], n)};
      running_angle *= tri_angle;

      //== compute gradient
      if (grad) {
        const T3 &g0 = xp[a];
        const T3 &g1 = xp[b];
        T n2 = len_squared(n);
        T scale = ((-dot(g0, g1) + dot(g0, g0)) / len(g0) +
                   (-dot(g0, g1) + dot(g1, g1)) / len(g1));
        (*grad)[0] += n[0] / n2 * scale;
        (*grad)[1] += n[1] / n2 * scale;
        (*grad)[2] += n[2] / n2 * scale;
      }
    }

    return 2 * std::arg(running_angle);
  };

  // returns solid angle of disk iD and adds gradient to grad
  auto disk_solid_angle = [&](uint iD, const T3 &x, T3 *grad) -> T {
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

    // Formulas from Solid Angle Calculation for a Circular Disk by
    // Paxton 1959
    T3 displacement = diff(x, diskCenters[iD]);  // might need to switch to diff_f
    const T3 &zHat = diskNormals[iD];
    T L = dot(displacement, diskNormals[iD]);
    T3 inPlane = fma(displacement, -L, zHat);
    T r0 = len(inPlane);
    T sign = copysign(1, L);
    T aL = std::abs(L);  // use absolute value of length when evaluating solid
                         // angle to shift branch cut to the horizontal plane
    T rm = diskRadii[iD];
    T alphaSq = 4. * r0 * rm / std::pow(r0 + rm, 2);
    T R1Sq = std::pow(L, 2) + std::pow(r0 - rm, 2);
    T Rmax = std::sqrt(std::pow(L, 2) + std::pow(r0 + rm, 2));
    T kSq = 1 - R1Sq / (Rmax * Rmax);
    T3 rHat = over(inPlane, r0);

    T F = elliptic_fm(kSq, elliptic_errtol);

    // TODO: adjust tolerance
    // ( When r0 and rm are too close, alphaSq goes to 1, which leads to
    // a singularity in elliptic_pim )
    T omega;
    if (std::abs(r0 - rm) < 1e-3) {
      omega = sign * (M_PI - 2 * aL / Rmax * F);
    }
    else if (r0 < rm) {
      omega = sign * (2 * M_PI - 2 * aL / Rmax * F +
                      2 * aL / Rmax * (r0 - rm) / (r0 + rm) *
                          elliptic_pim(alphaSq, kSq, elliptic_errtol));
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
  };

  auto triangulated_solid_angle = [&](const T3 &x, T3 *grad) -> T {
    T omega = 0;
    for (uint iL : polygonLoops)
      omega += triangulated_loop_solid_angle(iL, x, grad);
    for (uint iD = 0; iD < diskCenters.size(); iD++)
      omega += disk_solid_angle(iD, x, grad);
    return omega;
  };

  T3 grad{0, 0, 0};
  if (false) {
    T h = 0.000001;
    T omega = triangulated_solid_angle(p, nullptr);
    T omega_x = triangulated_solid_angle(T3{p[0] + h, p[1], p[2]}, nullptr);
    T omega_y = triangulated_solid_angle(T3{p[0], p[1] + h, p[2]}, nullptr);
    T omega_z = triangulated_solid_angle(T3{p[0], p[1], p[2] + h}, nullptr);

    T fd_df_dx = (omega_x - omega) / h;
    T fd_df_dy = (omega_y - omega) / h;
    T fd_df_dz = (omega_z - omega) / h;

    grad = T3{fd_df_dx, fd_df_dy, fd_df_dz};
  }

  triangulated_solid_angle(p, &grad);

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
  auto from_float3 = [](const float3 &p) -> T3 { return {(T)p.x, (T)p.y, (T)p.z}; };

  T3 p = from_float3(pf);

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
