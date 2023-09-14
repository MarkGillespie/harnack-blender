namespace {
using f3 = std::array<float, 3>;
using f4 = std::array<float, 4>;
float dot(const f3 &a, const f3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
f3 cross(const f3 &a, const f3 &b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
float len_squared(const f3 &a)
{
  return dot(a, a);
}
float len(const f3 &a)
{
  return sqrt(len_squared(a));
}
f3 diff(const f3 &a, const f3 &b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
f3 diff_f(const float3 &a, const f3 &b)
{
  return {(float)a.x - b[0], (float)a.y - b[1], (float)a.z - b[2]};
}
// a + s * b
f3 fma(const f3 &a, float s, const f3 &b)
{
  return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
}

//== Quaternions
f3 orthogonal(const f3 &v)  // find a vector orthogonal to v
{
  if (std::abs(v[0]) <= std::abs(v[1]) && std::abs(v[0]) <= std::abs(v[2])) {
    return f3{0., -v[2], v[1]};
  }
  else if (std::abs(v[1]) <= std::abs(v[0]) && std::abs(v[1]) <= std::abs(v[2])) {
    return f3{v[2], 0., -v[0]};
  }
  else {
    return f3{-v[1], v[0], 0.};
  }
}
f3 mul_s(float s, const f3 &v)
{
  return {s * v[0], s * v[1], s * v[2]};
}
float q_re(const f4 &q)
{
  return q[0];
}
f3 q_im(const f4 &q)
{
  return {q[1], q[2], q[3]};
}
f4 build_T4(float x, const f3 &yzw)
{
  return {x, yzw[0], yzw[1], yzw[2]};
}
float q_dot(const f4 &a, const f4 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
f4 q_mul(const f4 &a, const f4 &b)
{
  f3 u = mul_s(q_re(a), q_im(b));
  f3 v = mul_s(q_re(b), q_im(a));
  f3 w = cross(q_im(a), q_im(b));
  return {q_re(a) * q_re(b) - dot(q_im(a), q_im(b)),
          u[0] + v[0] + w[0],
          u[1] + v[1] + w[1],
          u[2] + v[2] + w[2]};
}
f4 q_conj(const f4 &q)
{
  return {q[0], -q[1], -q[2], -q[3]};
}
f4 q_div_s(const f4 &q, float s)
{
  return {q[0] / s, q[1] / s, q[2] / s, q[3] / s};
}
f4 q_inv(const f4 &q)
{
  return q_div_s(q_conj(q), q_dot(q, q));
}
f4 q_div(const f4 &a, const f4 &b)
{
  return q_mul(a, q_inv(b));
}

// dihedral of two points on the unit sphere, as defined by Chern & Ishida
// https://arxiv.org/abs/2303.14555
// https://stackoverflow.com/a/11741520
f4 dihedral(const f3 &p1, const f3 &p2)
{
  float lengthProduct = len(p1) * len(p2);

  // antiparallel vectors
  if (std::abs(dot(p1, p2) / lengthProduct + (float)1.) < (float)0.0001)
    return build_T4(0., orthogonal(p1));

  // can skip normalization since we don't care about magnitude
  return build_T4(dot(p1, p2) + lengthProduct, cross(p1, p2));
}
// arg(\bar{q2} q1) as defined by Chern & Ishida https://arxiv.org/abs/2303.14555
float fiberArg(const f4 &q1, const f4 &q2)
{
  f4 s = q_mul(q_conj(q2), q1);
  return atan2(s[1], s[0]);
}

using d3 = std::array<double, 3>;
using d4 = std::array<double, 4>;
double dot(const d3 &a, const d3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
d3 cross(const d3 &a, const d3 &b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double len_squared(const d3 &a)
{
  return dot(a, a);
}
double len(const d3 &a)
{
  return sqrt(len_squared(a));
}
d3 diff(const d3 &a, const d3 &b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
d3 diff_f(const float3 &a, const d3 &b)
{
  return {(double)a.x - b[0], (double)a.y - b[1], (double)a.z - b[2]};
}
// a + s * b
d3 fma(const d3 &a, double s, const d3 &b)
{
  return {a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]};
}

//== Quaternions
d3 orthogonal(const d3 &v)  // dind a vector orthogonal to v
{
  if (std::abs(v[0]) <= std::abs(v[1]) && std::abs(v[0]) <= std::abs(v[2])) {
    return d3{0., -v[2], v[1]};
  }
  else if (std::abs(v[1]) <= std::abs(v[0]) && std::abs(v[1]) <= std::abs(v[2])) {
    return d3{v[2], 0., -v[0]};
  }
  else {
    return d3{-v[1], v[0], 0.};
  }
}
d3 mul_s(double s, const d3 &v)
{
  return {s * v[0], s * v[1], s * v[2]};
}
double q_re(const d4 &q)
{
  return q[0];
}
d3 q_im(const d4 &q)
{
  return {q[1], q[2], q[3]};
}
d4 build_T4(double x, const d3 &yzw)
{
  return {x, yzw[0], yzw[1], yzw[2]};
}
double q_dot(const d4 &a, const d4 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
d4 q_mul(const d4 &a, const d4 &b)
{
  d3 u = mul_s(q_re(a), q_im(b));
  d3 v = mul_s(q_re(b), q_im(a));
  d3 w = cross(q_im(a), q_im(b));
  return {q_re(a) * q_re(b) - dot(q_im(a), q_im(b)),
          u[0] + v[0] + w[0],
          u[1] + v[1] + w[1],
          u[2] + v[2] + w[2]};
}
d4 q_conj(const d4 &q)
{
  return {q[0], -q[1], -q[2], -q[3]};
}
d4 q_div_s(const d4 &q, double s)
{
  return {q[0] / s, q[1] / s, q[2] / s, q[3] / s};
}
d4 q_inv(const d4 &q)
{
  return q_div_s(q_conj(q), q_dot(q, q));
}
d4 q_div(const d4 &a, const d4 &b)
{
  return q_mul(a, q_inv(b));
}

// dihedral of two points on the unit sphere, as defined by Chern & Ishida
// https://arxiv.org/abs/2303.14555
// https://stackoverflow.com/a/11741520
d4 dihedral(const d3 &p1, const d3 &p2)
{
  double lengthProduct = len(p1) * len(p2);

  // antiparallel vectors
  if (std::abs(dot(p1, p2) / lengthProduct + 1.) < 0.0001)
    return build_T4(0., orthogonal(p1));

  // can skip normalization since we don't care about magnitude
  return build_T4(dot(p1, p2) + lengthProduct, cross(p1, p2));
}
// arg(\bar{q2} q1) as defined by Chern & Ishida https://arxiv.org/abs/2303.14555
double fiberArg(const d4 &q1, const d4 &q2)
{
  d4 s = q_mul(q_conj(q2), q1);
  return atan2(s[1], s[0]);
}
}  // namespace
