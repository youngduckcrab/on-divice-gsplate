// Face covering on a synthetic standing person whose head has dark eyes and a
// mouth: finding the head, placing the face from landmarks a detector would
// report, and what each cover style leaves visible. CPU only.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gs/face_cover.h"
#include "test_util.h"

namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr float kPi = 3.14159265358979f;

struct V {
  float x, y, z;
};
V add(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V mul(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float len(V a) { return std::sqrt(dot(a, a)); }

// A person facing `facing` (horizontal, unit) with the head centred at (0, 1.6, 0), y up.
struct Person {
  gs::SplatScene scene;
  V eyes[2];  // viewer's left, then right
  V mouth;
  V facing;
  std::vector<uint32_t> eye_splats;
};

Person make_person(float yaw) {
  Person p;
  p.facing = {std::sin(yaw), 0.0f, std::cos(yaw)};
  const V side = {std::cos(yaw), 0.0f, -std::sin(yaw)};  // facing x up: the person's left, the viewer's right
  const V centre = {0.0f, 1.6f, 0.0f};
  const float rx = 0.075f, ry = 0.1f, rz = 0.09f;
  auto local = [&](float x, float y, float z) { return add(centre, add(add(mul(side, x), V{0, y, 0}), mul(p.facing, z))); };
  // Where the front of the head is at (x, y).
  auto front = [&](float x, float y) {
    const float t = 1.0f - (x / rx) * (x / rx) - (y / ry) * (y / ry);
    return local(x, y, rz * std::sqrt(std::max(t, 0.0f)));
  };
  p.eyes[0] = front(-0.0315f, 0.0f);
  p.eyes[1] = front(0.0315f, 0.0f);
  p.mouth = front(0.0f, -0.05f);

  std::vector<V> pos, rgb;
  std::vector<float> sigma;
  // Head surface: a Fibonacci lattice on the ellipsoid.
  const int head = 7000;
  for (int i = 0; i < head; ++i) {
    const float t = (float(i) + 0.5f) / float(head);
    const float polar = std::acos(1.0f - 2.0f * t);
    const float az = kPi * (3.0f - std::sqrt(5.0f)) * float(i);
    const float sx = std::sin(polar) * std::cos(az), sy = std::cos(polar), sz = std::sin(polar) * std::sin(az);
    const V q = local(rx * sx, ry * sy, rz * sz);
    V c = {0.82f, 0.62f, 0.52f};
    bool eye = false;
    for (const V& e : p.eyes) eye = eye || len(sub(q, e)) < 0.011f;
    const V m = sub(q, p.mouth);
    if (eye) {
      c = {0.05f, 0.04f, 0.04f};
    } else if (std::abs(dot(m, side)) < 0.022f && std::abs(m.y) < 0.006f && dot(m, p.facing) > -0.02f) {
      c = {0.45f, 0.08f, 0.1f};
    }
    if (eye) p.eye_splats.push_back(uint32_t(pos.size()));
    pos.push_back(q);
    rgb.push_back(c);
    sigma.push_back(0.004f);
  }
  // Body: a column below the head.
  uint32_t seed = 7;
  auto rnd = [&]() {
    seed = seed * 1664525u + 1013904223u;
    return float(seed >> 8) / float(1u << 24);
  };
  for (int i = 0; i < 4000; ++i) {
    const float a = 2.0f * kPi * rnd(), r = 0.15f * std::sqrt(rnd());
    pos.push_back({r * std::cos(a), 1.48f * rnd(), r * std::sin(a)});
    rgb.push_back({0.3f, 0.35f, 0.5f});
    sigma.push_back(0.02f);
  }
  gs::SplatScene& s = p.scene;
  s.convention = gs::AxisConvention::RUB;
  s.resize(uint32_t(pos.size()), 1);
  for (size_t i = 0; i < pos.size(); ++i) {
    s.positions[i * 3 + 0] = pos[i].x;
    s.positions[i * 3 + 1] = pos[i].y;
    s.positions[i * 3 + 2] = pos[i].z;
    for (int k = 0; k < 3; ++k) s.log_scales[i * 3 + k] = std::log(sigma[i]);
    s.rotations[i * 4] = 1.0f;
    s.opacity_logits[i] = 5.0f;  // an opaque head: its back hides the inside of its face
    s.sh_dc[i * 3 + 0] = (rgb[i].x - 0.5f) / kShC0;
    s.sh_dc[i * 3 + 1] = (rgb[i].y - 0.5f) / kShC0;
    s.sh_dc[i * 3 + 2] = (rgb[i].z - 0.5f) / kShC0;
    // A little view-dependent colour, which covering must remove.
    for (int k = 0; k < 9; ++k) s.sh_rest[i * 9 + k] = 0.02f * float((i + k) % 5) - 0.04f;
  }
  return p;
}

// Where `point` lands in view v (pixel coordinates, centres at +0.5).
void project(const gs::OrthoView& v, V point, float xy[2]) {
  const V rel = sub(point, V{v.origin[0], v.origin[1], v.origin[2]});
  xy[0] = dot(rel, V{v.right[0], v.right[1], v.right[2]}) / v.pixel + 0.5f * float(v.width);
  xy[1] = dot(rel, V{v.down[0], v.down[1], v.down[2]}) / v.pixel + 0.5f * float(v.height);
}

// A detector's report: the eyes and mouth in every view that sees the face,
// and with `fakes`, wider "faces" in the back of the head as detectors find
// in hair.
std::string landmarks_json(const Person& p, const gs::FaceViews& views, bool fakes = false) {
  std::string json = "[";
  for (size_t i = 0; i < views.views.size(); ++i) {
    const gs::OrthoView& v = views.views[i];
    const V towards = {-v.forward[0], -v.forward[1], -v.forward[2]};
    if (fakes && dot(towards, p.facing) < -0.5f) {
      const float c = 0.5f * float(v.width);
      char buf[160];
      std::snprintf(buf, sizeof(buf), "%s{\"view\":%zu,\"eyes\":[[%.1f,%.1f],[%.1f,%.1f]],\"mouth\":[%.1f,%.1f]}",
                    json.size() > 1 ? "," : "", i, c - 26, c - 10, c + 26, c - 10, c, c + 30);
      json += buf;
      continue;
    }
    if (dot(towards, p.facing) < std::cos(50.0f * kPi / 180.0f)) continue;
    float e0[2], e1[2], m[2];
    project(v, p.eyes[1], e0);  // detectors may report either eye first
    project(v, p.eyes[0], e1);
    project(v, p.mouth, m);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s{\"view\":%zu,\"eyes\":[[%.2f,%.2f],[%.2f,%.2f]],\"mouth\":[%.2f,%.2f]}",
                  json.size() > 1 ? "," : "", i, e0[0], e0[1], e1[0], e1[1], m[0], m[1]);
    json += buf;
  }
  return json + "]";
}

// The scene seen from `facing` (or from behind it), looking at the eyes.
gs::OrthoView look(const gs::SplatScene& scene, const Person& p, bool behind) {
  gs::OrthoView v;
  const V mid = mul(add(p.eyes[0], p.eyes[1]), 0.5f);
  const V dir = behind ? p.facing : mul(p.facing, -1.0f);  // viewing direction
  const V origin = sub(mid, mul(dir, 0.4f));
  const V down = {0, -1, 0};
  const V right = {down.y * dir.z - down.z * dir.y, down.z * dir.x - down.x * dir.z, down.x * dir.y - down.y * dir.x};
  v.origin[0] = origin.x, v.origin[1] = origin.y, v.origin[2] = origin.z;
  v.forward[0] = dir.x, v.forward[1] = dir.y, v.forward[2] = dir.z;
  v.down[0] = down.x, v.down[1] = down.y, v.down[2] = down.z;
  v.right[0] = right.x, v.right[1] = right.y, v.right[2] = right.z;
  v.width = v.height = 128;
  v.pixel = 0.3f / 128.0f;
  const float bg[3] = {0, 0, 0};
  gs::render_ortho(scene, {}, v, bg);
  return v;
}

const float* pixel(const gs::OrthoView& v, V point) {
  float xy[2];
  project(v, point, xy);
  const int x = int(xy[0]), y = int(xy[1]);
  return &v.rgb[(size_t(y) * v.width + x) * 3];
}

void test_render_ortho() {
  gs::SplatScene s;
  s.resize(1, 0);
  s.positions = {0.1f, 0.0f, 0.5f};
  s.log_scales = {std::log(0.02f), std::log(0.02f), std::log(0.02f)};
  s.rotations = {1, 0, 0, 0};
  s.opacity_logits = {6.0f};
  s.sh_dc = {(0.9f - 0.5f) / kShC0, (0.2f - 0.5f) / kShC0, (0.1f - 0.5f) / kShC0};
  gs::OrthoView v;
  v.width = v.height = 64;
  v.pixel = 0.01f;
  const float bg[3] = {0, 0, 1};
  gs::render_ortho(s, {}, v, bg);
  // The splat sits 10 px right of the centre (on the corner of pixels 41 and 42), half a metre in.
  const size_t p = size_t(32) * 64 + 42;
  EXPECT(v.rgb[p * 3] > 0.8f && v.rgb[p * 3 + 2] < 0.2f, "splat colour %.3f %.3f %.3f", v.rgb[p * 3],
         v.rgb[p * 3 + 1], v.rgb[p * 3 + 2]);
  EXPECT(std::abs(v.depth[p] - 0.5f) < 1e-4f, "depth %.4f", v.depth[p]);
  EXPECT(v.rgb[3 * 2 + 2] > 0.99f && std::isnan(v.depth[2]), "background corner");
  float q[3];
  v.point(42.0f, 32.0f, 0.5f, q);
  EXPECT(std::abs(q[0] - 0.1f) < 1e-4f && std::abs(q[1]) < 1e-4f && std::abs(q[2] - 0.5f) < 1e-4f,
         "point (%.4f %.4f %.4f)", q[0], q[1], q[2]);
}

void test_place(const Person& p, const gs::FaceViews& views, gs::FacePlacement* placed) {
  const V mid = mul(add(p.eyes[0], p.eyes[1]), 0.5f);
  EXPECT(views.head.found, "head not found");
  EXPECT(len(sub(V{views.head.center[0], views.head.center[1], views.head.center[2]}, V{0, 1.6f, 0})) < 0.06f,
         "head centre (%.3f %.3f %.3f)", views.head.center[0], views.head.center[1], views.head.center[2]);
  const auto faces = gs::parse_face_landmarks(landmarks_json(p, views));
  EXPECT(faces.size() >= 2, "%zu views see the face", faces.size());
  const gs::FacePlacement face = gs::place_face(p.scene, views, faces);
  EXPECT(face.found && face.detected, "face not placed");
  const V c = {face.center[0], face.center[1], face.center[2]};
  const V f = {face.forward[0], face.forward[1], face.forward[2]};
  const V r = {face.right[0], face.right[1], face.right[2]};
  EXPECT(len(sub(c, mid)) < 0.006f, "centre off by %.1f mm", 1000 * len(sub(c, mid)));
  EXPECT(dot(f, p.facing) > 0.97f, "forward . facing = %.3f", dot(f, p.facing));
  EXPECT(dot(r, sub(p.eyes[1], p.eyes[0])) > 0.0f, "right points to the viewer's left");
  EXPECT(std::abs(face.eye_distance - 0.063f) < 0.006f, "eye distance %.1f mm", 1000 * face.eye_distance);
  *placed = face;

  // Wider false detections in the back of the head do not win over the face several views agree on.
  const gs::FacePlacement despite = gs::place_face(p.scene, views, gs::parse_face_landmarks(landmarks_json(p, views, true)));
  const V dc = {despite.center[0], despite.center[1], despite.center[2]};
  EXPECT(despite.found && len(sub(dc, mid)) < 0.006f, "with false detections the centre is off by %.1f mm",
         1000 * len(sub(dc, mid)));

  const float facing[3] = {p.facing.x, p.facing.y, p.facing.z};
  const gs::FacePlacement guess = gs::guess_face(p.scene, views, facing);
  const V gc = {guess.center[0], guess.center[1], guess.center[2]};
  EXPECT(guess.found && !guess.detected, "guess not made");
  // A guess from the head's size alone: this bald head's eyes sit higher than usual.
  EXPECT(len(sub(gc, mid)) < 0.05f, "guessed centre off by %.1f mm", 1000 * len(sub(gc, mid)));
  const V gf = {guess.forward[0], guess.forward[1], guess.forward[2]};
  EXPECT(dot(gf, p.facing) > 0.99f, "guessed forward . facing = %.3f", dot(gf, p.facing));
}

void test_styles(const Person& p, const gs::FacePlacement& face) {
  const gs::OrthoView before = look(p.scene, p, false);
  const gs::OrthoView back_before = look(p.scene, p, true);
  const V cheek = add(p.eyes[0], V{0, -0.03f, 0});
  const V bridge = mul(add(p.eyes[0], p.eyes[1]), 0.5f);
  const float* eye_before = pixel(before, p.eyes[0]);
  EXPECT(eye_before[0] < 0.2f, "the eye should be dark before covering (%.2f)", eye_before[0]);
  const char* names[] = {"mosaic", "blur", "sunglasses", "mask", "smile", "eyes"};
  for (const char* name : names) {
    gs::FaceCoverOptions options;
    EXPECT(gs::parse_face_style(name, &options.style), "style %s", name);
    gs::FaceCoverStats stats;
    const gs::SplatScene covered = gs::cover_face(p.scene, face, options, &stats);
    const bool adds = std::string(name) != "eyes";  // only the eyes: the face's own splats, blurred
    EXPECT((adds ? stats.added > 300 : stats.added == 0) && covered.count == p.scene.count + stats.added,
           "%s added %u", name, stats.added);
    const gs::OrthoView after = look(covered, p, false);
    const float* eye = pixel(after, p.eyes[0]);
    const std::string style = name;
    if (style == "sunglasses") {
      EXPECT(stats.recoloured == 0, "sunglasses recoloured %u", stats.recoloured);
      EXPECT(eye[0] + eye[1] + eye[2] < 0.9f, "lens over the eye: %.2f %.2f %.2f", eye[0], eye[1], eye[2]);
      const float* m0 = pixel(before, p.mouth);
      const float* m1 = pixel(after, p.mouth);
      EXPECT(std::abs(m0[0] - m1[0]) < 0.03f, "mouth changed under sunglasses: %.2f -> %.2f", m0[0], m1[0]);
      continue;
    }
    EXPECT(stats.recoloured > 100, "%s recoloured %u", name, stats.recoloured);
    // The eyes' own splats lose their colour, view-dependent part included.
    for (uint32_t i : p.eye_splats) {
      const float red = covered.sh_dc[size_t(i) * 3] * kShC0 + 0.5f;
      float rest = 0;
      for (int k = 0; k < 9; ++k) rest += std::abs(covered.sh_rest[size_t(i) * 9 + k]);
      if (red < 0.25f || rest > 0.0f) {
        EXPECT(false, "%s left eye splat %u with red %.2f, SH rest %.3f", name, i, red, rest);
        break;
      }
    }
    // The back of the head is untouched (its middle: covers reach round the sides).
    const gs::OrthoView back = look(covered, p, true);
    double diff = 0;
    int counted = 0;
    for (uint32_t y = 0; y < back.height; ++y) {
      for (uint32_t x = 44; x < 84; ++x) {
        for (int c = 0; c < 3; ++c) {
          const size_t k = (size_t(y) * back.width + x) * 3 + c;
          diff += std::abs(back.rgb[k] - back_before.rgb[k]);
          ++counted;
        }
      }
    }
    diff /= double(counted);
    EXPECT(diff < 0.002, "%s changed the back of the head by %.4f", name, diff);
    if (style == "mosaic" || style == "blur") {
      EXPECT(eye[0] > 0.3f, "%s still shows a dark eye (%.2f)", name, eye[0]);
    } else if (style == "eyes") {
      EXPECT(eye[0] > 0.3f, "eyes still show a dark eye (%.2f)", eye[0]);
      // The rest of the face stays as it was.
      const float* m0 = pixel(before, p.mouth);
      const float* m1 = pixel(after, p.mouth);
      EXPECT(std::abs(m0[0] - m1[0]) + std::abs(m0[1] - m1[1]) < 0.02f, "eyes changed the mouth: %.2f -> %.2f", m0[0],
             m1[0]);
    } else if (style == "mask") {
      const float* c = pixel(after, cheek);
      EXPECT(c[0] > 0.55f && c[1] > 0.55f && c[2] > 0.5f, "mask cheek %.2f %.2f %.2f", c[0], c[1], c[2]);
    } else if (style == "smile") {
      const float* c = pixel(after, bridge);
      EXPECT(c[0] > 0.7f && c[1] > 0.5f && c[2] < 0.45f, "smile between the eyes %.2f %.2f %.2f", c[0], c[1], c[2]);
    }
  }
}

}  // namespace

int main() {
  test_render_ortho();
  for (float yaw_deg : {0.0f, 117.0f}) {
    const Person p = make_person(yaw_deg * kPi / 180.0f);
    const float up[3] = {0, 1, 0};
    const gs::FaceViews views = gs::render_face_views(p.scene, up, 12, 256);
    gs::FacePlacement face;
    test_place(p, views, &face);
    if (face.found) test_styles(p, face);
    // Views around a camera that looks at the face from the front.
    const float toward[3] = {-p.facing.x, -p.facing.y, -p.facing.z};
    const gs::FaceViews near = gs::render_face_views(p.scene, up, 12, 256, toward);
    EXPECT(near.views.size() == 3, "views around the camera: %zu", near.views.size());
    const gs::FacePlacement again = gs::place_face(p.scene, near, gs::parse_face_landmarks(landmarks_json(p, near)));
    EXPECT(again.found && again.views_used == 3, "placed from the camera's views: %d", again.views_used);
  }
  if (test::failures == 0) std::printf("face tests passed\n");
  return test::failures == 0 ? 0 : 1;
}
