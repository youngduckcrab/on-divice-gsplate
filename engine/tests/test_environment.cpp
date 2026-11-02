// 3D places: the .gsenv encoding, and putting a model into a place (floor
// found and met, the shadow under the feet, the light measured there, the
// place turned around the model). CPU only.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gs/environment.h"
#include "test_util.h"

namespace {

constexpr float kShC0 = 0.28209479177387814f;

void add(gs::SplatScene& s, float x, float y, float z, float size, const float rgb[3], float opacity = 0.95f) {
  const uint32_t i = s.count++;
  s.positions.insert(s.positions.end(), {x, y, z});
  s.log_scales.insert(s.log_scales.end(), {std::log(size), std::log(size), std::log(size * 0.1f)});
  s.rotations.insert(s.rotations.end(), {1, 0, 0, 0});
  s.opacity_logits.push_back(std::log(opacity / (1 - opacity)));
  for (int c = 0; c < 3; ++c) s.sh_dc.push_back((rgb[c] - 0.5f) / kShC0);
  (void)i;
}

void test_roundtrip() {
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  gs::SplatScene s;
  s.convention = gs::AxisConvention::RDF;
  for (int i = 0; i < 2000; ++i) {
    const float rgb[3] = {0.5f + 0.4f * u(rng), 0.5f + 0.4f * u(rng), 0.5f + 0.4f * u(rng)};
    add(s, 6 * u(rng), 2 * u(rng), 6 * u(rng), 0.02f + 0.01f * u(rng), rgb, 0.5f + 0.45f * u(rng));
    float* q = &s.rotations[size_t(i) * 4];
    float len = 0;
    for (int k = 0; k < 4; ++k) len += (q[k] = u(rng)) * q[k];
    for (int k = 0; k < 4; ++k) q[k] /= std::sqrt(len);
  }
  const std::vector<uint8_t> bytes = gs::write_gsenv(s);
  const gs::SplatScene r = gs::read_gsenv(bytes.data(), bytes.size());
  EXPECT(r.count == s.count && r.sh_degree == 0 && r.convention == gs::AxisConvention::RDF, "count %u", r.count);
  float worst_pos = 0, worst_rot = 1, worst_alpha = 0;
  for (uint32_t i = 0; i < r.count; ++i) {
    for (int k = 0; k < 3; ++k) worst_pos = std::max(worst_pos, std::abs(r.positions[i * 3 + k] - s.positions[i * 3 + k]));
    float d = 0;
    for (int k = 0; k < 4; ++k) d += r.rotations[i * 4 + k] * s.rotations[i * 4 + k];
    worst_rot = std::min(worst_rot, std::abs(d));
    auto sig = [](float x) { return 1 / (1 + std::exp(-x)); };
    worst_alpha = std::max(worst_alpha, std::abs(sig(r.opacity_logits[i]) - sig(s.opacity_logits[i])));
  }
  EXPECT(worst_pos < 0.002f, "position error %.4f", worst_pos);
  EXPECT(worst_rot > 0.995f, "rotation agreement %.4f", worst_rot);
  EXPECT(worst_alpha < 0.003f, "opacity error %.4f", worst_alpha);
}

// A standing person (RDF: y down, feet at y = 0) and a reddish room floor at
// y = 0.13 around the spot, with a marker splat 2 m along +x from the spot.
void test_stage() {
  gs::SplatScene person;
  person.convention = gs::AxisConvention::RDF;
  const float skin[3] = {0.7f, 0.6f, 0.5f};
  for (int i = 0; i <= 170; ++i) add(person, 0.3f, -0.01f * float(i), -0.2f, 0.05f, skin);
  gs::SplatScene place;
  place.convention = gs::AxisConvention::RDF;
  const float floor_rgb[3] = {0.7f, 0.4f, 0.35f};
  const float spot[2] = {-4.4f, -4.4f};
  for (int i = -40; i <= 40; ++i) {
    for (int k = -40; k <= 40; ++k) add(place, spot[0] + 0.05f * i, 0.13f, spot[1] + 0.05f * k, 0.04f, floor_rgb);
  }
  const float marker_rgb[3] = {0.1f, 0.9f, 0.1f};
  add(place, spot[0] + 2.0f, 0.13f - 1.0f, spot[1], 0.05f, marker_rgb);
  const uint32_t marker = place.count - 1;

  gs::Spot s;
  s.x = spot[0];
  s.z = spot[1];
  for (auto& row : s.sky) row[0] = row[1] = row[2] = 0.5f;  // grey where the place is not
  const gs::Staging staged = gs::stage(person, place, s);
  EXPECT(staged.model_count == person.count && staged.scene.count == person.count + place.count, "counts %u %u",
         staged.model_count, staged.scene.count);
  EXPECT(std::abs(staged.floor - 0.13f) < 0.02f, "floor found at %.3f", staged.floor);
  // The person is unchanged, and the place's floor meets their feet (y 0).
  EXPECT(staged.scene.positions[0] == person.positions[0] && staged.scene.positions[1] == person.positions[1],
         "the person moved");
  float darkest = 1, far = 0;
  for (uint32_t i = person.count; i < staged.scene.count - 1; ++i) {
    const float* p = &staged.scene.positions[size_t(i) * 3];
    EXPECT(std::abs(p[1]) < 0.005f, "floor splat at y %.3f", p[1]);
    if (std::abs(p[1]) >= 0.005f) break;
    const float red = staged.scene.sh_dc[size_t(i) * 3] * kShC0 + 0.5f;
    const float r = std::hypot(p[0] - 0.3f, p[2] + 0.2f);
    if (r < 0.05f) darkest = std::min(darkest, red);
    if (r > 1.8f) far = std::max(far, red);
  }
  EXPECT(darkest < 0.5f * floor_rgb[0] + 0.05f && far > floor_rgb[0] - 0.01f, "shadow: %.2f under the feet, %.2f away",
         darkest, far);
  // The marker sits 2 m along +x from the feet, 1 m up.
  const float* m = &staged.scene.positions[size_t(person.count + marker) * 3];
  EXPECT(std::abs(m[0] - 2.3f) < 0.01f && std::abs(m[1] + 1.0f) < 0.01f && std::abs(m[2] + 0.2f) < 0.01f,
         "marker at (%.2f %.2f %.2f)", m[0], m[1], m[2]);
  // A reddish place lights the person reddish.
  EXPECT(staged.light.on && staged.light.sh[0] > staged.light.sh[2], "light %.2f %.2f %.2f", staged.light.sh[0],
         staged.light.sh[1], staged.light.sh[2]);

  // Turned a quarter: the marker swings round the person's middle.
  s.yaw_deg = 90.0f;
  const gs::Staging turned = gs::stage(person, place, s);
  const float* t = &turned.scene.positions[size_t(person.count + marker) * 3];
  EXPECT(std::abs(t[0] - 0.3f) < 0.01f && std::abs(std::abs(t[2] + 0.2f) - 2.0f) < 0.01f, "turned marker at (%.2f %.2f %.2f)",
         t[0], t[1], t[2]);

  // A person in y-up axes stands on the same floor, flipped.
  gs::SplatScene up = person;
  gs::convert_convention(up, gs::AxisConvention::RUB);
  s.yaw_deg = 0.0f;
  const gs::Staging upright = gs::stage(up, place, s);
  const float* um = &upright.scene.positions[size_t(person.count + marker) * 3];
  EXPECT(std::abs(um[1] - 1.0f) < 0.01f, "marker in y-up axes at height %.2f", um[1]);
}

}  // namespace

int main() {
  test_roundtrip();
  test_stage();
  if (test::failures == 0) std::printf("environment tests passed\n");
  return test::failures == 0 ? 0 : 1;
}
