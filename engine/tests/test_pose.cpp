// Re-posing on a synthetic person standing in a T-pose: their skeleton from
// landmarks a body pose detector would report on views from around them
// (mirrored in the views from behind, as detectors tend to, one view
// garbage), the splats' bones, the named poses, the CPU posing and the GPU
// skinning against it.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "gs/camera.h"
#include "gs/offscreen.h"
#include "gs/pose.h"
#include "gs/splat_renderer.h"
#include "test_util.h"

namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr float kPi = 3.14159265358979f;

struct V {
  float x = 0, y = 0, z = 0;
};
V add(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V mul(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float len(V a) { return std::sqrt(dot(a, a)); }
V norm(V a) { return mul(a, 1.0f / len(a)); }
V from(const float* p) { return {p[0], p[1], p[2]}; }

// A 1.7 m person in a T-pose facing `facing` (horizontal), y up.
struct Person {
  gs::SplatScene scene;
  V facing, left;
  V points[gs::kPosePoints];   // true landmark positions
  std::vector<uint32_t> forearm, torso_side, head, elbow;  // splats in places worth checking
};

Person make_person(float yaw) {
  Person p;
  p.facing = {std::sin(yaw), 0.0f, std::cos(yaw)};
  p.left = {p.facing.z, 0.0f, -p.facing.x};  // up x facing
  auto at = [&](float l, float u, float f) { return add(add(mul(p.left, l), V{0, u, 0}), mul(p.facing, f)); };
  const float lm[gs::kPosePoints][3] = {
      {0, 1.58f, 0.1f},                                                               // 0 nose
      {0.015f, 1.62f, 0.085f}, {0.032f, 1.62f, 0.08f}, {0.048f, 1.62f, 0.07f},        // left eye
      {-0.015f, 1.62f, 0.085f}, {-0.032f, 1.62f, 0.08f}, {-0.048f, 1.62f, 0.07f},     // right eye
      {0.075f, 1.6f, 0.0f}, {-0.075f, 1.6f, 0.0f},                                    // ears
      {0.025f, 1.53f, 0.08f}, {-0.025f, 1.53f, 0.08f},                                // mouth
      {0.19f, 1.42f, 0}, {-0.19f, 1.42f, 0},                                          // shoulders
      {0.47f, 1.42f, 0}, {-0.47f, 1.42f, 0},                                          // elbows
      {0.73f, 1.42f, 0}, {-0.73f, 1.42f, 0},                                          // wrists
      {0.81f, 1.41f, -0.02f}, {-0.81f, 1.41f, -0.02f},                                // pinkies
      {0.82f, 1.42f, 0.02f}, {-0.82f, 1.42f, 0.02f},                                  // index fingers
      {0.78f, 1.43f, 0.04f}, {-0.78f, 1.43f, 0.04f},                                  // thumbs
      {0.1f, 0.92f, 0}, {-0.1f, 0.92f, 0},                                            // hips
      {0.1f, 0.5f, 0.02f}, {-0.1f, 0.5f, 0.02f},                                      // knees
      {0.1f, 0.08f, 0}, {-0.1f, 0.08f, 0},                                            // ankles
      {0.1f, 0.03f, -0.05f}, {-0.1f, 0.03f, -0.05f},                                  // heels
      {0.1f, 0.02f, 0.15f}, {-0.1f, 0.02f, 0.15f},                                    // toes
  };
  for (int j = 0; j < gs::kPosePoints; ++j) p.points[j] = at(lm[j][0], lm[j][1], lm[j][2]);

  std::vector<V> pos, rgb;
  std::vector<float> sig;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  // A tube of radius r from a to b (local coordinates), its surface.
  auto tube = [&](V a, V b, float r, int n, V colour, std::vector<uint32_t>* mid) {
    const V axis = sub(b, a);
    const V helper = std::fabs(axis.y) < 0.9f * len(axis) ? V{0, 1, 0} : V{1, 0, 0};
    V e1 = norm(V{axis.y * helper.z - axis.z * helper.y, axis.z * helper.x - axis.x * helper.z,
                  axis.x * helper.y - axis.y * helper.x});
    const V ax = norm(axis);
    V e2 = {ax.y * e1.z - ax.z * e1.y, ax.z * e1.x - ax.x * e1.z, ax.x * e1.y - ax.y * e1.x};
    for (int i = 0; i < n; ++i) {
      const float t = uni(rng), a2 = 2 * kPi * uni(rng);
      const V q = add(add(a, mul(axis, t)), add(mul(e1, r * std::cos(a2)), mul(e2, r * std::sin(a2))));
      if (mid != nullptr && t > 0.4f && t < 0.6f) mid->push_back(uint32_t(pos.size()));
      pos.push_back(q);
      rgb.push_back(colour);
      sig.push_back(0.012f);
    }
  };
  auto L = [&](float l, float u, float f) { return at(l, u, f); };
  // Torso: an elliptical tube from the hips to the shoulders.
  for (int i = 0; i < 6000; ++i) {
    const float u = 0.88f + 0.56f * uni(rng), a2 = 2 * kPi * uni(rng);
    const float hw = 0.17f, hd = 0.1f;
    const V q = L(hw * std::cos(a2), u, hd * std::sin(a2));
    if (u > 1.12f && u < 1.28f && std::fabs(std::cos(a2)) > 0.97f) p.torso_side.push_back(uint32_t(pos.size()));
    pos.push_back(q);
    rgb.push_back({0.2f, 0.4f, 0.8f});
    sig.push_back(0.015f);
  }
  // Head and neck.
  for (int i = 0; i < 3000; ++i) {
    const float t = (float(i) + 0.5f) / 3000.0f;
    const float polar = std::acos(1.0f - 2.0f * t), az = kPi * (3.0f - std::sqrt(5.0f)) * float(i);
    p.head.push_back(uint32_t(pos.size()));
    pos.push_back(L(0.08f * std::sin(polar) * std::cos(az), 1.6f + 0.1f * std::cos(polar),
                    0.09f * std::sin(polar) * std::sin(az)));
    rgb.push_back({0.85f, 0.7f, 0.6f});
    sig.push_back(0.01f);
  }
  tube(L(0, 1.42f, 0), L(0, 1.52f, 0), 0.05f, 500, {0.85f, 0.7f, 0.6f}, nullptr);
  // Arms (the left forearm's middle marked), hands; legs, feet.
  for (float s : {1.0f, -1.0f}) {
    tube(L(s * 0.19f, 1.42f, 0), L(s * 0.47f, 1.42f, 0), 0.05f, 1500, {0.9f, 0.3f, 0.2f}, nullptr);
    std::vector<uint32_t> fore;
    tube(L(s * 0.47f, 1.42f, 0), L(s * 0.73f, 1.42f, 0), 0.04f, 1300, {0.9f, 0.6f, 0.2f}, &fore);
    if (s > 0) p.forearm = fore;
    tube(L(s * 0.73f, 1.42f, 0), L(s * 0.83f, 1.42f, 0), 0.035f, 400, {0.85f, 0.7f, 0.6f}, nullptr);
    tube(L(s * 0.1f, 0.92f, 0), L(s * 0.1f, 0.5f, 0.02f), 0.075f, 2500, {0.2f, 0.2f, 0.25f}, nullptr);
    tube(L(s * 0.1f, 0.5f, 0.02f), L(s * 0.1f, 0.08f, 0), 0.055f, 2200, {0.2f, 0.2f, 0.25f}, nullptr);
    tube(L(s * 0.1f, 0.05f, -0.05f), L(s * 0.1f, 0.04f, 0.16f), 0.04f, 500, {0.1f, 0.1f, 0.1f}, nullptr);
  }
  // Around the left elbow.
  for (uint32_t i = 0; i < pos.size(); ++i) {
    if (len(sub(pos[i], L(0.47f, 1.42f, 0))) < 0.06f) p.elbow.push_back(i);
  }
  gs::SplatScene& sc = p.scene;
  sc.resize(uint32_t(pos.size()), 0);
  for (uint32_t i = 0; i < sc.count; ++i) {
    sc.positions[i * 3] = pos[i].x;
    sc.positions[i * 3 + 1] = pos[i].y;
    sc.positions[i * 3 + 2] = pos[i].z;
    for (int k = 0; k < 3; ++k) sc.log_scales[i * 3 + k] = std::log(sig[i]);
    sc.rotations[i * 4] = 1.0f;
    sc.opacity_logits[i] = 3.0f;
    sc.sh_dc[i * 3] = (rgb[i].x - 0.5f) / kShC0;
    sc.sh_dc[i * 3 + 1] = (rgb[i].y - 0.5f) / kShC0;
    sc.sh_dc[i * 3 + 2] = (rgb[i].z - 0.5f) / kShC0;
  }
  return p;
}

constexpr int kMirror[gs::kPosePoints] = {0,  4,  5,  6,  1,  2,  3,  8,  7,  10, 9,  12, 11, 14, 13, 16, 15,
                                          18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31};

// What a detector would say: the true points, labelled as if the person
// faced the camera (so mirrored when seen from behind), a little noise; one
// view's points scattered.
std::string landmarks_json(const Person& p, const gs::BodyViews& views, int garbage_view, int* behind) {
  std::mt19937 rng(9);
  std::normal_distribution<float> noise(0.0f, 0.6f);
  std::uniform_real_distribution<float> anywhere(0.0f, float(views.views[0].width));
  std::string json = "[";
  *behind = 0;
  for (size_t k = 0; k < views.views.size(); ++k) {
    const gs::OrthoView& v = views.views[k];
    const bool from_behind = dot(from(v.forward), p.facing) > 0.1f;
    *behind += from_behind && int(k) != garbage_view;
    json += (k ? ",{\"view\":" : "{\"view\":") + std::to_string(k) + ",\"points\":[";
    for (int j = 0; j < gs::kPosePoints; ++j) {
      const V q = p.points[from_behind ? kMirror[j] : j];
      const V rel = sub(q, from(v.origin));
      float x = dot(rel, from(v.right)) / v.pixel + 0.5f * float(v.width) + noise(rng);
      float y = dot(rel, from(v.down)) / v.pixel + 0.5f * float(v.height) + noise(rng);
      float like = 0.9f;
      if (int(k) == garbage_view) {
        x = anywhere(rng);
        y = anywhere(rng);
        like = 0.6f;
      }
      char buf[96];
      std::snprintf(buf, sizeof(buf), "%s[%.2f,%.2f,%.2f]", j ? "," : "", x, y, like);
      json += buf;
    }
    json += "]}";
  }
  return json + "]";
}

float weight_of(const gs::Skin& skin, uint32_t i, int bone) {
  float w = 0;
  for (int k = 0; k < 4; ++k) {
    if (skin.bones[i * 4 + k] == bone) w += skin.weights[i * 4 + k] / 255.0f;
  }
  return w;
}

float mean_weight(const gs::Skin& skin, const std::vector<uint32_t>& splats, int bone) {
  float s = 0;
  for (uint32_t i : splats) s += weight_of(skin, i, bone);
  return splats.empty() ? 0.0f : s / float(splats.size());
}

V moved(const gs::BonePose& pose, int b, V q) {
  const float* r = pose.rotation[b];
  const float* t = pose.translation[b];
  return {r[0] * q.x + r[1] * q.y + r[2] * q.z + t[0], r[3] * q.x + r[4] * q.y + r[5] * q.z + t[1],
          r[6] * q.x + r[7] * q.y + r[8] * q.z + t[2]};
}

void test_fit_and_skin(const Person& p, gs::Skeleton* out, gs::Skin* skin_out) {
  const float up[3] = {0, 1, 0};
  const gs::BodyViews views = gs::render_body_views(p.scene, up, 8, 256);
  EXPECT(views.found && views.views.size() == 8, "body views");
  EXPECT(std::fabs(views.height - 1.72f) < 0.06f, "body height %.3f", views.height);
  int behind = 0;
  const auto marks = gs::parse_pose_landmarks(landmarks_json(p, views, 3, &behind));
  EXPECT(marks.size() == 8, "parsed %zu views", marks.size());
  const gs::Skeleton sk = gs::fit_skeleton(views, marks);
  EXPECT(sk.found, "skeleton found");
  float worst = 0;
  for (int j : {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}) {
    const float e = len(sub(from(sk.points[j]), p.points[j]));
    worst = std::max(worst, e);
  }
  EXPECT(worst < 0.02f, "joints within 2 cm of the truth: worst %.3f m", worst);
  EXPECT(dot(from(sk.forward), p.facing) > 0.97f, "facing found: %.3f", dot(from(sk.forward), p.facing));
  EXPECT(dot(from(sk.left), p.left) > 0.97f, "left found: %.3f", dot(from(sk.left), p.left));
  std::printf("skeleton: worst joint %.1f cm, error %.1f cm, %d views, %d mirrored (truth %d)\n", worst * 100,
              sk.error * 100, sk.views_used, sk.views_flipped, behind);
  const gs::Skin skin = gs::skin_splats(p.scene, sk);
  EXPECT(skin.count == p.scene.count, "skin for every splat");
  const float fore = mean_weight(skin, p.forearm, gs::kLeftForearm);
  const float side = mean_weight(skin, p.torso_side, gs::kChest) + mean_weight(skin, p.torso_side, gs::kPelvis);
  const float head = mean_weight(skin, p.head, gs::kHead);
  const float elbow_u = mean_weight(skin, p.elbow, gs::kLeftUpperArm);
  const float elbow_f = mean_weight(skin, p.elbow, gs::kLeftForearm);
  EXPECT(fore > 0.9f, "left forearm follows its bone: %.2f", fore);
  EXPECT(side > 0.9f, "the torso's side (under the arm) follows the torso: %.2f", side);
  EXPECT(head > 0.85f, "head follows the head: %.2f", head);
  EXPECT(elbow_u > 0.2f && elbow_f > 0.2f, "the elbow blends both arm bones: %.2f %.2f", elbow_u, elbow_f);
  for (uint32_t i = 0; i < skin.count; i += 997) {
    int sum = 0;
    for (int k = 0; k < 4; ++k) sum += skin.weights[i * 4 + k];
    EXPECT(sum == 255, "weights sum to 255 (splat %u: %d)", i, sum);
  }
  *out = sk;
  *skin_out = skin;
}

void test_poses(const Person& p, const gs::Skeleton& sk, const gs::Skin& skin) {
  const V U = {0, 1, 0};
  // Rest: nothing moves.
  const gs::SplatScene rest = gs::apply_pose(p.scene, skin, gs::pose_skeleton(sk, "rest"));
  float off = 0;
  for (uint32_t i = 0; i < rest.count * 3; ++i) off = std::max(off, std::fabs(rest.positions[i] - p.scene.positions[i]));
  EXPECT(off < 1e-5f, "rest pose leaves the splats: %g", off);
  // Arms down: the wrists hang below the shoulders by about the arm's length.
  const gs::BonePose down = gs::pose_skeleton(sk, "attention");
  const V wrist = moved(down, gs::kLeftForearm, from(sk.points[15]));
  const V shoulder = moved(down, gs::kLeftUpperArm, from(sk.points[11]));
  const float drop = dot(sub(shoulder, wrist), U);
  const float out = dot(sub(wrist, shoulder), p.left);
  EXPECT(drop > 0.5f && drop < 0.56f, "arms down: wrist %.3f m below the shoulder", drop);
  EXPECT(out > 0.0f && out < 0.1f, "arms down: wrist a little out to the side (%.3f m)", out);
  // Arms up: above the head.
  const gs::BonePose up = gs::pose_skeleton(sk, "arms_up");
  const V hand = moved(up, gs::kLeftForearm, from(sk.points[15]));
  EXPECT(dot(hand, U) > 1.85f, "arms up: wrist at %.2f m", dot(hand, U));
  // Hands on hips: the wrist at the waist's side.
  const gs::BonePose hips = gs::pose_skeleton(sk, "hands_on_hips");
  const V w2 = moved(hips, gs::kLeftForearm, from(sk.points[15]));
  EXPECT(std::fabs(dot(w2, U) - 1.0f) < 0.08f && dot(w2, p.left) > 0.15f && dot(w2, p.left) < 0.3f,
         "hands on hips: wrist at height %.2f, %.2f m to the side", dot(w2, U), dot(w2, p.left));
  // Walking keeps the lowest foot on the floor and moves with time.
  for (float t : {0.0f, 0.3f, 0.71f}) {
    const gs::BonePose walk = gs::pose_skeleton(sk, "walk", t);
    float low = 1e9f;
    for (int foot : {gs::kLeftFoot, gs::kRightFoot}) {
      for (int j : {foot == gs::kLeftFoot ? 27 : 28, foot == gs::kLeftFoot ? 29 : 30, foot == gs::kLeftFoot ? 31 : 32}) {
        low = std::min(low, dot(moved(walk, foot, from(sk.points[j])), U));
      }
    }
    float rest_low = 1e9f;
    for (int j : {27, 28, 29, 30, 31, 32}) rest_low = std::min(rest_low, sk.points[j][1]);
    EXPECT(std::fabs(low - rest_low) < 1e-3f, "walk t=%.2f: feet on the floor (%.4f vs %.4f)", t, low, rest_low);
  }
  const V k0 = moved(gs::pose_skeleton(sk, "walk", 0.0f), gs::kLeftShin, from(sk.points[27]));
  const V k1 = moved(gs::pose_skeleton(sk, "walk", 0.27f), gs::kLeftShin, from(sk.points[27]));
  EXPECT(len(sub(k0, k1)) > 0.1f, "walking moves the feet: %.3f m", len(sub(k0, k1)));
  EXPECT(gs::pose_moves("walk") && gs::pose_moves("wave") && !gs::pose_moves("bow"), "moving poses");
  EXPECT(gs::known_pose("bow") && !gs::known_pose("dance"), "pose names");

  // CPU posing: a splat bound to one bone moves with it, rotation and SH turned.
  gs::SplatScene one;
  one.resize(1, 1);
  const V q = from(sk.points[14]);  // somewhere on the right arm
  one.positions = {q.x, q.y, q.z};
  one.rotations = {1, 0, 0, 0};
  one.sh_rest = {0.3f, 0.1f, 0.0f, -0.2f, 0.4f, 0.1f, 0.05f, 0.0f, 0.2f};
  gs::Skin s1;
  s1.count = 1;
  s1.bones = {gs::kRightForearm, 0, 0, 0};
  s1.weights = {255, 0, 0, 0};
  const gs::BonePose wave = gs::pose_skeleton(sk, "wave", 0.2f);
  const gs::SplatScene posed = gs::apply_pose(one, s1, wave);
  const V expect = moved(wave, gs::kRightForearm, q);
  EXPECT(len(sub(from(posed.positions.data()), expect)) < 1e-4f, "one-bone splat follows its bone");
  // SH band 1 seen along the turned direction equals the rest's along the original.
  const float* R = wave.rotation[gs::kRightForearm];
  const V d = norm(V{0.3f, -0.5f, 0.8f});
  const V rd = {R[0] * d.x + R[1] * d.y + R[2] * d.z, R[3] * d.x + R[4] * d.y + R[5] * d.z,
                R[6] * d.x + R[7] * d.y + R[8] * d.z};
  auto band1 = [](const gs::SplatScene& s, V dir, int c) {
    return -dir.y * s.sh_rest[0 * 3 + c] + dir.z * s.sh_rest[1 * 3 + c] - dir.x * s.sh_rest[2 * 3 + c];
  };
  for (int c = 0; c < 3; ++c) {
    EXPECT(std::fabs(band1(posed, rd, c) - band1(one, d, c)) < 1e-4f, "SH band 1 turned (channel %d)", c);
  }
}

// The GPU skinning draws what the CPU posing makes.
void test_gpu(const Person& p, const gs::Skin& skin, const gs::Skeleton& sk) {
  gs::VkContext ctx;
  const uint32_t w = 160, h = 160;
  gs::OrbitCamera orbit;
  orbit.target[0] = 0.0f;
  orbit.target[1] = 1.0f;
  orbit.target[2] = 0.0f;
  orbit.distance = 4.5f;
  orbit.yaw = 0.4f;
  orbit.pitch = 0.1f;
  const gs::Camera camera = orbit.camera(w, h);
  auto draw = [&](const gs::SplatScene& scene, const std::vector<uint32_t>* packed, const gs::SplatSkinning* skinning) {
    auto gpu = gs::upload_scene(ctx, scene, packed);
    gs::OffscreenTarget target(ctx, w, h);
    gs::SplatRenderer renderer(ctx, target.render_pass());
    renderer.attach(gpu.get());
    if (skinning != nullptr) renderer.set_skinning(*skinning);
    const float clear[4] = {0, 0, 0, 1};
    ctx.run([&](VkCommandBuffer cmd) {
      renderer.record_prepare(cmd, camera);
      target.record(cmd, clear, [&](VkCommandBuffer c) { renderer.record_draw(c, w, h); });
    });
    return target.pixels();
  };
  const std::vector<uint32_t> packed = skin.packed();
  for (const char* name : {"attention", "walk"}) {
    const gs::BonePose pose = gs::pose_skeleton(sk, name, 0.3f);
    const gs::SplatSkinning skinning = gs::skinning_for(pose, skin.count);
    const auto gpu = draw(p.scene, &packed, &skinning);
    const auto cpu = draw(gs::apply_pose(p.scene, skin, pose), nullptr, nullptr);
    double diff = 0;
    int big = 0;
    for (size_t i = 0; i < gpu.size(); i += 4) {
      for (int c = 0; c < 3; ++c) {
        const int d = std::abs(int(gpu[i + c]) - int(cpu[i + c]));
        diff += d;
        big += d > 40;
      }
    }
    diff /= double(w * h * 3);
    EXPECT(diff < 1.5 && big < int(w * h * 3 / 200), "%s: GPU skinning matches the CPU posing (mean %.2f, %d off)", name,
           diff, big);
  }
  // Skin uploaded but posing off: the rest pose.
  gs::SplatSkinning off;
  const auto skinned_rest = draw(p.scene, &packed, &off);
  const auto plain = draw(p.scene, nullptr, nullptr);
  EXPECT(skinned_rest == plain, "posing off draws the rest pose");
}

}  // namespace

int main() {
  try {
    for (float yaw : {0.0f, 2.3f}) {
      const Person p = make_person(yaw);
      gs::Skeleton sk;
      gs::Skin skin;
      test_fit_and_skin(p, &sk, &skin);
      if (!sk.found) continue;
      test_poses(p, sk, skin);
      if (yaw == 0.0f) test_gpu(p, skin, sk);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    return 1;
  }
  if (test::failures == 0) std::printf("pose tests passed\n");
  return test::failures == 0 ? 0 : 1;
}
