#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gs/face_cover.h"
#include "gs/splat_renderer.h"
#include "gs/splat_scene.h"

namespace gs {

// Re-posing a trained person: find their skeleton from pictures of the model
// (a body pose detector on views from around them, put together in 3D), bind
// every splat to the bones near it, and turn the bones.

// Pictures of a whole standing person from around them, level, all at the
// same scale, for a body pose detector.
struct BodyViews {
  bool found = false;
  float up[3] = {0, 1, 0};
  float centre[3] = {0, 0, 0};  // middle of the body (half way up)
  float height = 0;             // extent along up
  std::vector<OrthoView> views;
};
// `count` views evenly around the body, `size` pixels square.
BodyViews render_body_views(const SplatScene& scene, const float up[3], int count, uint32_t size);

// The 33 points of the BlazePose topology (ML Kit pose detection, MediaPipe
// pose landmarker): 0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists,
// 23/24 hips, 25/26 knees, 27/28 ankles, 31/32 toes, ... (odd: the person's
// left), in one view's pixels.
constexpr int kPosePoints = 33;
struct PoseLandmarks {
  int view = 0;
  float x[kPosePoints] = {};
  float y[kPosePoints] = {};
  float likelihood[kPosePoints] = {};  // 0..1: how sure the detector is
};
// [{"view": 3, "points": [[x, y, likelihood], ... 33]}, ...]
std::vector<PoseLandmarks> parse_pose_landmarks(std::string_view json);

enum Bone : int {
  kPelvis,
  kChest,
  kHead,
  kLeftUpperArm,
  kLeftForearm,
  kLeftHand,
  kRightUpperArm,
  kRightForearm,
  kRightHand,
  kLeftThigh,
  kLeftShin,
  kLeftFoot,
  kRightThigh,
  kRightShin,
  kRightFoot,
  kBoneCount
};

struct Skeleton {
  bool found = false;
  int views_used = 0;     // views with a detection
  int views_flipped = 0;  // of those, views whose left and right the detector had swapped
  float error = 0;        // RMS distance of the points from their sight lines (scene units)
  float points[kPosePoints][3] = {};  // the 33 points in 3D
  float confidence[kPosePoints] = {};
  // The body's frame: up, forward (out of the chest), left (the person's).
  float up[3] = {0, 1, 0};
  float forward[3] = {0, 0, 1};
  float left[3] = {1, 0, 0};
  float height = 0;
  struct BoneRest {
    int parent = -1;
    float pivot[3] = {};   // the joint it turns about
    float end[3] = {};     // where it points to
    float radius = 0;      // thickness of the body around it (limbs, head)
    float half_width = 0;  // torso bones: half width and depth across (left, forward)
    float half_depth = 0;
  };
  BoneRest bones[kBoneCount];
};
// The 3D points from every view's detection: each point is on a line through
// the model per view; the left/right labelling of each view is chosen so the
// views agree (detectors often mirror a person seen from behind), then the
// whole skeleton's left is checked against which way the nose points.
Skeleton fit_skeleton(const BodyViews& views, const std::vector<PoseLandmarks>& landmarks);

// Up to 4 bones per splat (the nearest, measured against each bone's
// thickness, blended with its neighbours in the hierarchy near joints).
struct Skin {
  uint32_t count = 0;
  std::vector<uint8_t> bones;    // 4 per splat
  std::vector<uint8_t> weights;  // 4 per splat, summing to 255
  // 2 uint32 per splat for the GPU: bone indices, weights (a byte each).
  std::vector<uint32_t> packed() const;
};
Skin skin_splats(const SplatScene& scene, const Skeleton& skeleton);

// Each bone's rigid motion: p -> rotation p + translation (rotation row-major).
struct BonePose {
  float rotation[kBoneCount][9];
  float translation[kBoneCount][3];
  BonePose();  // every bone at rest
};

// Named poses: "rest", "attention" (arms down), "hands_on_hips", "arms_up",
// "bow", and moving ones, "wave" and "walk", which follow `time` (seconds).
// The feet are kept on the floor.
bool known_pose(std::string_view name);
bool pose_moves(std::string_view name);
BonePose pose_skeleton(const Skeleton& skeleton, std::string_view name, float time = 0.0f);

// For SplatRenderer::set_skinning: the first `count` splats posed.
SplatSkinning skinning_for(const BonePose& pose, uint32_t count);

// The posed scene on the CPU (for saving it): positions blended over the
// splat's bones, rotations turned by the blend of theirs, view-dependent
// colour turned with them (to degree 1).
SplatScene apply_pose(const SplatScene& scene, const Skin& skin, const BonePose& pose);

}  // namespace gs
