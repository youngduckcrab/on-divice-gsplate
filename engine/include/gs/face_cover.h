#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "gs/dataset.h"
#include "gs/splat_scene.h"

namespace gs {

// An orthographic picture of part of a splat scene. Pixel (x, y) looks along
// `forward` through origin + ((x + 0.5) - width/2) * pixel * right
//                         + ((y + 0.5) - height/2) * pixel * down.
struct OrthoView {
  float origin[3] = {0, 0, 0};
  float right[3] = {1, 0, 0};
  float down[3] = {0, 1, 0};
  float forward[3] = {0, 0, 1};
  float pixel = 0.01f;  // scene units per pixel
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<float> rgb;    // 3 per pixel, 0..1, over the background
  std::vector<float> alpha;  // 1 - transmittance
  std::vector<float> depth;  // along forward from the origin's plane, where transmittance falls below 1/2; NaN if never

  // The scene point at pixel coordinates (x, y) (continuous, pixel centres at
  // +0.5) and depth d.
  void point(float x, float y, float d, float out[3]) const;
  // Depth near (x, y): the pixel's own, else the nearest opaque one within
  // `reach` pixels. NaN when there is none.
  float depth_near(float x, float y, int reach) const;
  std::vector<uint8_t> rgb8() const;
};

// Splats `subset` (indices into scene; every splat when empty) front to back,
// with the viewer's low-pass filter and SH colour seen along `forward`.
void render_ortho(const SplatScene& scene, const std::vector<uint32_t>& subset, OrthoView& view,
                  const float background[3]);

// Where a person's head is, from the splats alone (see estimate_head for a
// standing person; shorter subjects, such as a head-and-shoulders scan, get
// a head at their top). `up` is the scene's up direction.
HeadRegion find_head(const SplatScene& scene, const float up[3]);

// Pictures of the head from around it, looking level at it, for a face
// detector. Every view has the same scale, so a face seen from the front
// shows the widest eye distance.
struct FaceViews {
  HeadRegion head;
  float up[3] = {0, 1, 0};
  std::vector<uint32_t> subset;  // the head's splats
  std::vector<OrthoView> views;
  std::vector<float> azimuth;    // per view, radians
};
// `count` views evenly around the head; or, with `toward` set, three views
// within +-20 degrees of looking along `toward` (e.g. the viewer's camera).
FaceViews render_face_views(const SplatScene& scene, const float up[3], int count, uint32_t size,
                            const float* toward = nullptr);

// One face a detector found in one of the views, in that view's pixels.
struct FaceLandmarks {
  int view = 0;
  float eyes[2][2] = {{0, 0}, {0, 0}};  // either order
  bool has_mouth = false;
  float mouth[2] = {0, 0};
};
// [{"view": 3, "eyes": [[x, y], [x, y]], "mouth": [x, y]}, ...]; mouth optional.
std::vector<FaceLandmarks> parse_face_landmarks(std::string_view json);

// The face on the model: a frame with `right` (the viewer's right when facing
// the person), `up` and `forward` (out of the face), centred between the eyes
// on the surface, in eye distances.
struct FacePlacement {
  bool found = false;
  bool detected = false;  // from landmarks; otherwise guessed from the head's shape
  int views_used = 0;      // detections averaged
  int views_agreeing = 0;  // detections that put the face in the same place
  float center[3] = {0, 0, 0};
  float right[3] = {1, 0, 0};
  float up[3] = {0, 1, 0};
  float forward[3] = {0, 0, 1};
  float eye_distance = 0;
};
// Each detection's eyes and mouth put on the model's surface through its
// picture's depth; the detection most others agree with (in 3D) picks the
// face, and the most frontal of those (widest eye distance) are averaged.
FacePlacement place_face(const SplatScene& scene, const FaceViews& views, const std::vector<FaceLandmarks>& faces);
// Without a detection: the face looks along `facing` (horizontal part used),
// eyes at the usual height and spacing for the head's size.
FacePlacement guess_face(const SplatScene& scene, const FaceViews& views, const float facing[3]);

enum class FaceStyle { Mosaic, Blur, Sunglasses, Mask, Smile, Eyes };
// "mosaic", "blur", "sunglasses", "mask", "smile", "eyes" (only the eyes blurred)
bool parse_face_style(std::string_view name, FaceStyle* style);

struct FaceCoverOptions {
  FaceStyle style = FaceStyle::Mosaic;
  float size = 1.0f;   // scales the cover
  float raise = 0.0f;  // moves it up (negative: down), in eye distances
};

struct FaceCoverStats {
  uint32_t added = 0;       // new splats forming the cover
  uint32_t recoloured = 0;  // the face's own splats under it, given its colour
};

// A copy of `scene` with the face covered: the cover is a layer of small flat
// splats following the face's surface (mosaic, blur, mask, smile), a pair of
// sunglasses in front of it, or (eyes) a blur of the face's own splats over
// the eyes, adding none. Under the full-face styles the face's own splats
// take the cover's colour and lose their view-dependent colour, so the face
// does not come back from another angle or by deleting the cover.
SplatScene cover_face(const SplatScene& scene, const FacePlacement& face, const FaceCoverOptions& options,
                      FaceCoverStats* stats = nullptr);

}  // namespace gs
