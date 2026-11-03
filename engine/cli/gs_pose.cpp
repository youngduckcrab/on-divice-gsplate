// Re-posing on the desktop, as the Android viewer does it.
//
//   gs_pose <scene> views <out_dir> [--count 8] [--size 512]
//       pictures of the whole body from around it (view_00.jpg, ...) for a body pose detector
//   gs_pose <scene> pose <landmarks.json> <out.ply> [--pose attention|...|all] [--time 0]
//           [--count 8] [--size 512] [--skeleton <out.json>]
//       fits the skeleton to the detector's landmarks (the format parse_pose_landmarks
//       reads, in the pixels of the views above), binds the splats and writes the
//       posed scene; --pose all writes <out>-<pose>.ply for every pose.
//
// Up is -y for .ply (RDF) scenes and +y for RUB ones, as the viewer shows them.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "gs/png_write.h"
#include "gs/pose.h"
#include "gs/splat_scene.h"

namespace {

std::vector<uint8_t> read_all(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

const char* option(int argc, char** argv, const char* name, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  }
  return fallback;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

const char* const kBoneNames[gs::kBoneCount] = {
    "pelvis",         "chest",          "head",          "left collarbone", "left upper arm",  "left forearm",
    "left hand",      "right collarbone", "right upper arm", "right forearm", "right hand",     "left thigh",
    "left shin",      "left foot",      "right thigh",   "right shin",      "right foot"};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <scene> views <out_dir> | pose <landmarks.json> <out.ply> [options]\n", argv[0]);
    return 2;
  }
  try {
    const std::vector<uint8_t> bytes = read_all(argv[1]);
    const gs::SplatScene scene = gs::load_splat_file(bytes.data(), bytes.size());
    const float up[3] = {0.0f, scene.convention == gs::AxisConvention::RDF ? -1.0f : 1.0f, 0.0f};
    const int count = std::atoi(option(argc, argv, "--count", "8"));
    const uint32_t size = uint32_t(std::atoi(option(argc, argv, "--size", "512")));
    const std::string mode = argv[2];
    auto t0 = std::chrono::steady_clock::now();
    const gs::BodyViews views = gs::render_body_views(scene, up, count, size);
    std::printf("views: %.0f ms (%u splats)\n", ms_since(t0), scene.count);
    if (!views.found) throw std::runtime_error("no body found");
    std::printf("body: height %.2f, centre (%.2f %.2f %.2f)\n", views.height, views.centre[0], views.centre[1],
                views.centre[2]);

    if (mode == "views") {
      const std::string dir = argv[3];
      for (size_t i = 0; i < views.views.size(); ++i) {
        const gs::OrthoView& v = views.views[i];
        char name[64];
        std::snprintf(name, sizeof(name), "/view_%02zu.jpg", i);
        gs::write_file(dir + name, gs::encode_jpeg_rgb(v.rgb8().data(), v.width, v.height, 95));
      }
      std::printf("wrote %zu views to %s\n", views.views.size(), dir.c_str());
      return 0;
    }
    if (mode != "pose" || argc < 5) throw std::runtime_error("unknown mode " + mode);

    const std::vector<uint8_t> text = read_all(argv[3]);
    const auto marks = gs::parse_pose_landmarks(std::string_view(reinterpret_cast<const char*>(text.data()), text.size()));
    t0 = std::chrono::steady_clock::now();
    const gs::Skeleton sk = gs::fit_skeleton(views, marks);
    std::printf("skeleton: %.0f ms, %s, %d views (%d mirrored), error %.1f cm\n", ms_since(t0),
                sk.found ? "found" : "NOT found", sk.views_used, sk.views_flipped, sk.error * 100.0f);
    if (!sk.found) return 1;
    std::printf("forward (%.2f %.2f %.2f), left (%.2f %.2f %.2f)\n", sk.forward[0], sk.forward[1], sk.forward[2],
                sk.left[0], sk.left[1], sk.left[2]);
    for (int b = 0; b < gs::kBoneCount; ++b) {
      const auto& bone = sk.bones[b];
      std::printf("  %-16s (%.2f %.2f %.2f) -> (%.2f %.2f %.2f)\n", kBoneNames[b], bone.pivot[0], bone.pivot[1],
                  bone.pivot[2], bone.end[0], bone.end[1], bone.end[2]);
    }
    if (const char* path = option(argc, argv, "--skeleton", nullptr)) {
      std::string json = "{\"points\":[";
      for (int j = 0; j < gs::kPosePoints; ++j) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s[%.4f,%.4f,%.4f,%.2f]", j ? "," : "", sk.points[j][0], sk.points[j][1],
                      sk.points[j][2], sk.confidence[j]);
        json += buf;
      }
      json += "]}\n";
      gs::write_file(path, std::vector<uint8_t>(json.begin(), json.end()));
    }
    t0 = std::chrono::steady_clock::now();
    uint32_t split = 0;
    const gs::SplatScene scene_in = scene;
    const gs::SplatScene posable = gs::split_at_joints(scene_in, gs::skin_splats(scene_in, sk), sk, &split);
    const gs::Skin skin = gs::skin_splats(posable, sk);
    std::printf("skin: %.0f ms (%u splats added at joints)\n", ms_since(t0), split);
    if (const char* probe = option(argc, argv, "--probe", nullptr)) {
      float q[3] = {0, 0, 0};
      std::sscanf(probe, "%f,%f,%f", &q[0], &q[1], &q[2]);
      uint32_t best = 0;
      float best_d = 1e30f;
      for (uint32_t i = 0; i < posable.count; ++i) {
        float d = 0;
        for (int k = 0; k < 3; ++k) d += (posable.positions[i * 3 + k] - q[k]) * (posable.positions[i * 3 + k] - q[k]);
        if (d < best_d) {
          best_d = d;
          best = i;
        }
      }
      std::printf("probe: splat %u at (%.3f %.3f %.3f):", best, posable.positions[best * 3], posable.positions[best * 3 + 1],
                  posable.positions[best * 3 + 2]);
      for (int k = 0; k < 4; ++k) {
        if (skin.weights[best * 4 + k]) std::printf(" %s %d", kBoneNames[skin.bones[best * 4 + k]], skin.weights[best * 4 + k]);
      }
      std::printf("\n");
      for (int b = 0; b < gs::kBoneCount; ++b) std::printf("  %s radius %.3f half %.3f/%.3f\n", kBoneNames[b],
                                                           sk.bones[b].radius, sk.bones[b].half_width, sk.bones[b].half_depth);
    }
    const std::string which = option(argc, argv, "--pose", "attention");
    const float time = float(std::atof(option(argc, argv, "--time", "0")));
    std::vector<std::string> names;
    if (which == "all") {
      names = {"attention", "hands_on_hips", "arms_up", "bow", "wave", "walk"};
    } else {
      if (!gs::known_pose(which)) throw std::runtime_error("unknown pose " + which);
      names = {which};
    }
    std::string out = argv[4];
    if (names.size() > 1 && out.size() > 4 && out.substr(out.size() - 4) == ".ply") out.resize(out.size() - 4);
    for (const std::string& name : names) {
      const gs::SplatScene posed = gs::apply_pose(posable, skin, gs::pose_skeleton(sk, name, time));
      const std::string path = names.size() > 1 ? out + "-" + name + ".ply" : out;
      gs::write_file(path, gs::save_ply(posed));
      std::printf("wrote %s\n", path.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
