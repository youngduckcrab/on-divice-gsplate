// Face covering on the desktop, as the Android viewer does it.
//
//   gs_face <scene> views <out_dir> [--count 12] [--size 384]
//       pictures of the head from around it (view_00.jpg, ...) for a face detector
//   gs_face <scene> cover <landmarks.json> <out.ply> [--style mosaic|blur|eyes|sunglasses|mask|smile]
//           [--size 1] [--raise 0] [--count 12] [--view-size 384] [--guess <view>]
//       places the face from the detector's landmarks (the format parse_face_landmarks
//       reads, in the pixels of the views above), or with --guess facing the camera of
//       that view, and writes the covered scene; --style all writes <out>-<style>.ply each.
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

#include "gs/face_cover.h"
#include "gs/png_write.h"
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <scene> views <out_dir> | cover <landmarks.json> <out.ply> [options]\n", argv[0]);
    return 2;
  }
  try {
    const std::vector<uint8_t> bytes = read_all(argv[1]);
    const gs::SplatScene scene = gs::load_splat_file(bytes.data(), bytes.size());
    const float up[3] = {0.0f, scene.convention == gs::AxisConvention::RDF ? -1.0f : 1.0f, 0.0f};
    const int count = std::atoi(option(argc, argv, "--count", "12"));
    const std::string mode = argv[2];
    const uint32_t view_size = uint32_t(std::atoi(option(argc, argv, mode == "views" ? "--size" : "--view-size", "384")));
    auto t0 = std::chrono::steady_clock::now();
    const gs::FaceViews views = gs::render_face_views(scene, up, count, view_size);
    std::printf("views: %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    if (!views.head.found) throw std::runtime_error("no head found");
    std::printf("head at (%.3f %.3f %.3f), radius %.3f, subject height %.2f, %zu splats near it\n",
                views.head.center[0], views.head.center[1], views.head.center[2], views.head.radius,
                views.head.subject_height, views.subset.size());

    if (mode == "views") {
      const std::string dir = argv[3];
      std::string json = "[";
      for (size_t i = 0; i < views.views.size(); ++i) {
        const gs::OrthoView& v = views.views[i];
        char name[64];
        std::snprintf(name, sizeof(name), "/view_%02zu.jpg", i);
        gs::write_file(dir + name, gs::encode_jpeg_rgb(v.rgb8().data(), v.width, v.height, 95));
        json += (i ? "," : "") + std::string("{\"view\":") + std::to_string(i) +
                ",\"azimuth_deg\":" + std::to_string(views.azimuth[i] * 180.0f / 3.14159265f) + "}";
      }
      json += "]\n";
      gs::write_file(dir + "/views.json", std::vector<uint8_t>(json.begin(), json.end()));
      std::printf("wrote %zu views to %s\n", views.views.size(), dir.c_str());
      return 0;
    }
    if (mode != "cover" || argc < 5) throw std::runtime_error("unknown mode " + mode);

    gs::FacePlacement face;
    const char* guess = option(argc, argv, "--guess", nullptr);
    if (guess != nullptr) {
      const gs::OrthoView& v = views.views.at(size_t(std::atoi(guess)));
      const float facing[3] = {-v.forward[0], -v.forward[1], -v.forward[2]};
      face = gs::guess_face(scene, views, facing);
    } else {
      const std::vector<uint8_t> text = read_all(argv[3]);
      const auto faces = gs::parse_face_landmarks(std::string_view(reinterpret_cast<const char*>(text.data()), text.size()));
      face = gs::place_face(scene, views, faces);
    }
    if (!face.found) throw std::runtime_error("face not placed");
    std::printf("face %s from %d view(s) (%d agree): centre (%.3f %.3f %.3f) forward (%.2f %.2f %.2f) up (%.2f %.2f %.2f), eyes %.1f mm apart\n",
                face.detected ? "detected" : "guessed", face.views_used, face.views_agreeing, face.center[0], face.center[1], face.center[2],
                face.forward[0], face.forward[1], face.forward[2], face.up[0], face.up[1], face.up[2],
                face.eye_distance * 1000.0f);

    gs::FaceCoverOptions options;
    options.size = float(std::atof(option(argc, argv, "--size", "1")));
    options.raise = float(std::atof(option(argc, argv, "--raise", "0")));
    const std::string style = option(argc, argv, "--style", "mosaic");
    std::vector<std::string> styles;
    if (style == "all") {
      styles = {"mosaic", "blur", "eyes", "sunglasses", "mask", "smile"};
    } else {
      styles = {style};
    }
    for (const std::string& name : styles) {
      if (!gs::parse_face_style(name, &options.style)) throw std::runtime_error("unknown style " + name);
      gs::FaceCoverStats stats;
      t0 = std::chrono::steady_clock::now();
      const gs::SplatScene covered = gs::cover_face(scene, face, options, &stats);
      std::printf("cover: %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      std::string out = argv[4];
      if (styles.size() > 1) out = out.substr(0, out.size() - 4) + "-" + name + ".ply";
      gs::write_file(out, gs::save_ply(covered));
      std::printf("%s: added %u splats, recoloured %u -> %s\n", name.c_str(), stats.added, stats.recoloured, out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
