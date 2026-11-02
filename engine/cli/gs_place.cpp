// Converts a splat file (.ply / .spz) into a place for the viewer (.gsenv):
// RDF axes, colour only (places are turned around the model, so view-
// dependent colour is dropped).
//
//   gs_place <in.ply|in.spz> <out.gsenv>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <vector>

#include "gs/environment.h"
#include "gs/png_write.h"
#include "gs/splat_scene.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <in.ply|in.spz> <out.gsenv>\n", argv[0]);
    return 2;
  }
  try {
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) throw std::runtime_error(std::string("cannot open ") + argv[1]);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    gs::SplatScene scene = gs::load_splat_file(bytes.data(), bytes.size());
    gs::convert_convention(scene, gs::AxisConvention::RDF);
    gs::SplatScene place = scene;
    place.resize(scene.count, 0);
    place.convention = gs::AxisConvention::RDF;
    place.antialiased = scene.antialiased;
    place.positions = scene.positions;
    place.log_scales = scene.log_scales;
    place.rotations = scene.rotations;
    place.opacity_logits = scene.opacity_logits;
    place.sh_dc = scene.sh_dc;
    const std::vector<uint8_t> out = gs::write_gsenv(place);
    gs::write_file(argv[2], out);
    std::printf("%u splats -> %s (%.1f MB)\n", place.count, argv[2], out.size() / 1e6);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
