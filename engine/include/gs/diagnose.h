#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gs/trainer.h"

namespace gs {

// Where a trained model still disagrees with its photos, to tell what limits
// quality on a real capture. Every view is rendered as its camera saw it
// (pose correction, the subject's motion then) over black and compared with
// its photo times the mask, divided by the view's exposure gains when the
// trainer refines them (so what remains is what they did not explain).
struct Diagnosis {
  uint32_t views = 0;
  double psnr = 0.0;  // whole images, mean over the views

  // Brightness and colour per view: the least-squares gain photo = g x render
  // over the subject's inside (mask and render both opaque, away from the
  // mask's edge). Gains that vary between views mean the camera's exposure or
  // white balance changed while walking round; a model can only average them.
  uint32_t gain_views = 0;  // views with enough inside pixels
  double gain_low = 1.0, gain_high = 1.0;  // 5th and 95th percentile of the brightness gain
  double gain_spread = 0.0;                // standard deviation of the brightness gain
  double tint_spread = 0.0;  // standard deviation of the red/green and blue/green gain ratios (their mean)
  double inside_psnr = 0.0;         // over those pixels of all views
  double inside_psnr_fitted = 0.0;  // the same with each view's best gain and offset per channel

  // Where the squared error is, as shares of the total (they sum to 1).
  double share_edge = 0.0;     // within 3 px of the mask's edge
  double share_outline = 0.0;  // elsewhere, where render and mask disagree about subject or not
  double share_inside = 0.0;   // inside the subject, the head included
  double share_head = 0.0;     // the part of share_inside in the head circle
  double share_outside = 0.0;  // background where both are empty
  double edge_pixels = 0.0;    // fraction of all pixels in the edge band

  // Sharpness of the photos: variance of the Laplacian of brightness inside
  // the subject, relative to the median view.
  double blurry_fraction = 0.0;  // views under half the median

  struct View {
    size_t frame = 0;
    bool holdout = false;
    double psnr = 0.0;
    double gain = 1.0;       // brightness gain (1 without enough inside pixels)
    double sharpness = 1.0;  // relative to the median view
  };
  std::vector<View> worst;  // the lowest PSNR first, up to 3
};

// Renders every view once (about as long as an evaluation without alignment).
Diagnosis diagnose(Trainer& trainer);

}  // namespace gs
