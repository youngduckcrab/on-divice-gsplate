// A finished capture's dataset, kept in memory so training can start without
// reading the files back (capture_jni.cpp hands it over, the trainer's
// prepare() takes it).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "gs/dataset.h"

struct CapturedDataset {
  std::string name;     // the dataset folder's name
  uint32_t short_side;  // training resolution it was made at
  gs::Dataset dataset;
};

// Replaces any capture waiting before.
void hand_over_capture(std::unique_ptr<CapturedDataset> capture);
// The waiting capture if it is `name` at `short_side`, else null (it stays).
std::unique_ptr<CapturedDataset> take_capture(const std::string& name, uint32_t short_side);
void discard_capture();
