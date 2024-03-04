#pragma once
#include <chrono>

auto time_start() { return std::chrono::high_resolution_clock::now(); }
auto time_stop(std::chrono::high_resolution_clock::time_point start) {
  auto stop = std::chrono::high_resolution_clock::now() - start;
  return std::chrono::duration_cast<std::chrono::microseconds>(stop).count();
}
