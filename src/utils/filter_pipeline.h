#pragma once
#include "../input_args_parser/input_args_parser.h"
#include "filter_utils.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace cuda_filter {

struct FilterStage {
  FilterType type;
  FilterOptions options;

  cv::Mat h_kernel;

  float *d_kernel = nullptr;
  int kernelSize = 0;

  float *d_x = nullptr;
  float *d_y = nullptr;
  float *d_Y = nullptr;
  float *d_Y_blurred = nullptr;

  FilterStage(FilterType t, const FilterOptions &opt, const cv::Mat &k) : type(t), options(opt), h_kernel(k) { kernelSize = k.rows; }
};

class FilterPipeline {
private:
  std::vector<FilterStage> stages;

public:
  std::vector<cudaStream_t> streams;
  int numStreams = 4;

  FilterPipeline() = default;

  ~FilterPipeline() { clear(); }

  void initStreams() {
    streams.resize(numStreams);
    for (int i = 0; i < numStreams; ++i) {
      cudaStreamCreate(&streams[i]);
    }
  }

  void destroyStreams() {
    for (auto &stream : streams) {
      cudaStreamDestroy(stream);
    }
    streams.clear();
  }

  void addFilter(FilterType type, const FilterOptions &options, const cv::Mat &kernel) { stages.emplace_back(type, options, kernel); }

  size_t size() const { return stages.size(); }
  FilterStage &getStage(size_t index) { return stages[index]; }
  const FilterStage &getStage(size_t index) const { return stages[index]; }

  void clear() {
    for (auto &stage : stages) {
      if (stage.d_kernel != nullptr) {
        cudaFree(stage.d_kernel);
        stage.d_kernel = nullptr;
      }
      if (stage.d_x != nullptr) {
        cudaFree(stage.d_x);
        stage.d_x = nullptr;
      }
      if (stage.d_y != nullptr) {
        cudaFree(stage.d_y);
        stage.d_y = nullptr;
      }
      if (stage.d_Y != nullptr) {
        cudaFree(stage.d_Y);
        stage.d_Y = nullptr;
      }
      if (stage.d_Y_blurred != nullptr) {
        cudaFree(stage.d_Y_blurred);
        stage.d_Y_blurred = nullptr;
      }
    }
    stages.clear();
    destroyStreams();
  }
};

} // namespace cuda_filter
