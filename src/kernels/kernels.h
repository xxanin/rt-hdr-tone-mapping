#pragma once
#include "../input_args_parser/input_args_parser.h"
#include "filter_pipeline.h"
#include <opencv2/opencv.hpp>

namespace cuda_filter {

void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel, const FilterOptions &options);
void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel, const FilterOptions &options);

void initPipelineGPU(FilterPipeline &pipeline, int width, int height, int channels, unsigned char **d_bufA, unsigned char **d_bufB);
void applyPipelineGPU(const cv::Mat &input, cv::Mat &output, FilterPipeline &pipeline, unsigned char *d_bufA, unsigned char *d_bufB);
void cleanupPipelineGPU(unsigned char *d_bufA, unsigned char *d_bufB);

void applyWipeTransitionGPU(const unsigned char *d_imgA, const unsigned char *d_imgB, unsigned char *d_output, int width, int height, int channels, int wipe_x);

namespace cuda {
// CUDA-specific type declarations and helper functions
#ifdef __CUDACC__
// These will only be visible to CUDA compiler
__host__ __device__ inline int divUp(int a, int b) { return (a + b - 1) / b; }
#endif
} // namespace cuda

} // namespace cuda_filter
