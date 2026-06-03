#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <plog/Log.h>

namespace cuda_filter {

#define CHECK_CUDA_ERROR(call)                                                                                                                                                                         \
  {                                                                                                                                                                                                    \
    cudaError_t err = call;                                                                                                                                                                            \
    if (err != cudaSuccess) {                                                                                                                                                                          \
      PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err);                                                                                                                      \
      return;                                                                                                                                                                                          \
    }                                                                                                                                                                                                  \
  }

__global__ void convolutionKernel(const unsigned char *input, unsigned char *output, const float *kernel, int width, int height, int channels, int kernelSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;
  int radius = kernelSize / 2;

  for (int c = 0; c < channels; c++) {
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ky++) {
      for (int kx = -radius; kx <= radius; kx++) {
        int ix = min(max(x + kx, 0), width - 1);
        int iy = min(max(y + ky, 0), height - 1);
        float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
        float pixelValue = input[(iy * width + ix) * channels + c];
        sum += pixelValue * kernelValue;
      }
    }
    output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
  }
}

__global__ void kernel_bgr_to_xyY(const unsigned char *input, float *d_x, float *d_y, float *d_Y, int width, int height, float gamma) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx_pixel = (y * width + x) * 3;
    int idx_flat = y * width + x;

    float B = powf(input[idx_pixel] / 255.0f, gamma);
    float G = powf(input[idx_pixel + 1] / 255.0f, gamma);
    float R = powf(input[idx_pixel + 2] / 255.0f, gamma);

    float X = 0.4124f * R + 0.3576f * G + 0.1805f * B;
    float Y_luma = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    float Z = 0.0193f * R + 0.1192f * G + 0.9505f * B;

    float divisor = X + Y_luma + Z;
    divisor = (divisor == 0.0f) ? 1e-6f : divisor;

    d_Y[idx_flat] = Y_luma;
    d_x[idx_flat] = X / divisor;
    d_y[idx_flat] = Y_luma / divisor;
  }
}

__global__ void kernel_global_tone_map(float *d_Y, int width, int height, float exposure) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * width + x;
    float Y = d_Y[idx];
    d_Y[idx] = 1.0f - expf(-exposure * Y);
  }
}

#define BLUR_RADIUS 4
#define BLOCK_DIM 16
#define SHARED_DIM (BLOCK_DIM + 2 * BLUR_RADIUS)

__global__ void kernel_shared_memory_blur(const float *d_Y_in, float *d_Y_out, int width, int height) {
  __shared__ float s_Y[SHARED_DIM][SHARED_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;

  int x_left = max(0, x - BLUR_RADIUS);
  int x_right = min(width - 1, x + BLUR_RADIUS);
  int y_top = max(0, y - BLUR_RADIUS);
  int y_bottom = min(height - 1, y + BLUR_RADIUS);

  s_Y[ty + BLUR_RADIUS][tx + BLUR_RADIUS] = (x < width && y < height) ? d_Y_in[y * width + x] : 0.0f;

  if (tx < BLUR_RADIUS) {
    s_Y[ty + BLUR_RADIUS][tx] = d_Y_in[min(max(y, 0), height - 1) * width + x_left];
    s_Y[ty + BLUR_RADIUS][tx + BLOCK_DIM + BLUR_RADIUS] = d_Y_in[min(max(y, 0), height - 1) * width + x_right];
  }
  if (ty < BLUR_RADIUS) {
    s_Y[ty][tx + BLUR_RADIUS] = d_Y_in[y_top * width + min(max(x, 0), width - 1)];
    s_Y[ty + BLOCK_DIM + BLUR_RADIUS][tx + BLUR_RADIUS] = d_Y_in[y_bottom * width + min(max(x, 0), width - 1)];
  }
  if (tx < BLUR_RADIUS && ty < BLUR_RADIUS) {
    s_Y[ty][tx] = d_Y_in[y_top * width + x_left];
    s_Y[ty][tx + BLOCK_DIM + BLUR_RADIUS] = d_Y_in[y_top * width + x_right];
    s_Y[ty + BLOCK_DIM + BLUR_RADIUS][tx] = d_Y_in[y_bottom * width + x_left];
    s_Y[ty + BLOCK_DIM + BLUR_RADIUS][tx + BLOCK_DIM + BLUR_RADIUS] = d_Y_in[y_bottom * width + x_right];
  }

  __syncthreads();

  if (x < width && y < height) {
    float sum = 0.0f;
    int count = 0;

    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++) {
      for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++) {
        sum += s_Y[ty + BLUR_RADIUS + ky][tx + BLUR_RADIUS + kx];
        count++;
      }
    }
    d_Y_out[y * width + x] = sum / (float)count;
  }
}

__global__ void kernel_local_tone_map(float *d_Y, const float *d_Y_blurred, int width, int height, float exposure) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * width + x;
    float Y_orig = d_Y[idx];
    float Y_base = d_Y_blurred[idx];

    float detail = Y_orig - Y_base;
    float compressed_base = 1.0f - expf(-exposure * Y_base);
    float Y_new = compressed_base + detail;

    d_Y[idx] = fmaxf(0.0f, Y_new);
  }
}

__global__ void kernel_xyY_to_bgr(unsigned char *output, float *d_x, float *d_y, float *d_Y, int width, int height, float gamma, float saturation) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx_pixel = (y * width + x) * 3;
    int idx_flat = y * width + x;

    float x_chroma = d_x[idx_flat];
    float y_chroma = d_y[idx_flat];
    float Y_luma = d_Y[idx_flat];

    float x_white = 0.3127f;
    float y_white = 0.3290f;

    x_chroma = x_white + saturation * (x_chroma - x_white);
    y_chroma = y_white + saturation * (y_chroma - y_white);
    y_chroma = (y_chroma <= 0.0f) ? 1e-6f : y_chroma;

    float X = (Y_luma / y_chroma) * x_chroma;
    float Z = (Y_luma / y_chroma) * (1.0f - x_chroma - y_chroma);

    float R = 3.2406f * X - 1.5372f * Y_luma - 0.4986f * Z;
    float G = -0.9689f * X + 1.8758f * Y_luma + 0.0415f * Z;
    float B = 0.0557f * X - 0.2040f * Y_luma + 1.0570f * Z;

    float inv_gamma = 1.0f / gamma;
    R = powf(fmaxf(0.0f, R), inv_gamma);
    G = powf(fmaxf(0.0f, G), inv_gamma);
    B = powf(fmaxf(0.0f, B), inv_gamma);

    output[idx_pixel] = static_cast<unsigned char>(fminf(fmaxf(B * 255.0f, 0.0f), 255.0f));
    output[idx_pixel + 1] = static_cast<unsigned char>(fminf(fmaxf(G * 255.0f, 0.0f), 255.0f));
    output[idx_pixel + 2] = static_cast<unsigned char>(fminf(fmaxf(R * 255.0f, 0.0f), 255.0f));
  }
}

void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel, const FilterOptions &options) {
  if (input.empty())
    return;
  output.create(input.size(), input.type());

  int width = input.cols;
  int height = input.rows;
  int channels = input.channels();
  size_t imageSize = width * height * channels * sizeof(unsigned char);
  size_t flatSize = width * height * sizeof(float);

  unsigned char *d_input = nullptr, *d_output = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
  CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));

  dim3 blockDim(16, 16);
  dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

  if (options.filterType == "hdr") {
    float *d_x = nullptr, *d_y = nullptr, *d_Y = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, flatSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, flatSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Y, flatSize));

    kernel_bgr_to_xyY<<<gridDim, blockDim>>>(d_input, d_x, d_y, d_Y, width, height, options.gamma);
    CHECK_CUDA_ERROR(cudaGetLastError());

    if (options.toneMappingAlgo == 1) {
      float *d_Y_blurred = nullptr;
      CHECK_CUDA_ERROR(cudaMalloc(&d_Y_blurred, flatSize));

      kernel_shared_memory_blur<<<gridDim, blockDim>>>(d_Y, d_Y_blurred, width, height);
      CHECK_CUDA_ERROR(cudaGetLastError());

      kernel_local_tone_map<<<gridDim, blockDim>>>(d_Y, d_Y_blurred, width, height, options.exposure);
      CHECK_CUDA_ERROR(cudaGetLastError());

      cudaFree(d_Y_blurred);
    } else {
      kernel_global_tone_map<<<gridDim, blockDim>>>(d_Y, width, height, options.exposure);
      CHECK_CUDA_ERROR(cudaGetLastError());
    }

    kernel_xyY_to_bgr<<<gridDim, blockDim>>>(d_output, d_x, d_y, d_Y, width, height, options.gamma, options.saturation);
    CHECK_CUDA_ERROR(cudaGetLastError());

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_Y);
  } else {
    // RESTORED FALLBACK FOR NORMAL FILTERS (Blur, Edge Detection, etc.)
    float *d_kernel = nullptr;
    int kernelSize = kernel.rows;
    size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

    float *h_kernel = new float[kernelSize * kernelSize];
    for (int i = 0; i < kernelSize; i++) {
      for (int j = 0; j < kernelSize; j++) {
        h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
      }
    }

    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);
    CHECK_CUDA_ERROR(cudaGetLastError());

    cudaFree(d_kernel);
    delete[] h_kernel;
  }

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));
  cudaFree(d_input);
  cudaFree(d_output);
}

void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel, const FilterOptions &options) {
  if (input.empty())
    return;
  output.create(input.size(), input.type());

  if (options.filterType == "hdr") {
    int width = input.cols;
    int height = input.rows;
    float inv_gamma = 1.0f / options.gamma;
    float x_white = 0.3127f;
    float y_white = 0.3290f;

    // Temporary matrices to hold the channels
    cv::Mat Y_mat(height, width, CV_32F);
    cv::Mat x_mat(height, width, CV_32F);
    cv::Mat y_mat(height, width, CV_32F);

    // Pass 1: BGR to xyY
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
        float B = std::pow(pixel[0] / 255.0f, options.gamma);
        float G = std::pow(pixel[1] / 255.0f, options.gamma);
        float R = std::pow(pixel[2] / 255.0f, options.gamma);

        float X = 0.4124f * R + 0.3576f * G + 0.1805f * B;
        float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;
        float Z = 0.0193f * R + 0.1192f * G + 0.9505f * B;

        float divisor = X + Y + Z;
        divisor = (divisor == 0.0f) ? 1e-6f : divisor;

        Y_mat.at<float>(y, x) = Y;
        x_mat.at<float>(y, x) = X / divisor;
        y_mat.at<float>(y, x) = Y / divisor;
      }
    }

    // Optional Base Layer Blur for Local Tone Mapping
    cv::Mat Y_base;
    if (options.toneMappingAlgo == 1) {
      // A 9x9 blur roughly matches our BLUR_RADIUS 4 on the GPU
      cv::blur(Y_mat, Y_base, cv::Size(9, 9));
    }

    // Pass 2: Tone Map and Convert Back
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float Y_orig = Y_mat.at<float>(y, x);
        float x_chroma = x_mat.at<float>(y, x);
        float y_chroma = y_mat.at<float>(y, x);

        // Saturation
        x_chroma = x_white + options.saturation * (x_chroma - x_white);
        y_chroma = y_white + options.saturation * (y_chroma - y_white);
        y_chroma = (y_chroma <= 0.0f) ? 1e-6f : y_chroma;

        float Y_mapped;
        if (options.toneMappingAlgo == 1) {
          // Local Tone Mapping Math
          float Y_blurred = Y_base.at<float>(y, x);
          float detail = Y_orig - Y_blurred;
          float compressed = 1.0f - std::exp(-options.exposure * Y_blurred);
          Y_mapped = std::max(0.0f, compressed + detail);
        } else {
          // Global Tone Mapping Math
          Y_mapped = 1.0f - std::exp(-options.exposure * Y_orig);
        }

        // xyY to RGB
        float X_new = (Y_mapped / y_chroma) * x_chroma;
        float Z_new = (Y_mapped / y_chroma) * (1.0f - x_chroma - y_chroma);

        float R_new = 3.2406f * X_new - 1.5372f * Y_mapped - 0.4986f * Z_new;
        float G_new = -0.9689f * X_new + 1.8758f * Y_mapped + 0.0415f * Z_new;
        float B_new = 0.0557f * X_new - 0.2040f * Y_mapped + 1.0570f * Z_new;

        output.at<cv::Vec3b>(y, x) =
            cv::Vec3b(cv::saturate_cast<uchar>(std::pow(std::max(0.0f, B_new), inv_gamma) * 255.0f), cv::saturate_cast<uchar>(std::pow(std::max(0.0f, G_new), inv_gamma) * 255.0f),
                      cv::saturate_cast<uchar>(std::pow(std::max(0.0f, R_new), inv_gamma) * 255.0f));
      }
    }
  } else {
    cv::filter2D(input, output, -1, kernel);
  }
}
} // namespace cuda_filter
