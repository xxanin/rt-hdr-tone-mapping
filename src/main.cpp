#include "filter_pipeline.h"
#include "input_args_parser/input_args_parser.h"
#include "kernels/kernels.h"
#include "utils/filter_utils.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <deque>
#include <opencv2/opencv.hpp>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

void drawPerformanceGraph(cv::Mat &img, const std::deque<float> &history, int x_offset, int y_offset, int width, int height) {
  cv::rectangle(img, cv::Point(x_offset, y_offset), cv::Point(x_offset + width, y_offset + height), cv::Scalar(40, 40, 40), -1);
  cv::rectangle(img, cv::Point(x_offset, y_offset), cv::Point(x_offset + width, y_offset + height), cv::Scalar(150, 150, 150), 1);

  if (history.empty())
    return;

  float max_val = *std::max_element(history.begin(), history.end());
  if (max_val < 1.0f)
    max_val = 1.0f;

  float step_x = static_cast<float>(width) / 30.0f;

  for (size_t i = 0; i < history.size(); ++i) {
    int bar_height = static_cast<int>((history[i] / max_val) * (height - 25));
    int pt_x = x_offset + static_cast<int>(i * step_x);
    int pt_y = y_offset + height - bar_height - 5;

    cv::Scalar bar_color = (history[i] > 33.3f) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::rectangle(img, cv::Point(pt_x, pt_y), cv::Point(pt_x + static_cast<int>(step_x) - 2, y_offset + height - 5), bar_color, -1);
  }

  std::string labelText = "GPU Hist (Max: " + std::to_string(max_val).substr(0, 4) + "ms)";
  cv::putText(img, labelText, cv::Point(x_offset + 6, y_offset + 16), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
}

int main(int argc, char **argv) {
  plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::info, &consoleAppender);

  cuda_filter::InputArgsParser parser(argc, argv);
  cuda_filter::FilterOptions options = parser.parseArgs();

  cuda_filter::FilterType filterType = cuda_filter::FilterUtils::stringToFilterType(options.filterType);
  cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(filterType, options.kernelSize, options.intensity);

  PLOG_INFO << "Filter: " << options.filterType << " | Tone Algo: " << options.toneMappingAlgo << " | Streams: " << options.numStreams;

  if (options.targetFilter != "none") {
    PLOG_INFO << "Transition requested to: " << options.targetFilter << " over " << options.transitionDurationMs << "ms.";
  }

  std::string videoFile = options.inputPath;
  if (videoFile == "test_image.jpg")
    videoFile = "test.mp4";

  cv::VideoCapture cap(videoFile);
  if (!cap.isOpened()) {
    PLOG_ERROR << "Failed to open video file: " << videoFile;
    return -1;
  }

  int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0)
    fps = 30.0;

  cv::Size outSize(width * 2, height);
  cv::VideoWriter writer("final_output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, outSize);

  if (!writer.isOpened()) {
    PLOG_ERROR << "Failed to initialize VideoWriter.";
    return -1;
  }

  cv::Mat frame, filteredCPU, filteredGPU;
  cap >> frame;
  if (frame.empty())
    return -1;

  int channels = frame.channels();
  size_t imageSize = width * height * channels * sizeof(unsigned char);

  cuda_filter::FilterPipeline pipeline;
  pipeline.numStreams = options.numStreams;
  pipeline.initStreams();
  pipeline.addFilter(filterType, options, kernel);
  unsigned char *d_bufA = nullptr, *d_bufB = nullptr;
  cuda_filter::initPipelineGPU(pipeline, width, height, channels, &d_bufA, &d_bufB);

  bool inTransition = (options.targetFilter != "none");
  double transitionDurationSec = options.transitionDurationMs / 1000.0;

  cuda_filter::FilterPipeline targetPipeline;
  targetPipeline.numStreams = options.numStreams;
  unsigned char *d_bufA_target = nullptr, *d_bufB_target = nullptr;
  unsigned char *d_wipe_out = nullptr;

  if (inTransition) {
    cuda_filter::FilterType targetType = cuda_filter::FilterUtils::stringToFilterType(options.targetFilter);
    cv::Mat targetKernel = cuda_filter::FilterUtils::createFilterKernel(targetType, options.kernelSize, options.intensity);

    targetPipeline.initStreams();
    targetPipeline.addFilter(targetType, options, targetKernel);
    cuda_filter::initPipelineGPU(targetPipeline, width, height, channels, &d_bufA_target, &d_bufB_target);

    if (cudaMalloc(&d_wipe_out, imageSize) != cudaSuccess) {
      PLOG_ERROR << "Failed to allocate d_wipe_out";
      return -1;
    }
  }

  cudaEvent_t gpuStartEvent, gpuStopEvent;
  cudaEventCreate(&gpuStartEvent);
  cudaEventCreate(&gpuStopEvent);
  std::deque<float> gpuTimeHistory;

  int frameIndex = 1;
  PLOG_INFO << "Starting video streaming processing...";

  double transitionStartTime = static_cast<double>(cv::getTickCount());

  do {
    const double cpuStart = static_cast<double>(cv::getTickCount());
    cuda_filter::applyFilterCPU(frame, filteredCPU, kernel, options);
    const double cpuEnd = static_cast<double>(cv::getTickCount());
    const double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency();
    double fpsCPU = 1.0 / cpuTime;

    cudaEventRecord(gpuStartEvent, 0);

    if (inTransition) {
      double elapsedSec = 0.0;
      if (options.inputSource == cuda_filter::InputSource::VIDEO) {
        elapsedSec = static_cast<double>(frameIndex) / fps; 
      } else {
        double currentTime = static_cast<double>(cv::getTickCount());
        elapsedSec = (currentTime - transitionStartTime) / cv::getTickFrequency(); 
      }

      if (elapsedSec >= transitionDurationSec) {
        inTransition = false;

        cuda_filter::cleanupPipelineGPU(d_bufA, d_bufB);

        pipeline = std::move(targetPipeline);
        d_bufA = d_bufA_target;
        d_bufB = d_bufB_target;

        if (d_wipe_out)
          cudaFree(d_wipe_out);
        d_wipe_out = nullptr;

        PLOG_INFO << "Transition Complete!";

        cuda_filter::applyPipelineGPU(frame, filteredGPU, pipeline, d_bufA, d_bufB);
      } else {
        cv::Mat dummyB;

        cuda_filter::applyPipelineGPU(frame, filteredGPU, pipeline, d_bufA, d_bufB);
        cuda_filter::applyPipelineGPU(frame, dummyB, targetPipeline, d_bufA_target, d_bufB_target);

        int wipe_x = static_cast<int>((elapsedSec / transitionDurationSec) * width);

        unsigned char *final_outA = (pipeline.size() % 2 == 0) ? d_bufA : d_bufB;
        unsigned char *final_outB = (targetPipeline.size() % 2 == 0) ? d_bufA_target : d_bufB_target;

        cuda_filter::applyWipeTransitionGPU(final_outA, final_outB, d_wipe_out, width, height, channels, wipe_x);

        cudaMemcpy(filteredGPU.data, d_wipe_out, imageSize, cudaMemcpyDeviceToHost);
      }
    } else {
      cuda_filter::applyPipelineGPU(frame, filteredGPU, pipeline, d_bufA, d_bufB);
    }

    cudaEventRecord(gpuStopEvent, 0);
    cudaEventSynchronize(gpuStopEvent);

    float gpuTimeMs = 0.0f;
    cudaEventElapsedTime(&gpuTimeMs, gpuStartEvent, gpuStopEvent);
    double fpsGPU = 1000.0 / gpuTimeMs;

    gpuTimeHistory.push_back(gpuTimeMs);
    if (gpuTimeHistory.size() > 30) {
      gpuTimeHistory.pop_front();
    }

    if (frameIndex % 30 == 0) {
      PLOG_INFO << "Processed frame " << frameIndex << " | GPU Hardware Latency: " << gpuTimeMs << " ms";
    }

    std::string cpuText = "CPU FPS: " + std::to_string(static_cast<int>(fpsCPU)) + " Time: " + std::to_string(cpuTime * 1000).substr(0, 4) + "ms";
    std::string gpuText = "GPU FPS: " + std::to_string(static_cast<int>(fpsGPU)) + " Time: " + std::to_string(gpuTimeMs).substr(0, 4) + "ms";
    cv::putText(filteredCPU, cpuText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(filteredGPU, gpuText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

    cv::Mat combined;
    cv::hconcat(filteredCPU, filteredGPU, combined);
    cv::putText(combined, "CPU Version", cv::Point(10, combined.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(combined, "GPU Version", cv::Point(combined.cols / 2 + 10, combined.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

    int graphW = 180;
    int graphH = 90;
    drawPerformanceGraph(combined, gpuTimeHistory, combined.cols - graphW - 20, combined.rows - graphH - 20, graphW, graphH);

    writer.write(combined);
    frameIndex++;

    cap >> frame;
  } while (!frame.empty());

  cuda_filter::cleanupPipelineGPU(d_bufA, d_bufB);
  if (d_wipe_out)
    cudaFree(d_wipe_out);

  cudaEventDestroy(gpuStartEvent);
  cudaEventDestroy(gpuStopEvent);

  cap.release();
  writer.release();
  PLOG_INFO << "Application terminated successfully.";
  return 0;
}
