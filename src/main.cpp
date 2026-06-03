#include "input_args_parser/input_args_parser.h"
#include "kernels/kernels.h"
#include "utils/filter_utils.h"
#include <opencv2/opencv.hpp>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

int main(int argc, char **argv) {
  plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::info, &consoleAppender);

  cuda_filter::InputArgsParser parser(argc, argv);
  cuda_filter::FilterOptions options = parser.parseArgs();

  cuda_filter::FilterType filterType = cuda_filter::FilterUtils::stringToFilterType(options.filterType);
  cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(filterType, options.kernelSize, options.intensity);

  PLOG_INFO << "Filter: " << options.filterType << " | Tone Algo: " << options.toneMappingAlgo;

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
    PLOG_ERROR << "Failed to initialize VideoWriter. Codec might not be supported on this server.";
    return -1;
  }

  cv::Mat frame, filteredCPU, filteredGPU;
  int frameIndex = 1;

  PLOG_INFO << "Starting video streaming processing...";

  while (true) {
    cap >> frame;
    if (frame.empty()) {
      PLOG_INFO << "Finished processing video!";
      break;
    }

    const double cpuStart = static_cast<double>(cv::getTickCount());
    cuda_filter::applyFilterCPU(frame, filteredCPU, kernel, options);
    const double cpuEnd = static_cast<double>(cv::getTickCount());
    const double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency();
    double fpsCPU = 1.0 / cpuTime;

    const double gpuStart = static_cast<double>(cv::getTickCount());
    cuda_filter::applyFilterGPU(frame, filteredGPU, kernel, options);
    const double gpuEnd = static_cast<double>(cv::getTickCount());
    const double gpuTime = (gpuEnd - gpuStart) / cv::getTickFrequency();
    double fpsGPU = 1.0 / gpuTime;

    if (frameIndex % 30 == 0) {
      PLOG_INFO << "Processed frame " << frameIndex << " | GPU FPS: " << fpsGPU;
    }

    std::string cpuText = "CPU FPS: " + std::to_string(static_cast<int>(fpsCPU)) + " Time: " + std::to_string(cpuTime * 1000).substr(0, 4) + "ms";
    std::string gpuText = "GPU FPS: " + std::to_string(static_cast<int>(fpsGPU)) + " Time: " + std::to_string(gpuTime * 1000).substr(0, 4) + "ms";
    cv::putText(filteredCPU, cpuText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(filteredGPU, gpuText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

    cv::Mat combined;
    cv::hconcat(filteredCPU, filteredGPU, combined);
    cv::putText(combined, "CPU Version", cv::Point(10, combined.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(combined, "GPU Version", cv::Point(combined.cols / 2 + 10, combined.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

    writer.write(combined);
    frameIndex++;
  }

  cap.release();
  writer.release();
  PLOG_INFO << "Application terminated successfully. Video saved as final_output.mp4";
  return 0;
}
