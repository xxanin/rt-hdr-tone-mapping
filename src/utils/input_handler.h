#pragma once

#include "../input_args_parser/input_args_parser.h"
#include <opencv2/opencv.hpp>

namespace cuda_filter {

class InputHandler {
public:
  InputHandler(const FilterOptions &options);
  ~InputHandler();

  bool isOpened() const;
  bool readFrame(cv::Mat &frame);
  void displayFrame(const cv::Mat &frame);
  void displaySideBySide(const cv::Mat &frame1, const cv::Mat &frame2);

private:
  FilterOptions m_options;
  cv::VideoCapture m_cap;
  cv::Mat m_staticFrame;
  int m_frameCounter;
  bool m_isStaticImage;

  void createSyntheticFrame(cv::Mat &frame);
  void createCheckerboard(cv::Mat &frame);
  void createGradient(cv::Mat &frame);
  void createNoise(cv::Mat &frame);
};

} // namespace cuda_filter
