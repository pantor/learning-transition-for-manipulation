#pragma once

#include <iostream>
#include <limits>
#include <thread>

#include <opencv2/opencv.hpp>
#include <nxLib.h>

#include <ensenso/config.hpp>


class Ensenso {
  double min_depth; // [m]
  double max_depth; // [m]
  double factor_depth;

  NxLibItem root; // Reference to the API tree root
  NxLibItem camera; // Reference to the nxLib camera

  template<typename T, typename U>
  T clampLimits(U value) {
    return std::min<U>(std::max<U>(value, std::numeric_limits<T>::min()), std::numeric_limits<T>::max());
  }

  void configureCapture(const EnsensoConfig config);


public:
  Ensenso(EnsensoConfig config);
  ~Ensenso();

  EnsensoConfig raw_capture_config;
  EnsensoConfig depth_capture_config;

  void configureRawCaptureParams(EnsensoConfig config);
  void configureDepthCaptureParams(EnsensoConfig config);

  cv::Mat takeDepthImage();
  std::pair<cv::Mat, cv::Mat> takeImages();
};
