#pragma once

/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * Full license terms provided in LICENSE.md file.
 */

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <pybind11/numpy.h>


using namespace nvinfer1;


namespace TensorRT {

struct NetConfig
{
  std::string planPath;
  std::string inputNodeName;
  std::string outputNodeName;
  int inputHeight;
  int inputWidth;
  int inputChannels;
  int numOutputCategories;
  int maxBatchSize;
  NetConfig(std::string, std::string, std::string, int, int, int, int, int);
};

class TRTEngine
{
  int inputBindingIndex;
  int outputBindingIndex;
  int numOutputCategories;
  size_t inputSize;
  float * inputDevice;
  float * outputDevice;
  float * bindings[2];
  float * output;

  IRuntime* runtime;
  ICudaEngine* engine;
  IExecutionContext* context;

public:
  TRTEngine(const NetConfig&);
  ~TRTEngine();

  std::vector<float> execute(pybind11::array_t<float, pybind11::array::c_style>);
};

float * imageToTensor(pybind11::array_t<float, pybind11::array::c_style>);
} // namespace TensorRT
