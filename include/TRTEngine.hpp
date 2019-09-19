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


using namespace nvinfer1;


namespace TensorRT {

typedef void (*preprocess_fn_t)(float *input, size_t channels, size_t height, size_t width);

struct NetConfig
{
  std::string planPath;
  std::string inputNodeName;
  std::string outputNodeName;
  std::string preprocessFnName;
  int inputHeight;
  int inputWidth;
  int numOutputCategories;
  int maxBatchSize;
  
  NetConfig(std::string, std::string, std::string, std::string, int, int, int, int);

  preprocess_fn_t preprocessFn() const;
};

class TRTEngine
{
  int inputBindingIndex;
  int outputBindingIndex;
  int inputWidth;
  int inputHeight;
  int numOutputCategories;
  size_t inputSize;

  preprocess_fn_t preprocessFn;
  IRuntime* runtime;
  ICudaEngine* engine;
  IExecutionContext* context;

public:
  TRTEngine(const NetConfig&);
  ~TRTEngine();

  std::vector<float> execute(std::string);
  double measureThroughput(std::string, int);
};

float * imageToTensor(const cv::Mat &);
void preprocessVgg(float*, size_t, size_t, size_t);
void preprocessInception(float*, size_t, size_t, size_t);
} // namespace TensorRT
