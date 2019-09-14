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


using namespace std;
using namespace nvinfer1;


namespace TensorRT {

typedef void (*preprocess_fn_t)(float *input, size_t channels, size_t height, size_t width);

struct NetConfig
{
  string planPath;
  string inputNodeName;
  string outputNodeName;
  string preprocessFnName;
  int inputHeight;
  int inputWidth;
  int numOutputCategories;
  int maxBatchSize;
  
  NetConfig(string, string, string, string, int, int, int, int);

  string toString();
  preprocess_fn_t preprocessFn() const;
};

class InferenceEngine
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

  float* prepareInput(string);

public:
  InferenceEngine(const NetConfig&);
  ~InferenceEngine();

  std::vector<float> execute(string);
};

float * imageToTensor(const cv::Mat &);
void preprocessVgg(float*, size_t, size_t, size_t);
void preprocessInception(float*, size_t, size_t, size_t);
int execute(const NetConfig &, string);
} // namespace TensorRT
