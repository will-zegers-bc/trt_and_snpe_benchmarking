/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * Full license terms provided in LICENSE.md file.
 */
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "trt_inference_engine.hpp"


using namespace nvinfer1;


namespace TensorRT
{

typedef void (*preprocess_fn_t)(float *input, size_t channels, size_t height, size_t width);

class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
      std::cout << msg << std::endl;
  }
} gLogger;

NetConfig::NetConfig(std::string planPath,
                     std::string inputNodeName,
                     std::string outputNodeName,
                     std::string preprocessFnName,
                     int inputHeight,
                     int inputWidth,
                     int numOutputCategories,
                     int maxBatchSize) 
  : planPath(planPath)
  , inputNodeName(inputNodeName)
  , outputNodeName(outputNodeName)
  , preprocessFnName(preprocessFnName)
  , inputHeight(inputHeight)
  , inputWidth(inputWidth)
  , numOutputCategories(numOutputCategories)
  , maxBatchSize(maxBatchSize)
{
}

preprocess_fn_t NetConfig::preprocessFn() const {
  if (preprocessFnName == "preprocess_vgg")
     return preprocessVgg;
  else if (preprocessFnName == "preprocess_inception")
     return preprocessInception;
  else
     throw std::runtime_error("Invalid preprocessing function name.");
}

float *imageToTensor(const cv::Mat & image)
{
  const size_t height = image.rows;
  const size_t width = image.cols;
  const size_t channels = image.channels();
  const size_t numel = height * width * channels;

  const size_t stridesCv[3] = { width * channels, channels, 1 };
  const size_t strides[3] = { height * width, width, 1 };

  float * tensor;
  cudaHostAlloc((void**)&tensor, numel * sizeof(float), cudaHostAllocMapped);

  for (size_t i = 0; i < height; i++) 
  {
    for (size_t j = 0; j < width; j++) 
    {
      for (size_t k = 0; k < channels; k++) 
      {
        const size_t offsetCv = i * stridesCv[0] + j * stridesCv[1] + k * stridesCv[2];
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        tensor[offset] = (float) image.data[offsetCv];
      }
    }
  }

  return tensor;
}


void preprocessVgg(float * tensor, size_t channels, size_t height, size_t width)
{
  const size_t strides[3] = { height * width, width, 1 };
  const float mean[3] = { 123.68, 116.78, 103.94 };

  for (size_t i = 0; i < height; i++) 
  {
    for (size_t j = 0; j < width; j++) 
    {
      for (size_t k = 0; k < channels; k++) 
      {
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        tensor[offset] -= mean[k];
      }
    }
  }
}


void preprocessInception(float * tensor, size_t channels, size_t height, size_t width)
{
  const size_t numel = channels * height * width;
  for (size_t i = 0; i < numel; i++)
    tensor[i] = 2.0 * (tensor[i] / 255.0 - 0.5);
}

InferenceEngine::InferenceEngine(const NetConfig &netConfig) 
  : inputWidth(netConfig.inputWidth)
  , inputHeight(netConfig.inputHeight)
  , numOutputCategories(netConfig.numOutputCategories)
{
  std::ifstream planFile(netConfig.planPath);
  std::stringstream planBuffer;
  planBuffer << planFile.rdbuf();
  std::string plan = planBuffer.str();
  runtime = createInferRuntime(gLogger);
  engine = runtime->deserializeCudaEngine((void*)plan.data(),
      plan.size(), nullptr);

  context = engine->createExecutionContext();
  inputBindingIndex = engine->getBindingIndex(netConfig.inputNodeName.c_str());
  outputBindingIndex = engine->getBindingIndex(netConfig.outputNodeName.c_str());

  inputSize = inputHeight * inputWidth * 3 * sizeof(float);

  preprocessFn = netConfig.preprocessFn();
}

InferenceEngine::~InferenceEngine()
{
  context->destroy();  
  engine->destroy();
  runtime->destroy();
}

double InferenceEngine::measureThroughput(std::string imagePath, int numRuns)
{
  cv::Mat image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
  cv::resize(image, image, cv::Size(inputWidth, inputHeight));
  float *input = imageToTensor(image);
  preprocessFn(input, 3, inputHeight, inputWidth);

  float *inputDevice, *outputDevice, *output;

  // allocate memory on host / device for input / output
  cudaHostAlloc(&output, numOutputCategories * sizeof(float), cudaHostAllocMapped);
  cudaMalloc(&inputDevice, inputSize);
  cudaMalloc(&outputDevice, numOutputCategories * sizeof(float));

  float *bindings[2];
  bindings[inputBindingIndex] = inputDevice;
  bindings[outputBindingIndex] = outputDevice;

  double avgTime = 0;
  for (int i = 0; i < numRuns+1; ++i)
  {
    auto t0 = std::chrono::steady_clock::now();

    cudaMemcpy(inputDevice, input, inputSize, cudaMemcpyHostToDevice);
    context->execute(1, (void**)bindings);
    cudaMemcpy(output, outputDevice, numOutputCategories * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> predictions(output, output + numOutputCategories);
    std::chrono::duration<double> diff = std::chrono::steady_clock::now() - t0;

    if (i != 0)
    {
      avgTime += diff.count();
    }
  }

  cudaFree(inputDevice);
  cudaFree(outputDevice);

  cudaFreeHost(input);
  cudaFreeHost(output);

  return avgTime / numRuns;
}

std::vector<float> InferenceEngine::execute(std::string imagePath)
{
  cv::Mat image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
  cv::resize(image, image, cv::Size(inputWidth, inputHeight));
  float *input = imageToTensor(image);
  preprocessFn(input, 3, inputHeight, inputWidth);

  // allocate memory on host / device for input / output
  float *inputDevice, *outputDevice, *output;
  cudaHostAlloc(&output, numOutputCategories * sizeof(float), cudaHostAllocMapped);
  cudaMalloc(&inputDevice, inputSize);
  cudaMalloc(&outputDevice, numOutputCategories * sizeof(float));

  float *bindings[2];
  bindings[inputBindingIndex] = inputDevice;
  bindings[outputBindingIndex] = outputDevice;

  cudaMemcpy(inputDevice, input, inputSize, cudaMemcpyHostToDevice);
  context->execute(1, (void**)bindings);
  cudaMemcpy(output, outputDevice, numOutputCategories * sizeof(float), cudaMemcpyDeviceToHost);

  std::vector<float> predictions(output, output + numOutputCategories);

  cudaFree(inputDevice);
  cudaFree(outputDevice);

  cudaFreeHost(input);
  cudaFreeHost(output);

  // TODO: fix execute interface to match inputs and outputs of TF session.run
  return predictions;
}
} // namespace TensorRT

