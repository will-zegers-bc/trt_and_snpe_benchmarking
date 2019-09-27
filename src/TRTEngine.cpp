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
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>

#include "TRTEngine.hpp"


using namespace nvinfer1;


namespace TensorRT
{

class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
    return;
  }
} gLogger;

NetConfig::NetConfig(std::string planPath,
                     std::string inputNodeName,
                     std::string outputNodeName,
                     int inputHeight,
                     int inputWidth,
                     int inputChannels,
                     int numOutputCategories,
                     int maxBatchSize) 
  : planPath(planPath)
  , inputNodeName(inputNodeName)
  , outputNodeName(outputNodeName)
  , inputHeight(inputHeight)
  , inputWidth(inputWidth)
  , inputChannels(inputChannels)
  , numOutputCategories(numOutputCategories)
  , maxBatchSize(maxBatchSize)
{
}

float *imageToTensor(pybind11::array_t<float, pybind11::array::c_style> image)
{
  const size_t height = image.shape(0);
  const size_t width = image.shape(1);
  const size_t channels = image.shape(2);
  const size_t numel = height * width * channels;

  const size_t strides[3] = { height * width, width, 1 };

  float * tensor;
  cudaHostAlloc((void**)&tensor, numel * sizeof(float), cudaHostAllocMapped);

  for (size_t i = 0; i < height; i++) 
  {
    for (size_t j = 0; j < width; j++) 
    {
      for (size_t k = 0; k < channels; k++) 
      {
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        tensor[offset] = *image.data(i, j, k);
      }
    }
  }

  return tensor;
}

TRTEngine::TRTEngine(const NetConfig &netConfig) 
  : numOutputCategories(netConfig.numOutputCategories)
  , runtime(createInferRuntime(gLogger))
{
  std::ifstream planFile(netConfig.planPath);
  std::stringstream planBuffer;
  planBuffer << planFile.rdbuf();
  std::string plan = planBuffer.str();

  engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr); 
  context = engine->createExecutionContext();

  // allocate memory on host / device for input / output
  inputSize = netConfig.inputHeight * netConfig.inputWidth * netConfig.inputChannels * sizeof(float);
  cudaMalloc(&inputDevice, inputSize);
  cudaMalloc(&outputDevice, numOutputCategories * sizeof(float));
  cudaHostAlloc(&output, numOutputCategories * sizeof(float), cudaHostAllocMapped);

  inputBindingIndex = engine->getBindingIndex(netConfig.inputNodeName.c_str());
  bindings[inputBindingIndex] = inputDevice;
  outputBindingIndex = engine->getBindingIndex(netConfig.outputNodeName.c_str());
  bindings[outputBindingIndex] = outputDevice;
}

TRTEngine::~TRTEngine()
{
  cudaFreeHost(output);
  cudaFree(outputDevice);
  cudaFree(inputDevice);

  context->destroy();  
  engine->destroy();
  runtime->destroy();
}

std::vector<float> TRTEngine::execute(pybind11::array_t<float, pybind11::array::c_style> image)
{
  float *input = imageToTensor(image);

  cudaMemcpy(inputDevice, input, inputSize, cudaMemcpyHostToDevice);
  context->execute(1, (void**)bindings);
  cudaMemcpy(output, outputDevice, numOutputCategories * sizeof(float), cudaMemcpyDeviceToHost);

  std::vector<float> predictions(output, output + numOutputCategories);

  cudaFreeHost(input);

  return predictions;
}
} // namespace TensorRT

