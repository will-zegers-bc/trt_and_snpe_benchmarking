#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "TRTEngine.hpp"


PYBIND11_MODULE(tensor_rt, m)
{
    m.doc() = "TensorRT Inference Engine";
    pybind11::class_<TensorRT::NetConfig>(m, "NetConfig")
        .def(pybind11::init<std::string, std::string, std::string, std::string, int, int, int, int>(),
            pybind11::arg("plan_path"),
            pybind11::arg("input_node_name"),
            pybind11::arg("output_node_name"),
            pybind11::arg("preprocess_fn_name"),
            pybind11::arg("input_height"),
            pybind11::arg("input_width"),
            pybind11::arg("num_output_categories"),
            pybind11::arg("max_batch_size"))
        .def_readwrite("plan_path", &TensorRT::NetConfig::planPath)
        .def_readwrite("input_node_name", &TensorRT::NetConfig::inputNodeName)
        .def_readwrite("output_node_name", &TensorRT::NetConfig::outputNodeName)
        .def_readwrite("preprocess_fn_name", &TensorRT::NetConfig::preprocessFnName)
        .def_readwrite("input_height", &TensorRT::NetConfig::inputHeight)
        .def_readwrite("input_width", &TensorRT::NetConfig::inputWidth)
        .def_readwrite("num_output_categories", &TensorRT::NetConfig::numOutputCategories)
        .def_readwrite("max_batch_size", &TensorRT::NetConfig::maxBatchSize);

    pybind11::class_<TensorRT::TRTEngine>(m, "InferenceEngine")
        .def(pybind11::init<const TensorRT::NetConfig&>())
        .def("execute", &TensorRT::TRTEngine::execute)
        .def("measure_throughput", &TensorRT::TRTEngine::measureThroughput);
}
