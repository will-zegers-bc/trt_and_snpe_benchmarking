#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "SNPEEngine.hpp"


PYBIND11_MODULE(snpe, m)
{
    m.doc() = "SNPE Inference Engine";

    pybind11::class_<SNPE::SNPEEngine>(m, "InferenceEngine")
        .def(pybind11::init<const std::string&, const std::string&>())
        .def("execute", &SNPE::SNPEEngine::execute)
        .def("measure_latency", &SNPE::SNPEEngine::measureLatency);
}
