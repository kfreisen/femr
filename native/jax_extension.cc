#include "jax_extension.hh"

#include <pybind11/stl.h>

#include <iostream>

#include "embedding_dot.hh"
#include "exp_mean.hh"
#include "gather_scatter.hh"
#include "local_attention.hh"
#include "common_jax_extension.hh"

namespace py = pybind11;

template <class To, class From>
std::enable_if_t<sizeof(To) == sizeof(From) &&
                     std::is_trivially_copyable_v<From> &&
                     std::is_trivially_copyable_v<To>,
                 To>
// constexpr support needs compiler magic
bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation additionally requires "
                  "destination type to be trivially constructible");

    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

template <typename T>
py::capsule convert_to_capsule(T *fn) {
    return py::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}


std::vector<std::tuple<std::string, py::capsule, std::string>> get_kernels() {
    std::vector<std::tuple<std::string, py::capsule, std::string>> result;

#define WRAP(result, A, B) result.emplace_back(#A, convert_to_capsule(A), B)

    WRAP(result, half_embedding_dot_forward, "gpu");
    WRAP(result, float_embedding_dot_forward, "gpu");
    WRAP(result, half_embedding_dot_backward, "gpu");
    WRAP(result, float_embedding_dot_backward, "gpu");

    WRAP(result, half_exp_mean_with_grad, "gpu");

    WRAP(result, convert_to_dense, "cpu");

    WRAP(result, half_local_attention_forward, "gpu");
    WRAP(result, half_local_attention_backward, "gpu");

    WRAP(result, float_gather_scatter, "gpu");

#undef WRAP

    return result;
}

void register_jax_extension(pybind11::module &root) {
    py::module m = root.def_submodule("jax");
    m.def("get_kernels", get_kernels);
    m.def("get_local_attention_shape", get_attention_shape);
    m.def("get_local_attention_data",
          [](int32_t b, int32_t n, int32_t k, int32_t w, bool causal) {
              const local_attention_info *data =
                  create_attention_info(b, n, k, w, causal);
              py::capsule capsule((void *)data,
                                  (void (*)(void *))free_attention_info);
              return capsule;
          });
}
