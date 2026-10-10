
#include <torch/extension.h>
namespace ornith_v37_h8 {
at::Tensor sm70_d256_gqa_v37_fwd(const at::Tensor&, const at::Tensor&,
                                 const at::Tensor&, at::Tensor&, double, bool);
void reserve_workspace(const at::Tensor&);
}  // namespace ornith_v37_h8
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", &ornith_v37_h8::sm70_d256_gqa_v37_fwd);
  m.def("reserve_workspace", &ornith_v37_h8::reserve_workspace);
}
