#include <Layer.hpp>

void MINI_MLsys::Layer::set_bias(const std::shared_ptr<Tensor<float>> &bias) {
  return;
}

void MINI_MLsys::Layer::set_bias(const std::vector<float> &bias) { return; }

void MINI_MLsys::Layer::set_weight(const std::vector<float> &weight) { return; }

void MINI_MLsys::Layer::set_weight(
    const std::shared_ptr<Tensor<float>> &weight) {
  return;
}

const std::shared_ptr<MINI_MLsys::Tensor<float>> MINI_MLsys::Layer::get_bias() {
  return std::shared_ptr<Tensor<float>>();
}

const std::shared_ptr<MINI_MLsys::Tensor<float>>
MINI_MLsys::Layer::get_weight() {
  return std::shared_ptr<Tensor<float>>();
}
