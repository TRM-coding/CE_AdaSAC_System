#ifndef SIGMOID_HPP
#define SIGMOID_HPP
#include <Layer/abstract/Layer.hpp>
namespace MINI_MLsys {
class Sigmoid : public Layer {
public:
  Sigmoid(std::string layer_name_) : Layer(layer_name_) {}
  static float sigmoid(const float &x) { return 1 / (1 + exp(-x)); }
  static bool deploy(const std::shared_ptr<Operator> &op);
  void forward(const std::vector<std::shared_ptr<Tensor<float>>> &input,
               std::vector<std::shared_ptr<Tensor<float>>> &output) override;
};
} // namespace MINI_MLsys
#endif