#ifndef RELU_HPP
#define RELU_HPP
#include <Layer/abstract/Layer.hpp>
namespace MINI_MLsys {
class Relu : public Layer {
public:
  Relu(std::string layer_name) : Layer(layer_name) {}
  static float relu(const float &x) { return x > 0 ? x : 0; }
  void forward(const std::vector<std::shared_ptr<Tensor<float>>> &input,
               std::vector<std::shared_ptr<Tensor<float>>> &output) override;
  static bool deploy(const std::shared_ptr<Operator> &op);
};
} // namespace MINI_MLsys
#endif