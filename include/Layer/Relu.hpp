#pragma once
#include <Layer/LayerRegister.hpp>
#include <Layer/abstract/Layer.hpp>
#include <string>
// #include<Operand.hpp>
namespace MINI_MLsys {
class Relu : public Layer {
public:
  Relu(std::string layer_name) : Layer(layer_name) {
  }
  static float relu(const float &x) { return x > 0 ? x : 0; }
  void forward(const Operand &input, Operand &output) override;
  static bool deploy(std::shared_ptr<Operator> &op);
};
} // namespace MINI_MLsys
