#pragma once
#include <Layer/abstract/Layer.hpp>
#include <Layer/LayerRegister.hpp>
namespace MINI_MLsys {
class Sigmoid : public Layer {
public:
  Sigmoid(std::string layer_name_) : Layer(layer_name_) {
    LayerRegisterAssistant reg_relu("F.sigmoid", Sigmoid::deploy);
  }
  static float sigmoid(const float &x) { return 1 / (1 + exp(-x)); }
  static bool deploy(std::shared_ptr<Operator> op);
  void forward(const Operand &input,
               Operand &output) override;
};
} // namespace MINI_MLsys
