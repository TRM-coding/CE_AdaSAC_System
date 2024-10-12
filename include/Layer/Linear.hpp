#pragma once
#include <Layer/LayerRegister.hpp>
#include <Layer/abstract/Layer.hpp>
#include <Tensor.hpp>
#include <memory>
#include <string>
#include <vector>
namespace MINI_MLsys {
class Operator;
class Linear : Layer {
public:
  Linear(std::string layer_name) : Layer(layer_name) {}
  void forward(const Operand &input,Operand &output) override ;
  static bool deploy(std::shared_ptr<Operator> &op);


  // std::shared_ptr<Tensor<float>> weight_;
  // std::shared_ptr<Tensor<float>> bias_;
  // void set_bias()override;
  // void set_weight()override;
};
} // namespace MINI_MLsys
