#ifndef MINI_ML_Linear_HPP
#define MINI_ML_Linear_HPP
#include <Layer/Linear.hpp>
#include <Layer/abstract/Layer.hpp>
#include <Operator.hpp>
#include <Tensor.hpp>
#include <memory>
#include <string>
#include <vector>
namespace MINI_MLsys {
class Linear : public Layer {
public:
  Linear(std::string layer_name) : Layer(layer_name) {}
  void forward(const Operand &input,Operand &output) override ;
  static bool deploy(const std::shared_ptr<Operator> &op);

private:
  std::shared_ptr<Tensor<float>> weight_;
  std::shared_ptr<Tensor<float>> bias_;
  void set_bias();
  void set_weight();
};
} // namespace MINI_MLsys
#endif