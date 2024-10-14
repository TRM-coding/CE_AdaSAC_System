#pragma once

#include <Operand.hpp>
#include <Operator.hpp>
#include <Tensor.hpp>
#include <ir.h>
#include <memory>
#include <string>
#include <vector>
namespace MINI_MLsys {

class Operator;

class Layer // 默认实现为无权重Layer
{
public:
  std::string layer_name_;
  std::weak_ptr<Operator> op_;
  std::map<std::string, pnnx::Parameter> params;
  std::map<std::string, std::shared_ptr<Attribute>> attrs;

  virtual ~Layer() = default;
  explicit Layer(std::string layer_name) : layer_name_(layer_name) {}

  virtual void forward(const Operand &input, Operand &output) = 0;

  // virtual set_attr and params;

  // virtual bool deploy(std::shared_ptr<Operator> &op) = 0;

  std::string get_layer_name() const { return layer_name_; }

  // std::shared_ptr<Tensor<float>> weight_;
  // std::shared_ptr<Tensor<float>> bias_;
  // virtual void set_bias();
  // virtual void set_weight();
};
} // namespace MINI_MLsys
  // MINI_ML_LAYER_HPP