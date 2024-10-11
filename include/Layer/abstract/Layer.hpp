#ifndef MINI_ML_LAYER_HPP
#define MINI_ML_LAYER_HPP

#include <Operator.hpp>
#include <Tensor.hpp>
#include <memory>
#include <string>
#include <vector>
#include<Operand.hpp>
namespace MINI_MLsys {
class Operator;
class Layer // 默认实现为无权重Layer
{
public:
  std::string layer_name_;
  std::weak_ptr<MINI_MLsys::Operator> op_;
  std::map<std::string,pnnx::Parameter> params;
  std::map<std::string,std::shared_ptr<Attribute>>attrs;

  virtual ~Layer() = default;
  explicit Layer(std::string layer_name) : layer_name_(layer_name) {}

  virtual void forward(const Operand &input,Operand &output) = 0;

  //virtual set_attr and params;

  

  // virtual bool deploy(std::shared_ptr<Operator> &op) = 0;

  std::string get_layer_name() const { return layer_name_; }
};
} // namespace MINI_MLsys
#endif // MINI_ML_LAYER_HPP