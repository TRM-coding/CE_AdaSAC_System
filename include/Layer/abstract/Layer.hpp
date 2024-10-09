#ifndef MINI_ML_LAYER_HPP
#define MINI_ML_LAYER_HPP

#include <Operator.hpp>
#include <Tensor.hpp>
#include <memory>
#include <string>
#include <vector>
namespace MINI_MLsys {
class Operator;
class Layer // 默认实现为无权重Layer
{
public:
  std::string layer_name_;
  std::weak_ptr<MINI_MLsys::Operator> op_;

  virtual ~Layer() = default;
  explicit Layer(std::string layer_name) : layer_name_(layer_name) {}

  virtual void forward(const std::vector<std::shared_ptr<Tensor<float>>> &input,
                       std::vector<std::shared_ptr<Tensor<float>>> &output) = 0;

  virtual void set_bias(const std::shared_ptr<Tensor<float>> &bias);
  virtual void set_bias(const std::vector<float> &bias);
  virtual void set_weight(const std::vector<float> &weight);
  virtual void set_weight(const std::shared_ptr<Tensor<float>> &weight);

  virtual const std::shared_ptr<Tensor<float>> get_bias();
  virtual const std::shared_ptr<Tensor<float>> get_weight();

  // virtual bool deploy(std::shared_ptr<Operator> &op) = 0;

  std::string get_layer_name() const { return layer_name_; }
};
} // namespace MINI_MLsys
#endif // MINI_ML_LAYER_HPP