#include<Layer/abstract/Layer.hpp>
namespace MINI_MLsys {
  void Layer::set_bias(const std::shared_ptr<Tensor<float>> &bias) {
    std::cout<<"RUNING BASIC LAYER SET BIAS"<<std::endl;
    return;
  }

  void Layer::set_bias(const std::vector<float> &bias) {
    std::cout<<"RUNING BASIC LAYER SET BIAS"<<std::endl;
    return;
  }

  void Layer::set_weight(const std::vector<float> &weight) {
    std::cout<<"RUNING BASIC LAYER SET WEIGHT"<<std::endl;
    return;
  }

  void Layer::set_weight(const std::shared_ptr<Tensor<float>> &weight) {
    std::cout<<"RUNING BASIC LAYER SET WEIGHT"<<std::endl;
    return;
  }

  const std::shared_ptr<Tensor<float>> Layer::get_bias() {
    std::cout<<"RUNING BASIC LAYER GET BIAS"<<std::endl;
    return std::shared_ptr<Tensor<float>>();
  }

  const std::shared_ptr<Tensor<float>> Layer::get_weight() {
    std::cout<<"RUNING BASIC LAYER GET WEIGHT"<<std::endl;
    return std::shared_ptr<Tensor<float>>();
  }
}