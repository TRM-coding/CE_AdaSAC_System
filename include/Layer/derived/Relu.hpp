#include<Layer/abstract/Layer.hpp>
namespace MINI_MLsys {
class Relu : public Layer {
public:
  Relu(std::string layer_name) : Layer(layer_name) {}
  void forward(const std::vector<std::shared_ptr<Tensor<float>>> &input,
               std::vector<std::shared_ptr<Tensor<float>>> &output) override {
    output = input;
    for (size_t i = 0; i < input.size(); i++) {
      
    }
  }
  bool deploy(std::shared_ptr<Operator> &op) override {
    return true;
  }
};
} // namespace MINI_MLsys