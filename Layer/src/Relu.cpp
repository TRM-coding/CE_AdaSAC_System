#include <Layer/LayerRegister.hpp>
#include <Layer/Relu.hpp>
#include <Layer/abstract/Layer.hpp>
#include <string>
namespace MINI_MLsys {

void Relu::forward(const std::vector<std::shared_ptr<Tensor<float>>> &input,
                   std::vector<std::shared_ptr<Tensor<float>>> &output) {
  for (const auto &x : input) {
    auto out = x->func(relu);
    output.push_back(std::make_shared<Tensor<float>>(out));
  }
}

bool Relu::deploy(const std::shared_ptr<Operator> &op) {
  if (op == nullptr) {
    std::cout << "Relu: Operator is nullptr." << std::endl;
    return false;
  }
  op->layer = std::make_shared<Relu>("Relu");
  op->layer->op_ = op;

  // no params and attrs

  return true;
}

LayerRegisterAssistant reg_relu("nn.ReLU", Relu::deploy);

} // namespace MINI_MLsys