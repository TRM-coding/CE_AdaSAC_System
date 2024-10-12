#include <Layer/sigmoid.hpp>

namespace MINI_MLsys {

void Sigmoid::forward(const Operand &input, Operand &output) {
  auto inputs = input.data;
  auto outputs = std::vector<std::shared_ptr<Tensor<float>>>();
  for (const auto &x : inputs) {
    auto out = x->func(sigmoid);
    outputs.push_back(out);
  }
  auto name = this->layer_name_ + "_output";
  output = Operand(outputs, name);
  return;
}

bool Sigmoid::deploy(std::shared_ptr<Operator> &op) {
  if (op == nullptr) {
    std::cout << "Sigmoid: Operator is nullptr." << std::endl;
    return false;
  }
  op->layer = std::make_shared<Sigmoid>("Sigmoid");
  op->layer->op_ = op;
  // no params and attrs
  return true;
}
LayerRegisterAssistant reg_relu("nn.Sigmoid", Sigmoid::deploy);
} // namespace MINI_MLsys