
#include <Layer/Relu.hpp>

namespace MINI_MLsys {

void Relu::forward(const Operand &input, Operand &output) {
  auto inputs = input.data;
  std::vector<std::shared_ptr<Tensor<float>>>outputs;
  for (auto x : inputs) {
    auto op=x->func(relu);
    outputs.push_back(op);
  }
  std::string name = std::string(this->layer_name_ + "_output");
  auto ot=new Operand(outputs, name);
  output=*ot;
  return;
}

bool Relu::deploy(std::shared_ptr<Operator> op) {
  if (op == nullptr) {
    std::cout << "Relu: Operator is nullptr." << std::endl;
    return false;
  }
  op->layer = std::make_shared<Relu>("Relu");
  op->layer->op_ = op;
  // no params and attrs

  return true;
}

static LayerRegisterAssistant reg_relu("nn.ReLU", Relu::deploy);

} // namespace MINI_MLsys