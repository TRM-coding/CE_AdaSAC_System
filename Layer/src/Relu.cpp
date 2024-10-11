#include <Layer/LayerRegister.hpp>
#include <Layer/Relu.hpp>
#include <Layer/abstract/Layer.hpp>
#include <string>
namespace MINI_MLsys {

void Relu::forward(const Operand &input,Operand &output) {
  auto inputs=input.data;
  auto outputs=std::vector<std::shared_ptr<Tensor<float>>>();
  for (const auto &x : inputs) {
    auto out = x->func(relu);
    outputs.push_back(std::make_shared<Tensor<float>>(out));
  }
  auto name=this->layer_name_+"_output";
  output=Operand(outputs,name);
  return;
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