#include <Layer/Linear.hpp>
namespace MINI_MLsys {
void Linear::forward(const Operand &input,
                     Operand &output) {
  auto inputs=input.data;
  std::vector<std::shared_ptr<Tensor<float>>>outputs;
  for(auto ip:inputs)
  {
    auto op = (*this->weight_)*(*ip)+(*this->bias_);
    outputs.push_back(std::shared_ptr<Tensor<float>>(&op));
  }
  auto name=this->layer_name_+"_output";
  output=Operand(outputs,name);
  return;
}

bool Linear::deploy(const std::shared_ptr<Operator> &op) {
  if (op == nullptr) {
    std::cout << "Linear: Operator is nullptr." << std::endl;
    return false;
  }
  op->layer = std::make_shared<Linear>("Linear");
  op->layer->op_ = op;
  auto linear = std::dynamic_pointer_cast<Linear>(op->layer);
  linear->set_bias();
  linear->set_weight();
  return true;
}

void Linear::set_bias() {
  auto operator_ = op_;
  auto op = operator_.lock();
  auto operator_params = op->params;
  auto operator_attrs = op->attrs;
  int flag=1;
  if (operator_params.find("bias") == operator_params.end()) {
    std::cout << "Linear: No bias found." << std::endl; 
    flag=0;
  }
  if (operator_attrs.find("bias") == operator_attrs.end()) {
    std::cout << "Linear: No bias found." << std::endl;
    flag=0;
  }
  auto bias = operator_attrs["bias"];
  std::vector<float>bias_data;
  if(flag)bias_data = bias->data_;
  if(operator_params.find("out_features")==operator_params.end())
  {
    std::cout<<"Linear: No out_features found."<<std::endl;
    flag=0;
  }
  auto out_features=operator_params["out_features"].i;
  std::vector<int> shape={out_features,1,1};
  std::shared_ptr<Tensor<float>> bias_ =
      std::make_shared<Tensor<float>>(shape,bias_data);
  this->bias_ = bias_;
  return;
}

void Linear::set_weight() {
  auto operator_ = op_;
  auto op = operator_.lock();
  auto operator_params = op->params;
  auto operator_attrs = op->attrs;
  if (operator_params.find("weight") == operator_params.end()) {
    std::cout << "Linear: No weight found." << std::endl;
    return;
  }
  if (operator_attrs.find("weight") == operator_attrs.end()) {
    std::cout << "Linear: No weight found." << std::endl;
    return;
  }
  auto weight = operator_attrs["weight"];
  auto weight_data = weight->data_;
  if(operator_params.find("in_features")==operator_params.end())
  {
    std::cout<<"Linear: No in_features found."<<std::endl;
    return;
  }
  if(operator_params.find("out_features")==operator_params.end())
  {
    std::cout<<"Linear: No out_features found."<<std::endl;
    return;
  }
  auto in_features=operator_params["in_features"].i;
  auto out_features=operator_params["out_features"].i;
  std::vector<int> shape={out_features,in_features,1};
  std::shared_ptr<Tensor<float>> weight_ =
      std::make_shared<Tensor<float>>(shape,weight_data);
  this->weight_ = weight_;
  return;
}
} // namespace MINI_MLsys