#include <Operator.hpp>

bool MINI_MLsys::Operator::initAttribute(
    std::map<std::string, pnnx::Attribute> attrs) {

  for (auto &[name, attr] : attrs) {
    std::shared_ptr<Attribute> attr_data =
        std::make_shared<Attribute>(attr.data, attr.shape);
    this->attrs[name] = attr_data;
  }
  return true;
}

MINI_MLsys::Operator::Operator(pnnx::Operator *op) {
  this->name = op->name;
  this->type = op->type;
  this->params = op->params;
  this->initAttribute(op->attrs);

  for (const auto &opd : op->inputs) {
    std::shared_ptr<Operand> ip_opd = std::make_shared<Operand>(opd);
    this->inputs.push_back(ip_opd);
  }

  for (const auto &opd : op->outputs) {
    std::shared_ptr<Operand> op_opd = std::make_shared<Operand>(opd);
    this->inputs.push_back(op_opd);
  }
  this->pre_op.clear();
  this->next_op.clear();
}

void MINI_MLsys::Operator::forward(const Operand& input,Operand& output)
{
  if(this->layer==nullptr)
  {
    std::cout<<"Operator: Layer is nullptr."<<std::endl;
    return;
  }
  this->layer->forward(input,output);
  return;
}


// void MINI_MLsys::Operator::deploy()
// {
//   if(this->layer==nullptr)
//   {
//     std::cout<<"Operator: Layer is nullptr."<<std::endl;
//     return;
//   }
//   else{
//     auto registry=LayerRegister::get_registry();
//     auto type=this->type;
//     auto layer_creator_find=registry->find(type);
//     if(layer_creator_find==registry->end())
//     {
//       std::cout<<"Can not find the layer creator for type: "<<type<<std::endl;
//       return;
//     }
//     auto layer_creator=layer_creator_find->second;
//     if(!layer_creator)
//     {
//       std::cout<<"layer_creator is nullptr"<<std::endl;
//       return;
//     }
//     auto deploy_res=layer_creator(this->shared_from_this());
//   }
//   return;
// }