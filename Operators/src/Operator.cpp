#include <Attribute.hpp>
#include <Operand.hpp>
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
