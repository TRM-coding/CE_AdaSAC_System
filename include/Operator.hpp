#ifndef OPERATOR_HPP
#define OPERATOR_HPP
#include <Operand.hpp>
#include <Attribute.hpp>
#include <vector>
#include <string>
#include <map>
#include <ir.h>
#include <memory>
namespace MINI_MLsys
{
  class Operator
  {
  public:
    void initAttribute(std::map<std::string, pnnx::Attribute> attrs);
    Operator(pnnx::Operator *);
    Operator() = default;
    std::vector<std::shared_ptr<Operator>> pre_op;//
    std::vector<std::shared_ptr<Operator>> next_op;//
    std::vector<std::shared_ptr<Operand>> inputs; //
    std::vector<std::shared_ptr<Operand>> outputs;//
    std::string type; //
    std::string name; //
    // std::vector<std::string> inputsnames;
    std::map<std::string, pnnx::Parameter> params; //
    std::map<std::string, Attribute> attrs;//        
  };
} // namespace MINI_MLsys

#endif