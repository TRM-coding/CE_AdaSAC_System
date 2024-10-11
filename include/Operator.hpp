#ifndef OPERATOR_HPP
#define OPERATOR_HPP
#include <Operand.hpp>
#include <Attribute.hpp>
#include <vector>
#include <string>
#include <map>
#include <ir.h>
#include <memory>
#include <Layer/abstract/Layer.hpp>
namespace MINI_MLsys
{
  class Layer;
  class Operator
  {
  public:
    bool initAttribute(std::map<std::string, pnnx::Attribute> attrs);
    Operator(pnnx::Operator *);
    Operator() = default;


    std::shared_ptr<Layer> layer;
    void forward(const Operand &input, Operand &output);

    std::vector<std::shared_ptr<Operator>> pre_op;  //
    std::vector<std::shared_ptr<Operator>> next_op; //
    std::vector<std::shared_ptr<Operand>> inputs;   //
    std::vector<std::shared_ptr<Operand>> outputs;  //
    std::string type;                               //
    std::string name;                               //
    // std::vector<std::string> inputsnames;
    std::map<std::string, pnnx::Parameter> params; //
    std::map<std::string, std::shared_ptr<Attribute>> attrs;        //
  };
} // namespace MINI_MLsys

#endif