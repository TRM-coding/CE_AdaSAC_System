#ifndef OPERAND_HPP
#define OPERAND_HPP
#include <string>
#include <vector>
#include <memory>
#include <Tensor.hpp>
#include <ir.h>
namespace MINI_MLsys
{
    class Operand
    {
    public:
        int type;//
        std::vector<int> shape;//
        std::string name;//
        std::map<std::string, pnnx::Parameter> params;//
        std::shared_ptr<Tensor<float>> data;
        Operand(pnnx::Operand* pnn_op);
        bool set_data(std::shared_ptr<Tensor<float>> data);
    };
}//namespace MINI_MLsys

#endif