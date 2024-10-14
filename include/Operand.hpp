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
        std::vector<uint64_t> shape;//
        std::string name;//
        std::map<std::string, pnnx::Parameter> params;//
        std::vector<std::shared_ptr<Tensor<float>>> data;
        explicit Operand(pnnx::Operand* pnn_op);
        explicit Operand(const std::vector<Tensor<float>>& data,std::string name);
        explicit Operand(const std::vector<std::shared_ptr<Tensor<float>>>& data,std::string name);
        Operand()=default;
        bool set_data(std::vector<Tensor<float>> data);
    };
}//namespace MINI_MLsys

#endif