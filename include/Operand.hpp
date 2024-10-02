#include<string>
#include<vector>
#include<memory>
#include<Tensor.hpp>
namespace MINI_MLsys
{
    struct Operand
    {
        std::string name;
        std::vector<uint32_t>shape;
        std::vector<std::shared_ptr<Tensor<float>>>datas;
    };
}