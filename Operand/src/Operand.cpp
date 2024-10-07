#include "Operand.hpp"

MINI_MLsys::Operand::Operand(pnnx::Operand* pnn_op)
{
    if(!pnn_op)
    {
        std::cout<<"<Operand_constructor>::pnn_op is nullptr"<<std::endl;
        throw std::runtime_error("pnn_op is nullptr");
    }
    this->shape=pnn_op->shape;
    this->type=pnn_op->type;
    this->name=pnn_op->name;
    this->params=pnn_op->params;
    
}

bool MINI_MLsys::Operand::set_data(std::shared_ptr<Tensor<float>> data)
{
    if(data.get()==nullptr)
    {
        std::cout<<"<Operand_set_data>::data is nullptr"<<std::endl;
        return false;
    }
    this->data=data;
    return true;
}