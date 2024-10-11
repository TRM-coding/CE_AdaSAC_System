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

MINI_MLsys::Operand::Operand(std::vector<std::shared_ptr<MINI_MLsys::Tensor<float>>>data,std::string name)
{
    this->data=data;
    this->name=name;
}

MINI_MLsys::Operand::Operand(const std::vector<Tensor<float>>& data,std::string name)
{
    if(data.size()==0)
    {
        std::cout<<"<Operand_constructor>::data is empty"<<std::endl;
        throw std::runtime_error("data is empty");
    }
    this->name=name;

    for(auto& data_i :data)
    {
        std::shared_ptr<Tensor<float>>dt=std::make_shared<Tensor<float>>(data_i.get_data());
        this->data.push_back(dt);
    }
}

bool MINI_MLsys::Operand::set_data(std::vector<Tensor<float>> data)
{
    if(data.size()==0)
    {
        std::cout<<"<Operand_set_data>::data is nullptr"<<std::endl;
        return false;
    }
    this->data.clear();
    for(auto& data_i :data)
    {
        std::shared_ptr<Tensor<float>>dt=std::make_shared<Tensor<float>>(data_i.get_data());
        this->data.push_back(dt);
    }
    return true;
}