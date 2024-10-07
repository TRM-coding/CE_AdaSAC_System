#include<gtest/gtest.h>
#include<memory>
#include<Operator.hpp>
#include<ir.h>
#include<string>
#include<iomanip>
TEST(test_Operator,constructor)
{
    // std::shared_ptr<pnnx::Graph> G=std::make_share<pnnx::Graph>();
    std::string param_pth = "../../model/test_linear.pnnx.param";
    std::string bin_pth = "../../model/test_linear.pnnx.bin";
    std::shared_ptr<pnnx::Graph> G = std::make_shared<pnnx::Graph>();
    bool res = G->load(param_pth, bin_pth);
    ASSERT_EQ(res, 0);
    for (const auto &x : G->ops)
    {
        if (x == nullptr)
        {
            std::cout << "nullptr" << std::endl;
        }
        else
        {
            std::cout << "name:" << x->name << std::endl; 
        }
    }
}

TEST(test_Operator,attrs)
{
    // std::shared_ptr<pnnx::Graph> G=std::make_share<pnnx::Graph>();
    std::string param_pth = "../../model/test_linear.pnnx.param";
    std::string bin_pth = "../../model/test_linear.pnnx.bin";
    std::shared_ptr<pnnx::Graph> G = std::make_shared<pnnx::Graph>();
    bool res = G->load(param_pth, bin_pth);
    ASSERT_EQ(res, 0);
    for (const auto &x : G->ops)
    {
        if (x == nullptr)
        {
            std::cout << "nullptr" << std::endl;
        }
        else
        {
            int width=20;
            MINI_MLsys::Operator O(x);
            std::cout << "name:" << O.name << std::endl;
            std::cout<<"---------------------------------------------"<<std::endl;
            std::cout<<std::right<<std::setw(width)<<"type:"<<O.type<<std::endl;
            std::cout<<std::right<<std::setw(width)<<"pre_op.size:"<<O.pre_op.size()<<std::endl;
            std::cout<<std::right<<std::setw(width)<<"next_op.size:"<<O.next_op.size()<<std::endl;
            std::cout<<std::right<<std::setw(width)<<"inputs.size:"<<O.inputs.size()<<std::endl;
            std::cout<<std::right<<std::setw(width)<<"outputs.size:"<<O.outputs.size()<<std::endl;
            std::cout<<std::right<<std::setw(width)<<"params.size:"<<O.params.size()<<std::endl;
            std::cout<<std::right<<std::setw(width)<<"attrs.size:"<<O.attrs.size()<<std::endl;
        }
        std::cout<<std::endl;
    }
}