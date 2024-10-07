#include <gtest/gtest.h>
#include <ir.h>
#include <Tensor.hpp>
#include <memory>
#include <string>
#include <Operand.hpp>

TEST(testOperand, testConstruct)
{
    // std::shared_ptr<pnnx::Graph> G=std::make_share<pnnx::Graph>();
    std::string param_pth = "../../model/test_linear.pnnx.param";
    std::string bin_pth = "../../model/test_linear.pnnx.bin";
    std::shared_ptr<pnnx::Graph> G = std::make_shared<pnnx::Graph>();
    bool res = G->load(param_pth, bin_pth);
    ASSERT_EQ(res, 0);
    for (const auto &x : G->operands)
    {
        if (x == nullptr)
        {
            std::cout << "nullptr" << std::endl;
        }
        else
        {
            std::cout << "name:" << x->name << std::endl;
            MINI_MLsys::Operand O(x);
            std::cout << "name:" << O.name << std::endl;
            std::cout<<"shape:"<<O.shape[0]<<" "<<O.shape[1]<<" "<<O.shape[2]<<std::endl;
            std::cout<<"type:"<<O.type<<std::endl;
            std::cout<<"params:"<<O.params.size()<<std::endl;
            
        }
    }
}

TEST(testOperand, testSetData)
{
    std::string param_pth = "../../model/test_linear.pnnx.param";
    std::string bin_pth = "../../model/test_linear.pnnx.bin";
    std::shared_ptr<pnnx::Graph> G = std::make_shared<pnnx::Graph>();
    bool res = G->load(param_pth, bin_pth);
    ASSERT_EQ(res, 0);
    for (const auto &x : G->operands)
    {
        if (x == nullptr)
        {
            std::cout << "nullptr" << std::endl;
        }
        else
        {
            std::cout << "name:" << x->name << std::endl;
            MINI_MLsys::Operand O(x);
            std::cout << "name:" << O.name << std::endl;
            std::cout<<"shape:"<<O.shape[0]<<" "<<O.shape[1]<<" "<<O.shape[2]<<std::endl;
            std::cout<<"type:"<<O.type<<std::endl;
            std::cout<<"params:"<<O.params.size()<<std::endl;
            std::shared_ptr<MINI_MLsys::Tensor<float>> T = std::make_shared<MINI_MLsys::Tensor<float>>(O.shape);
            O.set_data(T);
            ASSERT_EQ(O.data, T);
        }
    }
}