#include<gtest/gtest.h>
#include<Graph.hpp>
TEST(attribute_test,show_attrs)
{

    std::cout<<"******************"<<std::endl;
    std::string param_path="../../model/test_linear.pnnx.param";
    std::string bin_path="../../model/test_linear.pnnx.bin";
    MINI_MLsys::Graph G(param_path,bin_path);
    G.init();
    std::cout<<"%%%%%%%%%%%%%"<<std::endl;
    for(auto [name,op]:G.operator_map_)
    {
        std::cout<<"op "<<name<<" attrs:\n";
        for(auto [atname,attr]:op->attrs)
        {
            std::cout<<atname<<":\n";
            std::cout<<attr<<std::endl;
        }
        std::cout<<"----------------------\n";
    }
}

TEST(attribute_test,show_parm)
{
    std::cout<<"******************"<<std::endl;
    std::string param_path="../../model/test_linear.pnnx.param";
    std::string bin_path="../../model/test_linear.pnnx.bin";
    MINI_MLsys::Graph G(param_path,bin_path);
    G.init();
    std::cout<<"%%%%%%%%%%%%%"<<std::endl;
    for(auto [name,op]:G.operator_map_)
    {
        std::cout<<"op "<<name<<" params:\n";
        for(auto [atname,attr]:op->params)
        {
            std::cout<<atname<<":\n";
            std::cout<<attr<<std::endl;
        }
        std::cout<<"----------------------\n";
    }
}