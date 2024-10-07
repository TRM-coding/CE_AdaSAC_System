#include <gtest/gtest.h>
#include <Graph.hpp>
#include <iostream>
#include <iomanip>

// TEST(test_Graph, build)
// {
//     using namespace MINI_MLsys;
//     std::string param_pth = "../../model/test_linear.pnnx.param";
//     std::string bin_pth = "../../model/test_linear.pnnx.bin";
//     Graph G(param_pth, bin_pth);
//     std::cout << G.bin_path_ << std::endl;
//     std::cout << G.param_path_ << std::endl;
//     auto root = G.build();
//     ASSERT_TRUE(root != nullptr);
//     std::cout << root->name << std::endl;
//     std::cout << root->pre_op.size() << std::endl;
//     std::cout << root->next_op.size() << std::endl;
//     std::cout << root->inputs.size() << std::endl;
//     std::cout << root->outputs.size() << std::endl;
//     std::cout << root->type << std::endl;
//     std::cout << root->params.size() << std::endl;
//     std::cout << root->attrs.size() << std::endl;
// }

TEST(test_graph, resnet_all)
{
    using namespace MINI_MLsys;
    std::string param_pth = "../../model/resnet18_batch1.param";
    std::string bin_pth = "../../model/resnet18_batch1.pnnx.bin";
    // std::string param_pth = "../../model/test_linear.pnnx.param";
    // std::string bin_pth = "../../model/test_linear.pnnx.bin";
    Graph G(param_pth, bin_pth);
    std::cout << G.bin_path_ << std::endl;
    std::cout << G.param_path_ << std::endl;
    auto res = G.init();
    // ASSERT_EQ(res,0);
    int whight = 20;
    for (const auto &[opname, op] : G.operator_map_)
    {
        std::cout << "Operator: " << opname << std::endl;
        std::cout << std::endl;
        std::cout << "------------Pre Operator:--------------- " << std::endl;
        for (const auto &pre : op->pre_op)
        {
            std::cout << pre->name << " ";
        }
        std::cout << std::endl
                  << "---------------------------------------- " << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl
                  << "------------Next Operator:--------------- " << std::endl;
        for (const auto &next : op->next_op)
        {
            std::cout << next->name << " ";
        }
        std::cout << std::endl
                  << "---------------------------------------- " << std::endl;
        std::cout << std::endl;
    }

    std::cout << "------------------TOPO------------------" << std::endl;
    std::cout << "TOPO SIZE: " << G.topo_operators_.size() << std::endl;
    std::cout << G.root->next_op.size() << std::endl;
    for (const auto &op : G.topo_operators_)
    {
        std::cout << op->name << " ";
    }
    std::cout << std::endl
              << std::endl;
}

TEST(test_graph,show_attrs)
{
    using namespace MINI_MLsys;
    std::string param_pth = "../../model/simple_ops2.pnnx.param";
    std::string bin_pth = "../../model/simple_ops2.pnnx.bin";
    Graph G(param_pth, bin_pth);
    auto res=G.init();
    ASSERT_EQ(res,0);
    for(const auto& op:G.graph_->ops)
    {
        std::cout<<op->name<<":"<<op->params.size()<<std::endl;
        for(const auto& [name,attr]:op->params)
        {
            std::cout<<name<<std::endl;
        }
    }
}
