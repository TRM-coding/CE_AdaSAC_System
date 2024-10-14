#include<gtest/gtest.h>
#include<Graph.hpp>
#include<Layer/sigmoid.hpp>
TEST(test_Graph,RUN)
{
    using namespace MINI_MLsys;
    // getchar();
    // extern LayerRegisterAssistant reg_sig;
    // std::cout<<reg_sig.type<<std::endl;
    std::string param_pth = "../../model/test_linear.pnnx.param";
    std::string bin_pth = "../../model/test_linear.pnnx.bin";
    MINI_MLsys::Graph G(param_pth,bin_pth);
    G.init();
    std::vector<Tensor<float>>ip;
    std::cout<<"INPUT::"<<std::endl;
    for(int i=1;i<=3;i++)
    {
        std::vector<uint64_t>shape={32,1,1};
        std::vector<float>data;
        for(int i=0;i<32;i++)
        {
            data.push_back(0);
        }
        Tensor<float>t(shape,data);
        std::cout<<t<<std::endl;
        ip.push_back(t);
    }
    std::cout<<"-------------------------"<<std::endl;
    auto res=G.RUN(ip);
    auto op=G.operator_map_["linear"];
    auto x=op->attrs;
    std::cout<<"bias::"<<std::endl;
    for(auto p:x["bias"]->data_)
    {
        std::cout<<p<<std::endl;
    }
    std::cout<<"outputs::::::"<<std::endl;
    for(auto oi:res)
    {
        std::cout<<oi<<std::endl;
        
    }
}