#include<gtest/gtest.h>
#include<Graph.hpp>
#include<Layer/sigmoid.hpp>

int main(int argc,char* argv[])
{
    testing::InitGoogleTest(&argc,argv);
    testing::GTEST_FLAG(filter) = "test_Graph.RUN";
    std::cout<<"Start test...\n";
    return RUN_ALL_TESTS();
}


// #include<gtest/gtest.h>
// #include<Graph.hpp>
// #include<Layer/sigmoid.hpp>

// int main()
// {
//     using namespace MINI_MLsys;
    
//     // std::cout<<reg_sig.type<<std::endl;
//     std::string param_pth = "../../model/test_linear.pnnx.param";
//     std::string bin_pth = "../../model/test_linear.pnnx.bin";
//     MINI_MLsys::Graph G(param_pth,bin_pth);
//     G.init();
//     std::vector<Tensor<float>>ip;
//     std::cout<<"INPUT::"<<std::endl;
//     for(int i=1;i<=3;i++)
//     {
//         std::vector<uint64_t>shape={3,3,1};
//         std::vector<float>data={1,0,0,
//                                 0,1,0,
//                                 0,0,1};
//         Tensor<float>t(shape,data);

//         std::cout<<t<<std::endl;
//         ip.push_back(t);
//     }
//     std::cout<<"-------------------------"<<std::endl;
//     auto res=G.RUN(ip);
//     for(auto oi:res)
//     {
//         std::cout<<oi<<std::endl;
//     }
// }




