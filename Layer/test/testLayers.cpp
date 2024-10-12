#include<gtest/gtest.h>
#include<Layer/Relu.hpp>
#include<Layer/sigmoid.hpp>
#include<Tensor.hpp>
// #include<Operand.hpp>

TEST(test_Layer,relu)
{
  using namespace MINI_MLsys;
  // Relu R("Relu");
  Tensor<float> input(2,2,1);
  input[0]={{-1,-2},{3,4}};
  std::cout<<"input:"<<std::endl;
  std::cout<<input<<std::endl;

  std::vector<std::shared_ptr<Tensor<float>>> input_v;
  input_v.push_back(std::make_shared<Tensor<float>>(input));
  std::vector<std::shared_ptr<Tensor<float>>> output_v;
  std::cout<<"output:"<<std::endl;
  // R.forward(input_v,output_v);
  std::cout<<(*(input_v[0]))<<std::endl;
}

TEST(test_Layer,deploy)
{
  using namespace MINI_MLsys;
  Relu R("Relu");
  // Sigmoid S("Sigmoid");
  std::vector<std::shared_ptr<Tensor<float>>> data;
  auto op = Operand(data,"input");
  // std::shared_ptr<Operator> op=std::make_shared<Operator>();
  // ASSERT_TRUE(R.deploy(op));
}