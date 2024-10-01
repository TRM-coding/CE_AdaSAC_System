#include <armadillo>
#include <gtest/gtest.h>
#include <Tensor.hpp>

TEST(operator_Test , multy)
{
    using namespace MINI_MLsys;
    Tensor<double> T1(3,4,5),T2(4,5,5);
    Tensor<double> T3 = T1*T2;
}