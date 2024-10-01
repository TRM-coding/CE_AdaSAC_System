#include <armadillo>
#include <gtest/gtest.h>
#include <Tensor.hpp>
#include <iostream>
#include <time.h>

TEST(operator_Test, multy)
{
    using namespace MINI_MLsys;

    int cnt = 10;
    while (cnt--)
    {
        srand(time(0));
        int row1 = rand()%100;
        int col1 = rand()%100;
        int col2 = rand()%100;
        int cha = rand()%100;
        Tensor<double> T1(row1, col1, cha), T2(col1, col2, cha);
        T1.randi(0, 3);
        T2.randi(0, 3);
        Tensor<double> T3 = T1 * T2;
        for (size_t i = 0; i < T3.channel_n(); i++)
        {
            auto i_3 = T3.channel(i);
            auto i_1 = T1.channel(i);
            auto i_2 = T2.channel(i);
            ASSERT_EQ(arma::approx_equal(i_3, i_1 * i_2, "absdiff", 1e-5), true);
        }
    }

    // std::cout<<T1<<std::endl;
    // std::cout<<T2<<std::endl;
    // std::cout<<T3<<std::endl;
}