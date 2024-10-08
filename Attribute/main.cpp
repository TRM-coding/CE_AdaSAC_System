#include<gtest/gtest.h>

int main()
{
    testing::InitGoogleTest();
    std::cout<<"Start testing..."<<std::endl;
    return RUN_ALL_TESTS();
}