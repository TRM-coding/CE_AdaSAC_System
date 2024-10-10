#include<gtest/gtest.h>

int main(int argc,char* argv[])
{
    testing::InitGoogleTest(&argc,argv);
    testing::GTEST_FLAG(filter) = "attribute_test.*";
    std::cout<<"Start test...\n";
    return RUN_ALL_TESTS();
}