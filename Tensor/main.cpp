#include <gtest/gtest.h>
// #include <glog/logging.h>

int main(int argc , char * argv[]){
    testing::InitGoogleTest(&argc, argv);
    std::cout<<"Start test...\n";
    return RUN_ALL_TESTS();
}

