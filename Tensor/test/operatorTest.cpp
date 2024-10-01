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
      int row1 = abs(rand() % 100)+1;
      int col1 = abs(rand() % 100)+1;
      int col2 = abs(rand() % 100)+1;
      int cha = abs(rand() % 100)+1;
      Tensor<float> T1(row1, col1, cha), T2(col1, col2, cha);
      T1.randu();
      T2.randu();
      Tensor<float> T3 = T1 * T2;
      for (size_t i = 0; i < T3.channel_n(); i++)
      {
         auto i_3 = T3.channel(i);
         auto i_1 = T1.channel(i);
         auto i_2 = T2.channel(i);
         ASSERT_EQ(arma::approx_equal(i_3, i_1 * i_2, "absdiff", 1e-5), true);
      }
   }
}

TEST(operator_Test, add)
{
   using namespace MINI_MLsys;
   int cnt = 10;
   while (cnt--)
   {
      srand(time(0));
      int col=abs(rand()%100)+1;
      int row=abs(rand()%100)+1;
      int cha=abs(rand()%100)+1;
      Tensor<float>T1(row,col,cha);
      Tensor<float>T2(row,col,cha);
      T1.randn();
      T2.randn();
      Tensor<float>T3=T1+T2;
      for(size_t i=0;i<T3.channel_n();i++)
      {
         auto i_3=T3.channel(i);
         auto i_2=T2.channel(i);
         auto i_1=T1.channel(i);
         ASSERT_EQ(arma::approx_equal(i_3,i_1+i_2,"absdiff",1e-5),true);
      }
   }
}


TEST(operator_Test,div)
{
   using namespace MINI_MLsys;
   int cnt=10;
   srand(time(0));
   while(cnt--)
   {
      
      int row=abs(rand()%100)+1;
      int col=abs(rand()%100)+1;
      int cha=abs(rand()%100)+1;
      std::cout<<row<<" "<<col<<" "<<cha<<std::endl;
      Tensor<float>T1(row,col,cha);
      Tensor<float>T2(row,col,cha);
      T1.randn();
      T2.randu();
      Tensor<float>T3=T1/T2;
      for(size_t i=0;i<T3.channel_n();i++)
      {
         auto i_3=T3.channel(i);
         auto i_2=T2.channel(i);
         auto i_1=T1.channel(i);
         ASSERT_EQ(arma::approx_equal(i_3,i_1/i_2,"absdiff",1e-5),true);
      }

   }
}

TEST(operator_Test,add_scalar)
{
   using namespace MINI_MLsys;
   int cnt=10;
   srand(time(0));
   while(cnt--)
   {
      int row=abs(rand()%10)+1;
      int col=abs(rand()%10)+1;
      int cha=abs(rand()%3)+1;
      std::cout<<"shape: "<<row<<" "<<col<<" "<<cha<<std::endl;
      Tensor<double> T(row,col,cha);
      T.randi(1,4);
      int scalar=rand()%10;
      Tensor<double>T2=T+scalar;
      for(size_t i=0;i<T.channel_n();i++){
         auto i_=T.channel(i);
         auto i_1=T2.channel(i);
         ASSERT_EQ(arma::approx_equal(i_1,i_+scalar,"absdiff",1e-5),true);
      }
      std::cout<<T<<std::endl;
      std::cout<<"scalar:"<<scalar<<std::endl;
      std::cout<<T2<<std::endl;
      // system("pause");
      std::cout<<"---------------------------------------------"<<std::endl;
   }
}

TEST(operator_Test,sub_scalar)
{
   using namespace MINI_MLsys;
   int cnt=10;
   srand(time(0));
   while(cnt--)
   {
      int row=abs(rand()%10)+1;
      int col=abs(rand()%10)+1;
      int cha=abs(rand()%3)+1;
      std::cout<<"shape: "<<row<<" "<<col<<" "<<cha<<std::endl;
      Tensor<double> T(row,col,cha);
      T.randi(1,4);
      int scalar=rand()%10;
      Tensor<double>T2=T-scalar;
      for(size_t i=0;i<T.channel_n();i++){
         auto i_=T.channel(i);
         auto i_1=T2.channel(i);
         ASSERT_EQ(arma::approx_equal(i_1,i_-scalar,"absdiff",1e-5),true);
      }
      std::cout<<T<<std::endl;
      std::cout<<"scalar:"<<scalar<<std::endl;
      std::cout<<T2<<std::endl;
      // system("pause");
      std::cout<<"---------------------------------------------"<<std::endl;
   }
}

TEST(operator_Test,div_scalar)
{
   using namespace MINI_MLsys;
   int cnt=10;
   srand(time(0));
   while(cnt--)
   {
      int row=abs(rand()%10)+1;
      int col=abs(rand()%10)+1;
      int cha=abs(rand()%3)+1;
      std::cout<<"shape: "<<row<<" "<<col<<" "<<cha<<std::endl;
      Tensor<double> T(row,col,cha);
      T.randi(1,4);
      int scalar=rand()%10+1;
      Tensor<double>T2=T/scalar;
      for(size_t i=0;i<T.channel_n();i++){
         auto i_=T.channel(i);
         auto i_1=T2.channel(i);
         ASSERT_EQ(arma::approx_equal(i_1,i_/scalar,"absdiff",1e-5),true);
      }
      std::cout<<T<<std::endl;
      std::cout<<"scalar:"<<scalar<<std::endl;
      std::cout<<T2<<std::endl;
      // system("pause");
      std::cout<<"---------------------------------------------"<<std::endl;
   }
}

TEST(operator_Test,multy_scalar)
{
   using namespace MINI_MLsys;
   int cnt=10;
   srand(time(0));
   while(cnt--)
   {
      int row=abs(rand()%10)+1;
      int col=abs(rand()%10)+1;
      int cha=abs(rand()%3)+1;
      std::cout<<"shape: "<<row<<" "<<col<<" "<<cha<<std::endl;
      Tensor<double> T(row,col,cha);
      T.randi(1,4);
      int scalar=rand()%10;
      Tensor<double>T2=T*scalar;
      for(size_t i=0;i<T.channel_n();i++){
         auto i_=T.channel(i);
         auto i_1=T2.channel(i);
         ASSERT_EQ(arma::approx_equal(i_1,i_*scalar,"absdiff",1e-5),true);
      }
      std::cout<<T<<std::endl;
      std::cout<<"scalar:"<<scalar<<std::endl;
      std::cout<<T2<<std::endl;
      // system("pause");
      std::cout<<"---------------------------------------------"<<std::endl;
   }
}



