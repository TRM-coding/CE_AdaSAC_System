#include<armadillo>
int main()
{
  arma::Cube<float> a(2,2,2);
  // a.randu();
  std::cout<<a<<std::endl;
  auto& b =a.slice(0);
  b.randn();
  std::cout<<a<<std::endl;
  arma::mat k(2,2);
  a.randn();
  std::cout<<k<<std::endl;
  std::cout<<k[0]<<std::endl;

  return 0;
}