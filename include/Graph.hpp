#include<string>
#include<vector>
#include<memory>
#include<Operator.hpp>
#include<ir.h>
namespace MINI_MLsys
{
  class Graph{
    public:
      Graph(std::string param_path,std::string bin_path);

      bool init();

    private:
      std::string input_name_;
      std::string output_name_;
      std::string param_name_;
      std::string bin_path_;
      std::string param_path_;

      std::vector<std::shared_ptr<Operator>> operators_;
      std::unique_ptr<pnnx::Graph> graph_;
  };
}//namespace MINI_MLsys