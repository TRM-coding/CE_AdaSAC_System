#ifndef GRAPH_HPP
#define GRAPH_HPP

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

      int init();
      std::vector<std::shared_ptr<Operator>> Topo(std::shared_ptr<MINI_MLsys::Operator> root);
      std::shared_ptr<Operator> build();
      std::string get_param_path();
      std::string get_bin_path();

    // private:
      std::string bin_path_;//
      std::string param_path_;//

      std::shared_ptr<MINI_MLsys::Operator> root;
      std::vector<std::shared_ptr<Operator>> operators_;
      std::vector<std::shared_ptr<Operator>> topo_operators_;
      std::unique_ptr<pnnx::Graph> graph_;//
      std::map<std::string,std::shared_ptr<Operator>> operator_map_;
  };
}//namespace MINI_MLsys

#endif