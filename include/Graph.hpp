#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <Layer/LayerRegister.hpp>
#include <Operand.hpp>
#include <Operator.hpp>
#include <ir.h>
#include <memory>
#include <string>
#include <vector>
namespace MINI_MLsys {
class Graph {
public:
  Graph(std::string param_path, std::string bin_path);
  std::vector<Tensor<float>> RUN(const std::vector<Tensor<float>> &inputs);
  int init();
  std::vector<std::shared_ptr<Operator>>
  Topo(std::shared_ptr<MINI_MLsys::Operator> root);
  std::shared_ptr<Operator> build();
  std::string get_param_path();
  std::string get_bin_path();
  bool initOperator_param();
  bool initOperator_attr();
  bool deploy_layers();

  // private:
  std::string bin_path_;   //
  std::string param_path_; //

  std::shared_ptr<MINI_MLsys::Operator> root;
  std::vector<std::shared_ptr<Operator>> operators_;
  std::vector<std::shared_ptr<Operator>> topo_operators_;
  std::unique_ptr<pnnx::Graph> graph_; //
  std::map<std::string, std::shared_ptr<Operator>> operator_map_;
};
} // namespace MINI_MLsys

#endif