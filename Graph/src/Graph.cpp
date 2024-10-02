#include <Graph.hpp>
#include <ir.h>
#include <iostream>

MINI_MLsys::Graph::
    Graph(std::string param_path, std::string bin_path)
{
   this->param_path_ = param_path;
   this->bin_path_ = bin_path;
   this->graph_ = std::make_unique<pnnx::Graph>();
}

bool MINI_MLsys::Graph::
    init()
{
   if (this->param_path_.empty() || this->bin_path_.empty())
   {
      std::cout << "The bin path or param path is empty" << std::endl;
      return false;
   }
   auto load_status = this->graph_->load(this->param_path_, this->bin_path_);
   if (!load_status)
   {
      std::cout << "Can not find the param path or bin path: " << param_path_
                << " " << bin_path_;
      return false;
   }

   auto pnnx_operators = this->graph_->ops;
   if (pnnx_operators.empty())
   {
      std::cout << "Can not read the layers' define";
      return false;
   }

   return true;
}