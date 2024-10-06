#include <Graph.hpp>
#include <ir.h>
#include <iostream>
#include <Operator.hpp>
#include <vector>
#include <memory>
#include<queue>



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

   this->root = build();
   Topo(this->root);

   return true;
}

std::vector<std::shared_ptr<MINI_MLsys::Operator>> MINI_MLsys::Graph::
    Topo(std::shared_ptr<MINI_MLsys::Operator> root)
{
   if(!root)
   {
      std::cout<<"EMPTY ROOT WHEN BUILDING TOPO!!!"<<std::endl;
      return std::vector<std::shared_ptr<Operator>>();

   }
   std::vector<std::shared_ptr<Operator>> res;
   std::queue<std::shared_ptr<Operator>>Q;
   Q.push(root);
   while(!Q.empty())
   {
      auto front = Q.front();
      Q.pop();
      res.push_back(front);
      for(const auto net :front->next_op)Q.push(net);
   }
   return res;
}

std::shared_ptr<MINI_MLsys::Operator>MINI_MLsys::Graph::
    build()
{
   // MINI_MLsys::Operator *root = new MINI_MLsys::Operator();
   std::shared_ptr<Operator>root=std::make_shared<Operator>();

   auto &ops = this->graph_->ops;
   for (auto op : ops)
   {
      if (!op)
         continue;

      auto type = op->type;
      auto name = op->name;

      std::shared_ptr<Operator> this_op;
      if (this->operator_map_.find(name) != this->operator_map_.end())
      {
         this_op = this->operator_map_[name];
      }
      else
      {
         this_op = std::make_shared<Operator>(op);
         this->operator_map_[name]=this_op;
      }

      auto input_opds = op->inputs;
      for (auto inputi : input_opds)
      {
         auto producer = inputi->producer;
         if (this->operator_map_.find(producer->name) != this->operator_map_.end())
         {
            this_op->pre_op.push_back(this->operator_map_[producer->name]);
         }
         else
         {
            std::shared_ptr<Operator> pre = std::make_shared<Operator>(producer);
            this->operator_map_[producer->name] = pre;
            this_op->pre_op.push_back(pre);
         }
      }


      auto &output_opds = op->outputs;
      for (auto output_e : output_opds)
      {
         for (auto net_nd : output_e->consumers)
         {
            if (this->operator_map_.find(net_nd->name) != this->operator_map_.end())
            {
               this_op->next_op.push_back(this->operator_map_[net_nd->name]);
            }
            else
            {
               std::shared_ptr<Operator> net = std::make_shared<Operator>(net_nd);
               this_op->next_op.push_back(net);
               this->operator_map_[net_nd->name] = net;
            }
         }
      }
   }

   for(const auto& [name,op]:this->operator_map_)
   {
      if(op->pre_op.size()==0)
      {
         op->pre_op.push_back(root);
         root->next_op.push_back(op);
      }
   }

   return root;
}

