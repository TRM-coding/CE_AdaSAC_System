#include <Graph.hpp>
#include <ir.h>
#include <iostream>
#include <Operator.hpp>
#include <vector>
#include <memory>
#include <queue>
#include <set>

MINI_MLsys::Graph::
    Graph(std::string param_path, std::string bin_path)
{
   this->param_path_ = param_path;
   this->bin_path_ = bin_path;
   this->graph_ = std::make_unique<pnnx::Graph>();
   auto load_status = this->graph_->load(this->param_path_, this->bin_path_);
   if (load_status != 0)
   {
      std::cout << "Can not find the param path or bin path: " << param_path_
                << " " << bin_path_;
      throw std::runtime_error("Can not find the param path or bin path");
   }
}

int MINI_MLsys::Graph::
    init()
{
   if (this->param_path_.empty() || this->bin_path_.empty())
   {
      std::cout << "The bin path or param path is empty" << std::endl;
      return -1;
   }

   auto pnnx_operators = this->graph_->ops;
   if (pnnx_operators.empty())
   {
      std::cout << "Can not read the layers' define";
      return -1;
   }
   if (!this->root)
      this->root = build();
   this->topo_operators_ = Topo(this->root);

   return 0;
}

std::vector<std::shared_ptr<MINI_MLsys::Operator>> MINI_MLsys::Graph::
    Topo(std::shared_ptr<MINI_MLsys::Operator> root)
{
   if (!root)
   {
      std::cout << "EMPTY ROOT WHEN BUILDING TOPO!!!" << std::endl;
      return std::vector<std::shared_ptr<Operator>>();
   }
   std::vector<std::shared_ptr<Operator>> res;
   std::queue<std::shared_ptr<Operator>> Q;
   Q.push(root);
   int cnt = 0;
   std::cout << this->operator_map_.size() << std::endl;
   std::set<std::string> visited;
   visited.insert(root->name);

   while (!Q.empty())
   {
      cnt++;
      // std::cout<<"Q size: "<<Q.size()<<std::endl;
      auto front = Q.front();
      Q.pop();
      res.push_back(front);
      // std::cout<<"front:"<<front->name<<" front_next_size:"<<front->next_op.size()<<std::endl;
      for (const auto net : front->next_op)
      {
         if (visited.find(net->name) == visited.end())
         {
            visited.insert(net->name);
            Q.push(net);
         }
      }
   }
   std::cout << cnt << std::endl;
   res.erase(res.begin());
   return res;
}

std::shared_ptr<MINI_MLsys::Operator> MINI_MLsys::Graph::
    build()
{
   // MINI_MLsys::Operator *root = new MINI_MLsys::Operator();
   std::shared_ptr<Operator> root = std::make_shared<Operator>();
   root->name = "root";

   auto &ops = this->graph_->ops;

   int cnt = 0;
   for (const auto &op : ops)
   {

      cnt++;
      std::cout << cnt << ":" << op->name << std::endl;
   }
   std::cout << std::endl
             << std::endl;
   std::cout << "pnnx_size::" << ops.size() << std::endl;
   std::cout << "cnt:" << cnt << std::endl;

   for (const auto &op : ops)
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
         this->operator_map_[name] = this_op;
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

   for (const auto &[name, op] : this->operator_map_)
   {
      if (op->pre_op.size() == 0)
      {
         op->pre_op.push_back(root);
         root->next_op.push_back(op);
      }
      if (op->next_op.size() == 0)
      {
         std::cout << "Operator: " << op->name << " has no next operator" << std::endl
                   << std::endl;
      }
   }

   std::cout << this->operator_map_.size() << std::endl;

   return root;
}

std::string MINI_MLsys::Graph::get_param_path()
{
   return this->param_path_;
}

std::string MINI_MLsys::Graph::get_bin_path()
{
   return this->bin_path_;
}
