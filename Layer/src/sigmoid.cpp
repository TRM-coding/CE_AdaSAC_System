#include<Layer/sigmoid.hpp>

namespace MINI_MLsys {

    void Sigmoid::forward(const std::vector<std::shared_ptr<Tensor<float>>> &input,
                          std::vector<std::shared_ptr<Tensor<float>>> &output) {
      for (const auto &x : input) {
        auto out = x->func(sigmoid);
        output.push_back(std::make_shared<Tensor<float>>(out));
      }
    }

    bool Sigmoid::deploy(const std::shared_ptr<Operator>& op)
    {
        if(op==nullptr)
        {
            std::cout<<"Sigmoid: Operator is nullptr."<<std::endl;
            return false;
        }
        op->layer=std::make_shared<Sigmoid>("Sigmoid");
        op->layer->op_=op;
        // no params and attrs
        return true;
    }

}// namespace MINI_MLsys