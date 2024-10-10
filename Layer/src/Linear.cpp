#include <Layer/Linear.hpp>
namespace MINI_MLsys
{
    void Linear::forward(const std::vector<std::shared_ptr<Tensor<float>>> &input,
                         std::vector<std::shared_ptr<Tensor<float>>> &output) {
                            return;
                         }
    

    bool Linear::deploy(const std::shared_ptr<Operator> &op)
    {
        if(op==nullptr)
        {
            std::cout<<"Linear: Operator is nullptr."<<std::endl;
            return false;
        }
        op->layer=std::make_shared<Linear>("Linear");
        op->layer->op_=op;
        auto layer=std::dynamic_pointer_cast<Linear>(op->layer);
        layer->set_bias();
        layer->set_weight();
        return true;
    }

    void Linear::set_bias()
    {
        
        return;
    }

    void Linear::set_weight()
    {
        return;
    }
}