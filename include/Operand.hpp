#include <string>
#include <vector>
#include <memory>
#include <Tensor.hpp>
#include <ir.h>
namespace MINI_MLsys
{
    class Operand
    {
    public:
        int type;//
        std::vector<int> shape;//
        std::string name;//
        std::map<std::string, pnnx::Parameter> params;
        std::shared_ptr<Tensor<float>> data;
        Operand(pnnx::Operand* pnn_op);
    };
}//namespace MINI_MLsys