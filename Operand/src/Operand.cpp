#include "Operand.hpp"

MINI_MLsys::Operand::Operand(pnnx::Operand* pnn_op)
{
    this->shape=pnn_op->shape;
    this->type=pnn_op->type;
    this->name=pnn_op->name;
    this->params=pnn_op->params;
}