#include"Tensor.hpp"

namespace MINI_MLsys{

  template<typename T>
  Tensor<T>::Tensor(uint64_t size)
  {
    CHECK(size>0)
    this->data_.reshape(1,size,1);
    this->data_.zeros();
    this->shape={1,size,1};
  }

  template <typename T>
  Tensor<T>::Tensor(uint32_t row, uint32_t column)
  {
    CHECK(row>0&&column>0)
    this->data_.reshape(row,column);
    this->data_.zeros();
    this->shape={row,column,1};
  }

  template <typename T>
  Tensor<T>::Tensor(const uint32_t &row, const uint32_t &column, const uint32_t &channel)
  {
    CHECK(row>0&&column>0&&channel>0)
    this->data_.reshape(row,column,channel);
    this->data_.zeros();
    this->shape={row,column,channel};
  }

  template <typename T>
  Tensor<T>::Tensor(const std::vector<uint32_t> &shape)
  {
    CHECK(shape.size()>0&&shape.size()<4);
    if(shape.size()==1)
    {
      this->data_.reshape(1,shape[0],1);
      this->data_.zeros();
      this->shape={1,shape[0],1};
    }
    else if (shape.size()==2)
    {
      this->data_.reshape(shape[0],shape[1],1);
      this->data_.zeros();
      this->shape={shape[0],shape[1],1};
    }
    else
    {
      this->data_.reshape(shape[0],shape[1],shape[2]);
      this->data_.zeros();
      this->shape={shape[0],shape[1],shape[2]};
    }
  }
  template <typename T>
  Tensor<T>::Tensor(const Tensor &tensor)
  {
    this->shape=tensor.shape();
    this->data_.reshape(this->shape[0],this->shape[1],this->shape[2]);
    this->data_.zeros();
  }

  template <typename T>
  Tensor<T>::Tensor(const arma::Cube<T> tensor)
  {
    auto row = tensor.n_rows;
    auto col = tensor.n_cols;
    auto cha = tensor.n_slices;
    this->data_=tensor;
    this->shape={row,col,cha};
  }

  template <typename T>
  uint64_t Tensor<T>::row_n() const
  {
    return this->shape[0];
  }

  template <typename T>
  uint64_t Tensor<T>::col_n() const
  {
    return this->shape[1];
  }

  template <typename T>
  uint64_t Tensor<T>::channel_n() const
  {
    return this->shape[2];
  }

  template <typename T>
  uint64_t Tensor<T>::size() const
  {
    return this->shape[0]*this->shape[1]*this->shape[2];
  }



}