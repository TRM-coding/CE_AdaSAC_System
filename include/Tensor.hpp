#pragma once

#include <armadillo>
#include <functional>
#include <gtest/gtest.h>
#include <vector>
namespace MINI_MLsys {
template <typename T>
/**
 * Tensor constructors are filled with default value 0
 */
class Tensor {
public:
  explicit Tensor() = default;
  Tensor(const Tensor &) = default;

  /**
   *Construct a 1-dim Tensor by {size}
   *@param size total elements number in the new tensor
   */
  explicit Tensor(int size) {
    assert(size > 0);
    this->data_.reshape(1, size, 1);
    this->data_.zeros();
    this->shape_ = {1, size, 1};
  }

  /**
   * Construct a 2-dim Tensor by {row,column}
   * @param row    the number of rows in the new tensor
   * @param column the number of column in the new tensor
   */
  explicit Tensor(int row, int column) {
    assert(row > 0 && column > 0);
    this->data_.reshape(row, column);
    this->data_.zeros();
    this->shape_ = {row, column, 1};
  }

  /**
   *Construct a 3-dim Tensor by {row,column,channel}
   *@param row     the number of rows in the new tensor
   *@param column  the number of column in the new tensor
   *@param channel the number of channel in the new tensor
   **/
  explicit Tensor(const uint64_t &row, const uint64_t &column,
                  const uint64_t &channel) {
    assert(row > 0 && column > 0 && channel > 0);
    this->data_.reshape(row, column, channel);
    this->data_.zeros();
    this->shape_ = {row, column, channel};
  }

  /**
   *construct a Tensor by shape
   std::vector<uint32_t>{dim1(row),dim2(column),dim3(channel)}
   @param shape std::vector if 1 elements then construct a 1-dim tensor.The same
   apply to 2,3
  */
  explicit Tensor(const std::vector<uint64_t> &shape) {
    assert(shape.size() > 0 && shape.size() < 4);
    if (shape.size() == 1) {
      this->data_.reshape(1, shape[0], 1);
      this->data_.zeros();
      this->shape_ = {1, shape[0], 1};
    } else if (shape.size() == 2) {
      this->data_.reshape(shape[0], shape[1], 1);
      this->data_.zeros();
      this->shape_ = {shape[0], shape[1], 1};
    } else {
      this->data_.reshape(shape[0], shape[1], shape[2]);
      this->data_.zeros();
      this->shape_ = {shape[0], shape[1], shape[2]};
    }
  }

  explicit Tensor(const std::vector<uint64_t>& shape,const std::vector<T>& data_)
  {
    
    assert(shape.size() > 0 && shape.size() < 4);
    if (shape.size() == 1) {
      this->data_.reshape(1, shape[0], 1);
      this->data_.zeros();
      this->shape_ = {1, shape[0], 1};
    } else if (shape.size() == 2) {
      this->data_.reshape(shape[0], shape[1], 1);
      this->data_.zeros();
      this->shape_ = {shape[0], shape[1], 1};
    } else {
      this->data_.reshape(shape[0], shape[1], shape[2]);
      this->data_.zeros();
      this->shape_ = {shape[0], shape[1], shape[2]};
    }

    if(data_.size()==0)
    {
      this->data_.zeros();
    }
    if(data_.size()!=shape[0]*shape[1]*shape[2])
    {
      std::cout<<"data size not match shape size"<<std::endl;
      assert(1==0);
    }

    for (uint32_t i = 0; i < shape[0]; i++) {
      for (uint32_t j = 0; j < shape[1]; j++) {
        for (uint32_t k = 0; k < shape[2]; k++) {
          this->data_.at(i, j, k) = data_[i * shape[1] * shape[2] + j * shape[2] + k];
        }
      }
    }
  }

  /**
   * construct a new Tensor by an existed Tensor,it will return a same shape
   * tensor but empty one
   */

  // Tensor(const Tensor tensor);

  /**
   * construct a new Tensor by an arma::Cube<T>
   * @param tensor a T-type Cube<T> tensor
   */

  explicit Tensor(const arma::Cube<T> tensor) {
    auto row = tensor.n_rows;
    auto col = tensor.n_cols;
    auto cha = tensor.n_slices;
    this->data_ = tensor;
    this->shape_ = {row, col, cha};
  }

  /**
   * return the Tensor's row number
   */
  uint64_t row_n() const { return this->shape_[0]; }

  /**
   * return the Tensor's column number
   */

  uint64_t col_n() const { return this->shape_[1]; }

  /**
   * return the Tensor's channel number
   */
  uint64_t channel_n() const { return this->shape_[2]; }

  /**
   * return the Tensor's shape in
   * format:std::vector<uint32_t>{row,column,channel}
   */

  std::vector<uint64_t> get_shape() const { return this->shape_; }

  arma::Mat<T> channel(size_t i) const { return this->data_.slice(i); }


  arma::Cube<T> get_data() const { return this->data_; }

  /**
   * return the Tensor's total elements number
   */
  int64_t size() const {
    return this->shape_[0] * this->shape_[1] * this->shape_[2];
  }

  /**
   * modify the tensor's shape and add supplement numbers
   */

  // void reshape();

  Tensor<T> operator*(const Tensor<T> &B) const {
    assert(this->channel_n() == B.channel_n());
    assert(this->col_n() == B.row_n());
    const uint32_t new_row = this->row_n();
    const uint32_t new_col = B.col_n();
    const uint32_t channel_n = B.channel_n();
    arma::Cube<T> new_tensor_data = arma::Cube<T>(new_row, new_col, channel_n);
    for (uint32_t i = 0; i < channel_n; i++) {
      new_tensor_data.slice(i) = this->data_.slice(i) * B.data_.slice(i);
    }
    Tensor<T> new_tensor=Tensor<T>(new_tensor_data);
    return new_tensor;
  }

  Tensor<T> operator*(const T &B) const {
    arma::Cube<T> new_tensor_data = this->data_ * B;
    Tensor<T> new_tensor(new_tensor_data);
    return new_tensor;
  }

  Tensor<T> operator+(const Tensor<T> &B) const {
    assert(this->channel_n() == B.channel_n());
    assert(this->col_n() == B.col_n());
    assert(this->row_n() == B.row_n());
    const uint32_t new_row = this->row_n();
    const uint32_t new_col = this->row_n();
    const uint32_t new_cha = this->channel_n();
    arma::Cube<T> new_tensor_data = arma::Cube<T>(new_row, new_col, new_cha);
    new_tensor_data = this->data_ + B.data_;
    Tensor<T> new_tensor(new_tensor_data);
    return new_tensor;
  }

  Tensor<T> operator+(const T &B) const {
    arma::Cube<T> new_tensor_data = this->data_ + B;
    Tensor<T> new_tensor(new_tensor_data);
    return new_tensor;
  }

  Tensor<T> operator-(const Tensor<T> &B) const {
    assert(this->channel_n() == B.channel_n());
    assert(this->col_n() == B.col_n());
    assert(this->row_n() == B.row_n());
    const uint32_t new_row = this->row_n();
    const uint32_t new_col = this->row_n();
    const uint32_t new_cha = this->channel_n();
    arma::Cube<T> new_tensor_data = arma::Cube<T>(new_row, new_col, channel_n);
    new_tensor_data = this->data_ - B.data_;
    Tensor<T> new_tensor(new_tensor_data);
    return new_tensor;
  }

  Tensor<T> operator-(const T &B) const {
    arma::Cube<T> new_tensor_data = this->data_ - B;
    Tensor<T> new_tensor(new_tensor_data);
    return new_tensor;
  }

  Tensor<T> operator/(const Tensor<T> &B) const {
    assert(this->channel_n() == B.channel_n());
    assert(this->col_n() == B.col_n());
    assert(this->row_n() == B.row_n());
    arma::Cube<T> new_tensor_data = this->data_ / B.data_;
    Tensor<T> new_tensor(new_tensor_data);
    return new_tensor;
  }

  Tensor<T> operator/(const T &B) const {
    assert(B != 0);
    arma::Cube<T> new_tensor_data = this->data_ / B;
    Tensor<T> new_tensor(new_tensor_data);
    return new_tensor;
  }

  arma::Mat<T> &operator[](int i) { return this->data_.slice(i); }
  T& at(int channel, int row, int col) { 
    std::cout<<"shape::"<<this->shape_.size()<<std::endl;
    // assert(1==0);
    assert(channel < this->shape_[2] && row < this->shape_[0] && col < this->shape_[1]);
    return this->data_.at(row, col, channel); 
  }

  void randu() { this->data_.randu(); }
  void randn() { this->data_.randn(); }
  void randi(int l, int r) {
    this->data_ =
        arma::randi<arma::Cube<T>>(this->shape_[0], this->shape_[1],
                                   this->shape_[2], arma::distr_param(l, r));
  }

  typedef T (*F)(const T &);

  std::shared_ptr<Tensor<T>> func(F funci) {
    std::shared_ptr<Tensor<T>> tensor = std::make_shared<Tensor<T>>(this->shape_);
    for (auto i = 0; i < data_.n_rows; i++) {
      for (auto j = 0; j < data_.n_cols; j++) {
        for (auto k = 0; k < data_.n_slices; k++) {
          // tensor.at(k,i,j) = funci(this->data_.at(i, j, k));
          
          tensor->at(k,i,j)=funci(this->data_.at(i, j, k));
        }
      }
    }
    return tensor;
  }

  /**
   * tensor Schur product
   */
  // Tensor<T> operator%(const Tensor<T> &A, const Tensor<T> &B) const;
  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

  template <typename U> arma::Cube<U> &operator[](int i) {
    return this->data_.slice(i);
  }

private:
  arma::Cube<T> data_;
  std ::vector<uint64_t> shape_;
  // shape{row,col,cha}
};

template <typename U>
inline std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor) {
  os << "The Tensor is:\n";
  for (uint32_t i_slice = 0; i_slice < tensor.data_.n_slices; i_slice++) {
    for (uint32_t i_row = 0; i_row < tensor.data_.n_rows; i_row++) {
      for (uint32_t i_col = 0; i_col < tensor.data_.n_cols; i_col++) {
        auto x_i = tensor.data_.at(i_row, i_col, i_slice);
        os << x_i << " ";
      }
      os << std::endl;
    }
    os << std::endl;
  }
  return os;
}

} // namespace MINI_MLsys
