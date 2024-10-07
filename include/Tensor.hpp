#ifndef Tensor_hpp
#define Tensor_hpp

#include <armadillo>
#include <vector>
#include <gtest/gtest.h>
namespace MINI_MLsys
{
  template <typename T>
  /**
   * Tensor constructors are filled with default value 0
   */
  class Tensor
  {
  public:
    explicit Tensor() = default;
    Tensor(const Tensor &) = default;

    /**
     *Construct a 1-dim Tensor by {size}
     *@param size total elements number in the new tensor
     */
    explicit Tensor(int64_t size)
    {
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
    explicit Tensor(uint32_t row, uint32_t column)
    {
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
    explicit Tensor(const uint32_t &row, const uint32_t &column, const uint32_t &channel)
    {
      assert(row > 0 && column > 0 && channel > 0);
      this->data_.reshape(row, column, channel);
      this->data_.zeros();
      this->shape_ = {row, column, channel};
    }

    /**
     *construct a Tensor by shape std::vector<uint32_t>{dim1(row),dim2(column),dim3(channel)}
     @param shape std::vector if 1 elements then construct a 1-dim tensor.The same apply to 2,3
    */
    explicit Tensor(const std::vector<uint32_t> &shape)
    {
      assert(shape.size() > 0 && shape.size() < 4);
      if (shape.size() == 1)
      {
        this->data_.reshape(1, shape[0], 1);
        this->data_.zeros();
        this->shape_ = {1, shape[0], 1};
      }
      else if (shape.size() == 2)
      {
        this->data_.reshape(shape[0], shape[1], 1);
        this->data_.zeros();
        this->shape_ = {shape[0], shape[1], 1};
      }
      else
      {
        this->data_.reshape(shape[0], shape[1], shape[2]);
        this->data_.zeros();
        this->shape_ = {shape[0], shape[1], shape[2]};
      }
    }

    explicit Tensor(const std::vector<int>& shape)
    {
      assert(shape.size() > 0 && shape.size() < 4);
      if (shape.size() == 1)
      {
        this->data_.reshape(1, shape[0], 1);
        this->data_.zeros();
        this->shape_ = {1, shape[0], 1};
      }
      else if (shape.size() == 2)
      {
        this->data_.reshape(shape[0], shape[1], 1);
        this->data_.zeros();
        this->shape_ = {shape[0], shape[1], 1};
      }
      else
      {
        this->data_.reshape(shape[0], shape[1], shape[2]);
        this->data_.zeros();
        this->shape_ = {shape[0], shape[1], shape[2]};
      }
    }

    /**
     * construct a new Tensor by an existed Tensor,it will return a same shape tensor but empty one
     */

    // Tensor(const Tensor tensor);

    /**
     * construct a new Tensor by an arma::Cube<T>
     * @param tensor a T-type Cube<T> tensor
     */

    explicit Tensor(const arma::Cube<T> tensor)
    {
      auto row = tensor.n_rows;
      auto col = tensor.n_cols;
      auto cha = tensor.n_slices;
      this->data_ = tensor;
      this->shape_ = {row, col, cha};
    }

    /**
     * return the Tensor's row number
     */
    uint64_t row_n() const
    {
      return this->shape_[0];
    }

    /**
     * return the Tensor's column number
     */

    uint64_t col_n() const
    {
      return this->shape_[1];
    }

    /**
     * return the Tensor's channel number
     */
    uint64_t channel_n() const
    {
      return this->shape_[2];
    }

    /**
     * return the Tensor's shape in format:std::vector<uint32_t>{row,column,channel}
     */

    std::vector<int64_t> get_shape() const
    {
      return this->shape_;
    }

    arma::Mat<T> channel(size_t i) const
    {
      return this->data_.slice(i);
    }

    /**
     * return the Tensor's total elements number
     */
    int64_t size() const
    {
      return this->shape_[0] * this->shape_[1] * this->shape_[2];
    }

    /**
     * modify the tensor's shape and add supplement numbers
     */

    // void reshape();

    Tensor<T> operator*(const Tensor<T> &B) const
    {
      assert(this->channel_n() == B.channel_n());
      assert(this->col_n() == B.row_n());
      const uint32_t new_row = this->row_n();
      const uint32_t new_col = B.col_n();
      const uint32_t channel_n = B.channel_n();
      arma::Cube<T> new_tensor_data = arma::Cube<T>(new_row, new_col, channel_n);
      for (uint32_t i = 0; i < channel_n; i++)
      {
        new_tensor_data.slice(i) = this->data_.slice(i) * B.data_.slice(i);
      }
      Tensor<T> new_tensor(new_tensor_data);
      return new_tensor;
    }

    Tensor<T> operator*(const T &B) const
    {
      arma::Cube<T> new_tensor_data = this->data_ * B;
      Tensor<T> new_tensor(new_tensor_data);
      return new_tensor;
    }

    Tensor<T> operator+(const Tensor<T> &B) const
    {
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

    Tensor<T> operator+(const T &B) const
    {
      arma::Cube<T> new_tensor_data = this->data_ + B;
      Tensor<T> new_tensor(new_tensor_data);
      return new_tensor;
    }

    Tensor<T> operator-(const Tensor<T> &B) const
    {
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

    Tensor<T> operator-(const T &B) const
    {
      arma::Cube<T> new_tensor_data = this->data_ - B;
      Tensor<T> new_tensor(new_tensor_data);
      return new_tensor;
    }

    Tensor<T> operator/(const Tensor<T> &B) const
    {
      assert(this->channel_n() == B.channel_n());
      assert(this->col_n() == B.col_n());
      assert(this->row_n() == B.row_n());
      arma::Cube<T> new_tensor_data = this->data_ / B.data_;
      Tensor<T> new_tensor(new_tensor_data);
      return new_tensor;
    }

    Tensor<T> operator/(const T &B) const
    {
      assert(B != 0);
      arma::Cube<T> new_tensor_data = this->data_ / B;
      Tensor<T> new_tensor(new_tensor_data);
      return new_tensor;
    }

    void randu()
    {
      this->data_.randu();
    }
    void randn()
    {
      this->data_.randn();
    }
    void randi(int l, int r)
    {
      this->data_ = arma::randi<arma::Cube<T>>(this->shape_[0], this->shape_[1], this->shape_[2], arma::distr_param(l, r));
    }

    /**
     * tensor Schur product
     */
    // Tensor<T> operator%(const Tensor<T> &A, const Tensor<T> &B) const;
    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

  private:
    arma::Cube<T> data_;
    std ::vector<int64_t> shape_;
    // shape{row,col,cha}
  };

  template <typename U>
  std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor)
  {
    os << "The Tensor is:\n";
    for (uint32_t i_slice = 0; i_slice < tensor.data_.n_slices; i_slice++)
    {
      for (uint32_t i_row = 0; i_row < tensor.data_.n_rows; i_row++)
      {
        for (uint32_t i_col = 0; i_col < tensor.data_.n_cols; i_col++)
        {
          auto x_i = tensor.data_.at(i_row, i_col, i_slice);
          os << x_i << " ";
        }
        os << std::endl;
      }
      os << std::endl;
    }
    return os;
  }
}//namespace MINI_MLsys

#endif