#ifndef Tensor_hpp
#define Tensor_hpp

#include<armadillo>
#include<vector>
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

      /** 
       *Construct a 1-dim Tensor by {size}
       *@param size total elements number in the new tensor
      */
      explicit Tensor(uint64_t size);


      /**
       * Construct a 2-dim Tensor by {row,column}
       * @param row    the number of rows in the new tensor
       * @param column the number of column in the new tensor
      */
      explicit Tensor(uint32_t row,uint32_t column);

      /** 
       *Construct a 3-dim Tensor by {row,column,channel}
       *@param row     the number of rows in the new tensor
       *@param column  the number of column in the new tensor
       *@param channel the number of channel in the new tensor
      **/
      explicit Tensor(const uint32_t& row,const uint32_t& column,const uint32_t& channel);

      /**
       *construct a Tensor by shape std::vector<uint32_t>{dim1(row),dim2(column),dim3(channel)} 
       @param shape std::vector if 1 elements then construct a 1-dim tensor.The same apply to 2,3
      */
      explicit Tensor(const std::vector<uint32_t>& shape);

      /**
       * construct a new Tensor by an existed Tensor,it will return a same shape tensor but empty one
      */
      explicit Tensor(const Tensor& tensor);

      /**
       * construct a new Tensor by an arma::cube
       * @param tensor a T-type cube tensor
       */

      explicit Tensor (const arma::Cube<T> tensor);

      /**
       * return the Tensor's row number
       */
      uint64_t row_n() const;

      /**
       * return the Tensor's column number
       */

      uint64_t col_n() const;

      /**
       * return the Tensor's channel number
       */
      uint64_t channel_n() const;

      /**
       * return the Tensor's shape in format:std::vector<uint32_t>{row,column,channel}
       */

      std::vector<uint32_t> shape() const;

      /**
       * return the Tensor's total elements number 
       */
      uint64_t size() const;

      /**
       * return whether the Tensor is empty
       */

      void reshape();

    private:
      arma:: cube data_;
      std :: vector<uint64_t> shape;
      //shape{row,col,cha}
  };
}

#endif