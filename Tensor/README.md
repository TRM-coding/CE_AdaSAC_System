# Tensor-document-ZH

### Tensor 类描述

* `Tensor`类为`MINI-MLsys`中的张量类，支持维度：`1-dim、2-dim、3-dim`
* Tensor类通过`Armadillo.cube`存储数据
* Tensor类提供私有成员变量`std::vector<uint32_t>shape`存储张量格式



### Tensor 类功能及函数

#### 构造器

- [ ] 提供 **元素个数**，构建**一维张量**
- [ ] 提供 **行、列数**，构建 **二维张量**
- [ ] 提供 **行、列、通道数**，构建 **三维张量**
- [ ] 提供 **vector存储的行列通道数**，构建 **对应形状的张量**
- [ ] 提供 **Tensor类**，构建一个 **相同形状，但是空的张量**
- [ ] 提供 **cube**类，返回一个 **同形状，同数据**的张量