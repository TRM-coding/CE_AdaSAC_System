# 文件目录说明
* ./exp* 对应每次实验的源代码
* ./ops为提取的算子和实现

# 使用方法

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

# 附：算子提取方法：
llamacpp/test/test-backend-ops.cpp中给出了各个算子的调用示例，必须要写buildgraph