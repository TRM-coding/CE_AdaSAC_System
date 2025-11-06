# 文件目录说明
* ./exp* 对应每次实验的源代码
* ./ops为提取的算子和实现

# 使用方法
* 请保证当前处于 /InfraPowerTest/src/llama.cpp 目录下

电脑端编译命令：
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

安卓库编译命令：
在编译之前请确认已经安装了Android NDK

```bash
# 1) 设置 SDK/NDK 路径（按你本机实际版本改）
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
# 查看 NDK 版本目录
ls $ANDROID_SDK_ROOT/ndk
# 假设看到 26.3.11579264
export ANDROID_NDK=$ANDROID_SDK_ROOT/ndk/26.3.11579264
```

```bash
# 1. 创建构建目录
mkdir build-android
cd build-android

# 2. 使用 Android NDK 配置 CMake
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_OPENMP=OFF

# 3. 编译
cmake --build .
```

运行：
```bash
./ops_test
```
python :
```bash
cd exp1
python main.py
```
# 可能报错排查：

* ./ops_test: /home/tianruiming/miniconda3/envs/power/lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by ./ops_test)
```bash
conda install -n base -c conda-forge mamba -y #使用mamba加速安装
mamba install -n power -c conda-forge libstdcxx-ng=14.* libgcc-ng=14.* -y
```

* VSCode IntelliSense 无法找到头文件


运行这个查看python开发的头文件位置
```bash
python3 -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
```
拿到形如：

```bash
"/path/to/miniconda/envs/xxx/include/python3.x"
```
在.vscode的设置中增加这三个路径：

```bash
"${workspaceFolder}/3dparty/pybind11/include",
"${workspaceFolder}/3dparty/llamacpp/include",
"${workspaceFolder}/llamacpp/ggml/include",
"/path/to/miniconda/envs/xxx/include/python3.x"
```


# 附：算子提取方法：
llamacpp/test/test-backend-ops.cpp中给出了各个算子的调用示例