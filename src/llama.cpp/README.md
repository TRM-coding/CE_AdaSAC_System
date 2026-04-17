# 文件目录说明
* ./exp* 对应每次实验的源代码
* ./ops为提取的算子和实现

# 使用方法
* 请保证当前处于 InfraPowerTest/src/llama.cpp 目录下

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
确认好后创建编译目录：
```bash
# 1. 创建构建目录
mkdir build-android-official-cpu
cd build-android-official-cpu

# 2. 使用 Android NDK 配置 CMake
# arm64 真机的当前推荐构建口径：
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=armv8.7a" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF

# 3. 编译协同实验二进制
cmake --build . --target decode_svd_test svd_mobile_server -j$(nproc)
```

说明：

- 当前项目里 Android/ARM 真机的已验证构建目录是 `build-android-official-cpu`。
- 改动 `ggml-cpu.c`、`ggml-svd-offload.c`、`llama-graph.cpp` 后，必须同时重编并重新推送 `decode_svd_test` 和 `svd_mobile_server`，否则手机端可能命不中新的 AArch64 优化路径。
- `x86_64` Android 构建仅适用于模拟器或特殊设备，不是当前真机实验的基准口径。

## Android 真机协同推理测速注意事项

真机上比较 `decode_svd_test`、`svd_mobile_server`、官方 dense `Q4_0` baseline 时，必须把温度和调度口径对齐。否则同一台设备上可能出现 `40 tok/s` 和 `55 tok/s` 级别的差异，这不一定是编译产物或算子路径问题。

推荐真机准备流程：

```bash
adb reboot
adb wait-for-device
until adb shell getprop sys.boot_completed 2>/dev/null | grep -q 1; do sleep 1; done

# InfraPowerTest 的高分实验按冷机口径执行，建议等温度接近 25.0 C。
adb shell 'su -c "cat /sys/class/power_supply/battery/temp"'

adb shell 'su -c "sync && echo 3 > /proc/sys/vm/drop_caches"'
adb shell 'su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'
adb shell 'su -c "echo performance > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor"'
adb shell 'su -c "grep . /sys/devices/system/cpu/cpufreq/policy*/scaling_governor"'
```

官方 dense `Q4_0` CPU baseline 建议使用和 `InfraPowerTest` 对齐的命令形态：

```bash
adb shell 'su -c "cd /data/local/tmp/CE_Ada && \
  LD_LIBRARY_PATH=/data/local/tmp/CE_Ada \
  taskset -a ff ./llama-bench-fastflags \
    -m /data/local/tmp/CE_Ada/qwen.q4_0.gguf \
    -pg 1,512 \
    -r 2 \
    -t 8 \
    --prio 3 \
    --cpu-mask 0xff \
    --cpu-strict 1"'
```

记录和汇报真机吞吐时，至少要同时写清楚：

- 模型文件名和量化方式
- token 数与 bench 参数，例如 `-pg 1,512 -r 2` 或 `-p 0 -n 256`
- 线程数、`taskset` mask、`--cpu-mask`、`--cpu-strict`
- 是否使用 `--prio 3`
- `policy0/policy6` governor 是否为 `performance`
- 跑分前电池温度或 SoC 温度

已验证现象：

- 同一台 OnePlus 15 上，对齐 `taskset -a ff + --cpu-mask 0xff --cpu-strict 1 + --prio 3` 后，dense `qwen.q4_0.gguf` 冷机可到约 `52.5 tok/s`。
- 设备升温到约 `30 C` 后，同类 CPU benchmark 可能掉回高 `30 tok/s` 区间。
- 因此，协同推理和 SVD 本地 decode 的性能对比必须使用同一套真机环境，不要把冷机高分和热机裸跑结果直接比较。

运行：

电脑端：
```bash
./ops_test
```
python :
```bash
cd exp1
python main.py
```

安卓端：
```
#虚拟机启动（若使用真机调试，跳过这一步）
#列出虚拟机：
avdmanager list avd
#启动虚拟机：
emulator -avd <虚拟机名字>
#adb 检查是否能够识别到虚拟机
adb devices
```

```
#先进入到build-android目录，然后执行：
adb push ops_test /data/local/tmp/#将编译好的可执行文件传到安卓设备上
#进入安卓设备的命令行界面，找到刚刚传入的文件位置并运行
adb shell
cd /data/local/tmp
chmod +x ops_test
./ops_test
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
