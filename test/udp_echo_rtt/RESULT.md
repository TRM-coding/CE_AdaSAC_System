# UDP 长连接往返测试

## 1. 测试目的

使用长连接方式，从主机向手机发送一段和“一个 token 编码后长度”接近的数据，再由手机原样回传，测量纯传输时间。

本次采用的近似是：

- 一个 `llama_token` 在内存中通常按 `int32_t` 存储
- 因此测试 payload 取 `4` 字节

本测试不统计：

- server 启动时间
- `adb` 推送时间
- 任何应用层握手时间

统计范围只包含：

- 主机 `sendto(...)`
- 手机接收并 `sendto(...)` 回传
- 主机 `recvfrom(...)`

也就是一次完整的 UDP 往返时间 `RTT`

## 2. 文件

- `udp_echo_server.c`：手机端 Android UDP echo server
- `udp_echo_client.c`：主机端 UDP RTT client
- `udp_echo_server.android`：编译后的 Android server
- `udp_echo_client`：编译后的 Linux client

## 3. 网络路径

- 主机地址：`10.20.0.2`
- 手机地址：`10.20.0.3`
- 端口：`19090`

## 4. 编译命令

主机端：

```bash
gcc -O3 -march=native -o udp_echo_client udp_echo_client.c
```

Android 端：

```bash
export NDK=/home/tianruiming/Android/Sdk/ndk/26.3.11579264
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang" \
  -O3 -o udp_echo_server.android udp_echo_server.c
```

## 5. 手机端部署与运行

```bash
adb shell 'mkdir -p /data/local/tmp/CE_Ada/udp_echo_rtt'
adb push udp_echo_server.android /data/local/tmp/CE_Ada/udp_echo_rtt/udp_echo_server
adb shell 'cd /data/local/tmp/CE_Ada/udp_echo_rtt && chmod +x udp_echo_server && nohup ./udp_echo_server 19090 > server.log 2>&1 < /dev/null &'
```

## 6. 主机端测试命令

单次往返：

```bash
./udp_echo_client 10.20.0.3 19090 4 1
```

100 轮长连接测试：

```bash
./udp_echo_client 10.20.0.3 19090 4 100
```

## 7. 实测结果

### 7.1 单次往返

- payload：`4` 字节
- rounds：`1`
- RTT：`28734.316 us`
- 约等于：`28.734 ms`

### 7.2 100 轮长连接测试

第 1 组：

- `avg_rtt_us = 29464.016`
- `p50_rtt_us = 29616.989`
- `p95_rtt_us = 39494.808`
- `avg_one_way_us = 14732.008`

第 2 组：

- `avg_rtt_us = 29445.260`
- `p50_rtt_us = 29685.588`
- `p95_rtt_us = 37334.687`
- `avg_one_way_us = 14722.630`

第 3 组：

- `avg_rtt_us = 30750.826`
- `p50_rtt_us = 30368.769`
- `p95_rtt_us = 44905.490`
- `avg_one_way_us = 15375.413`

### 7.3 结论

在当前主机到手机的这条链路上，`4` 字节 UDP payload 的长连接往返时间大致为：

- 平均 RTT：约 `29.4 ms` 到 `30.8 ms`
- 单向平均：约 `14.7 ms` 到 `15.4 ms`

如果只看“发送到手机，再由手机发回来”的总时间，可以近似记为：

- `约 30 ms / 次 RTT`

## 8. 备注

- 该测试使用的是 UDP，因此长轮数下可能出现偶发丢包
- 本次稳定完成的是 `100` 轮长连接窗口
- 测得时间反映的是当前这条 `10.20.0.x` 主机到手机网络路径，不代表 USB `adb` 文件传输速度
