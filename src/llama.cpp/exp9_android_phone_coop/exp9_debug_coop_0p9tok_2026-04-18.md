# 2026-04-18 协同吞吐仅 ~0.9 tok/s 的排查结论（exp9）

## 1. 现象

本次协同样例（Linux 客户端 + Android 服务端，decode=3，offload_rate=1.0）得到：

- Decode-only throughput: `0.956704 tok/s`
- End-to-end throughput: `0.955704 tok/s`

对应日志：`/tmp/linux_android_coop_3tok.log`

## 2. 关键证据

### 2.1 协同确实生效（不是 fallback）

- `[svd-offload-connect] connected host=192.168.5.164 port=7792`
- `[svd-ffn-exec] ... request_started=1 request_ok=1 path=remote_success`
- `ffn_req=84`

说明并非退回本地 FFN。

### 2.2 主瓶颈在 RPC 等待，不在 Linux 本地算力

客户端：

- `wait_ffn=3052.977 ms`
- `ffn_req=84`
- 平均每个 FFN RPC 等待约 `36.34 ms`

而同机本地（不协同）8 token 对照：

- Decode-only throughput: `34.1282 tok/s`
- 日志：`/tmp/exp9_local_8tok.log`

说明主机本地算力正常，慢在协同链路/远端执行路径。

### 2.3 远端服务端处于 quant fastpath 关闭状态

服务端日志显示：

- `quant_mode=off`
- `quant_fallbacks=168`
- `fallback_type_disabled=168`

这表示 Q4 相关量化路径被禁用，发生了大量 fallback。

对照历史文档（exp6）可见更优配置是：

- `quant_mode=q8_0`
- `quant_fallbacks=0`

### 2.4 当前服务器到手机 IP 不可直连

在 `10.126.59.25` 上测试：

- `ping 192.168.5.164` 丢包 100%
- TCP connect `192.168.5.164:7792` 超时

所以“有线直连”在这台 Linux 服务器视角下并没有形成可直连链路，实际大概率走了额外转发路径（延迟更高）。

## 3. 为什么会比此前 ~2 tok/s 更低

综合上面证据，当前降速是叠加效应：

1. `offload_rate=1.0`：decode token 的 FFN 全远端（84 次 RPC）
2. RPC 平均等待约 `36ms/req`，链路时延偏大
3. 服务端 `quant_mode=off` 导致 `quant_fallbacks=168`
4. 仅 decode `3` token，初始化与首轮 miss 的固定开销摊不薄

## 4. 建议的修复顺序（按优先级）

1. 先恢复服务端量化快路径：
   - 启动服务端时不要用 `off`
   - 改为 `q8_0`（并确认日志 `quant_fallbacks=0`）
2. 确认“真正可直连”的 endpoint：
   - 在客户端机器上必须 `ping/tcp connect` 手机 IP 成功
   - 若不通，不要把该路径当作“有线直连”结果
3. 协同测速不要只看 3 token：
   - 建议至少 24 或 64 token 观察稳态
4. 根据 exp9 文档，优先试 `0 < offload_rate < 1`（如 0.5/0.8）：
   - 让客户端本地保留 rank，命中 Q4 SVD vec 快路径

## 5. 建议的复测通过标准

客户端侧应看到：

- `[svd-offload-connect] connected ...`
- `[svd-ffn-exec] ... path=remote_success`
- `[svd-offload-client-stage-profile] rpc_wait_total` 显著低于当前 3296ms（同 token 口径）

服务端侧应看到：

- `quant_mode=q8_0`
- `quant_fallbacks=0`（或明显减少）

