#include <torch/extension.h>

// fused_svd_matmul: computes X @ U^T @ V^T + bias via rank-1 updates
// Supports input X of shape (m, d) or (b, t, d), U: (d, k), V: (n, k), bias: broadcastable to (b, t, n) or (n)

at::Tensor fused_svd_matmul(
    const at::Tensor& X,
    const at::Tensor& U,
    const at::Tensor& V,
    const at::Tensor& bias) {
  TORCH_CHECK(X.dim() == 2 || X.dim() == 3, "X must be 2D or 3D");
  TORCH_CHECK(U.dim() == 2 && V.dim() == 2, "U and V must be 2D");

  // reshape X to 2D: (m, d)
  int64_t m, d;
  at::Tensor X2;
  std::vector<int64_t> batch_dims;
  if (X.dim() == 3) {
    auto b = X.size(0);
    auto t = X.size(1);
    d = X.size(2);
    m = b * t;
    batch_dims = {b, t};
    X2 = X.reshape({m, d});
  } else {
    m = X.size(0);
    d = X.size(1);
    X2 = X;
  }

  auto k = U.size(1);
  auto n = V.size(0);
  TORCH_CHECK(U.size(0) == d, "U.size(0) must match X.size(-1)");
  TORCH_CHECK(V.size(1) == k, "V.size(1) must match U.size(1)");

  // compute output Z2 of shape (m, n)
  auto Z2 = at::zeros({m, n}, X.options());

  for (int64_t r = 0; r < k; ++r) {
    auto u_r = U.select(1, r);  // [d]
    auto v_r = V.select(1, r);  // [n]
    // t = X2 @ u_r -> [m]
    auto tvec = at::mv(X2, u_r);
    // outer product and accumulate
    Z2.add_(at::ger(tvec, v_r));
  }

  // reshape back to original dims
  at::Tensor Z;
  if (!batch_dims.empty()) {
    Z = Z2.reshape({batch_dims[0], batch_dims[1], n});
  } else {
    Z = Z2;
  }

  // add bias if provided
  if (bias.numel() != 0) {
    Z = Z + bias;
  }

  return Z;
}

// binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_svd_matmul", &fused_svd_matmul,
        "Fused SVD-based low-rank matmul (rank-1 update) supporting batched inputs");
}