// GAE (Generalized Advantage Estimation) with V-trace importance sampling
// CUDA implementation

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace playhouse {

// Device function for processing a single row
__host__ __device__ void gae_advantage_row_cuda(
    float* values,
    float* rewards,
    float* dones,
    float* importance,
    float* advantages,
    float gamma,
    float lambda,
    float rho_clip,
    float c_clip,
    int horizon
) {
    float last_gae = 0;
    for (int t = horizon - 2; t >= 0; t--) {
        int t_next = t + 1;
        float not_terminal = 1.0f - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t * (rewards[t_next] + gamma * values[t_next] * not_terminal - values[t]);
        last_gae = delta + gamma * lambda * c_t * last_gae * not_terminal;
        advantages[t] = last_gae;
    }
}

// Validate input tensors for CUDA
void validate_tensors_cuda(
    torch::Tensor values,
    torch::Tensor rewards,
    torch::Tensor dones,
    torch::Tensor importance,
    torch::Tensor advantages,
    int num_steps,
    int horizon
) {
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}

// CUDA kernel - each thread processes one row [num_steps, horizon]
__global__ void gae_advantage_kernel(
    float* values,
    float* rewards,
    float* dones,
    float* importance,
    float* advantages,
    float gamma,
    float lambda,
    float rho_clip,
    float c_clip,
    int num_steps,
    int horizon
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row * horizon;
    gae_advantage_row_cuda(
        values + offset,
        rewards + offset,
        dones + offset,
        importance + offset,
        advantages + offset,
        gamma,
        lambda,
        rho_clip,
        c_clip,
        horizon
    );
}

// CUDA implementation entry point
void compute_gae_advantage_cuda(
    torch::Tensor values,
    torch::Tensor rewards,
    torch::Tensor dones,
    torch::Tensor importance,
    torch::Tensor advantages,
    double gamma,
    double lambda,
    double rho_clip,
    double c_clip
) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    validate_tensors_cuda(values, rewards, dones, importance, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    gae_advantage_kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        importance.data_ptr<float>(),
        advantages.data_ptr<float>(),
        static_cast<float>(gamma),
        static_cast<float>(lambda),
        static_cast<float>(rho_clip),
        static_cast<float>(c_clip),
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(playhouse, CUDA, m) {
    m.impl("compute_gae_advantage", &compute_gae_advantage_cuda);
}

}  // namespace playhouse
