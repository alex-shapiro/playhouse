// GAE (Generalized Advantage Estimation) with V-trace importance sampling
// CPU implementation

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

extern "C" {
    // Creates an empty _C module that can be imported from Python.
    // The import loads the .so so that TORCH_LIBRARY initializers run.
    PyObject* PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            NULL,
            -1,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}

namespace playhouse {

// Process a single row (one trajectory segment)
void gae_advantage_row(
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

// Validate input tensors
void validate_tensors(
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

// Process all rows [num_steps, horizon]
void gae_advantage(
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
    for (int offset = 0; offset < num_steps * horizon; offset += horizon) {
        gae_advantage_row(
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
}

// CPU implementation entry point
void compute_gae_advantage_cpu(
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
    validate_tensors(values, rewards, dones, importance, advantages, num_steps, horizon);

    gae_advantage(
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
}

// Register the operator schema
TORCH_LIBRARY(playhouse, m) {
    m.def("compute_gae_advantage("
          "Tensor(a!) values, "
          "Tensor(b!) rewards, "
          "Tensor(c!) dones, "
          "Tensor(d!) importance, "
          "Tensor(e!) advantages, "
          "float gamma, "
          "float lambda, "
          "float rho_clip, "
          "float c_clip"
          ") -> ()");
}

// Register CPU implementation
TORCH_LIBRARY_IMPL(playhouse, CPU, m) {
    m.impl("compute_gae_advantage", &compute_gae_advantage_cpu);
}

}  // namespace playhouse
