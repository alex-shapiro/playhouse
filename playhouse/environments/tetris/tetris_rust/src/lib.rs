use std::mem;

use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

mod tetris;
pub mod tetromino;

use tetris::{Action, Tetris};

use crate::tetris::Log;

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

/// Wrapper for a single Tetris environment with Python-facing API
#[pyclass]
struct TetrisEnv {
    env: Tetris,
    obs_ptr: *mut f32,
    action_ptr: *mut u8,
    reward_ptr: *mut f32,
    terminal_ptr: *mut u8,
}

// Each TetrisEnv instance writes to non-overlapping regions of NumPy arrays
// so it is safe to send between threads. Pointers are accessed solely via the
// owning TetrisEnv instance.
unsafe impl Send for TetrisEnv {}
unsafe impl Sync for TetrisEnv {}

#[pymethods]
impl TetrisEnv {
    #[new]
    #[pyo3(signature = (observations, actions, rewards, terminals, truncations, seed, n_rows=20, n_cols=10, use_deck_obs=true, n_noise_obs=0, n_init_garbage=0))]
    fn new(
        observations: &Bound<'_, PyArray1<f32>>,
        actions: &Bound<'_, PyArray1<u8>>,
        rewards: &Bound<'_, PyArray1<f32>>,
        terminals: &Bound<'_, PyArray1<u8>>,
        truncations: &Bound<'_, PyArray1<u8>>,
        seed: u64,
        n_rows: usize,
        n_cols: usize,
        use_deck_obs: bool,
        n_noise_obs: usize,
        n_init_garbage: usize,
    ) -> PyResult<Self> {
        // Validate arrays are contiguous
        if !observations.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Observations must be contiguous",
            ));
        }
        if !actions.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Actions must be contiguous",
            ));
        }
        if !rewards.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Rewards must be contiguous",
            ));
        }
        if !terminals.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Terminals must be contiguous",
            ));
        }

        // Get raw pointers to numpy arrays
        let obs_ptr = unsafe { observations.as_slice_mut()?.as_mut_ptr() };
        let action_ptr = unsafe { actions.as_slice_mut()?.as_mut_ptr() };
        let reward_ptr = unsafe { rewards.as_slice_mut()?.as_mut_ptr() };
        let terminal_ptr = unsafe { terminals.as_slice_mut()?.as_mut_ptr() };

        // Create Tetris environment
        let env = Tetris::new(n_rows, n_cols, use_deck_obs, n_noise_obs, n_init_garbage);

        Ok(Self {
            env,
            obs_ptr,
            action_ptr,
            reward_ptr,
            terminal_ptr,
        })
    }

    fn reset(&mut self) {
        self.env.reset();
        self.sync_to_python();
    }

    fn step(&mut self) {
        // Read action from Python buffer
        let action = unsafe { Action::from(*self.action_ptr) };
        self.env.step(action);
        self.sync_to_python();
    }
}

impl TetrisEnv {
    /// Synchronize Tetris state to Python numpy arrays
    fn sync_to_python(&mut self) {
        // Copy observations
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.env.observations.as_ptr(),
                self.obs_ptr,
                self.env.observations.len(),
            );

            // Write reward
            *self.reward_ptr = self.env.rewards;

            // Write terminal
            *self.terminal_ptr = if self.env.terminals { 1 } else { 0 };
        }
    }
}

/// Vectorized environment wrapper
#[pyclass]
struct VecTetrisEnv {
    envs: Vec<TetrisEnv>,
}

#[pymethods]
impl VecTetrisEnv {
    #[new]
    #[pyo3(signature = (observations, actions, rewards, terminals, truncations, num_envs, seed, n_rows=20, n_cols=10, use_deck_obs=true, n_noise_obs=0, n_init_garbage=0))]
    fn new(
        _py: Python<'_>,
        observations: &Bound<'_, PyAny>,
        actions: &Bound<'_, PyArray1<u8>>,
        rewards: &Bound<'_, PyArray1<f32>>,
        terminals: &Bound<'_, PyArray1<u8>>,
        truncations: &Bound<'_, PyArray1<u8>>,
        num_envs: usize,
        seed: u64,
        n_rows: usize,
        n_cols: usize,
        use_deck_obs: bool,
        n_noise_obs: usize,
        n_init_garbage: usize,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_envs must be greater than 0",
            ));
        }

        let mut envs = Vec::with_capacity(num_envs);

        // Get mutable slices from the flat arrays
        let actions_slice = unsafe { actions.as_slice_mut()? };
        let rewards_slice = unsafe { rewards.as_slice_mut()? };
        let terminals_slice = unsafe { terminals.as_slice_mut()? };

        for i in 0..num_envs {
            // Get slices for this environment
            let obs_slice = observations.get_item(i)?;

            let obs_array: &Bound<'_, PyArray1<f32>> = obs_slice.cast()?;

            let obs_ptr = unsafe { obs_array.as_slice_mut()?.as_mut_ptr() };
            let action_ptr = &mut actions_slice[i] as *mut u8;
            let reward_ptr = &mut rewards_slice[i] as *mut f32;
            let terminal_ptr = &mut terminals_slice[i] as *mut u8;

            let tetris_env = Tetris::new(n_rows, n_cols, use_deck_obs, n_noise_obs, n_init_garbage);

            let mut env = TetrisEnv {
                env: tetris_env,
                obs_ptr,
                action_ptr,
                reward_ptr,
                terminal_ptr,
            };

            env.reset();
            envs.push(env);
        }

        Ok(Self { envs })
    }

    fn reset(&mut self, _seed: u64) {
        self.envs.par_iter_mut().for_each(|env| env.reset());
    }

    fn step(&mut self) {
        self.envs.par_iter_mut().for_each(|env| env.step());
    }

    fn log(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let totals = self
            .envs
            .par_iter_mut()
            .map(|env| mem::take(&mut env.env.log))
            .reduce(Log::default, |a, b| a + b);

        let dict = PyDict::new(py);

        if totals.n != 0.0 {
            let n = totals.n;
            dict.set_item("score", totals.score / n)?;
            dict.set_item("perf", totals.perf / n)?;
            dict.set_item("ep_length", totals.ep_length / n)?;
            dict.set_item("ep_return", totals.ep_return / n)?;
            dict.set_item("lines_deleted", totals.lines_deleted / n)?;
            dict.set_item("avg_combo", totals.avg_combo / n)?;
            dict.set_item("atn_frac_soft_drop", totals.atn_frac_soft_drop / n)?;
            dict.set_item("atn_frac_hard_drop", totals.atn_frac_hard_drop / n)?;
            dict.set_item("atn_frac_rotate", totals.atn_frac_rotate / n)?;
            dict.set_item("atn_frac_hold", totals.atn_frac_hold / n)?;
            dict.set_item("game_level", totals.game_level / n)?;
            dict.set_item("ticks_per_line", totals.ticks_per_line / n)?;
            dict.set_item("n", n)?;
        }

        Ok(dict.into())
    }

    fn render(&mut self, env_id: usize) -> PyResult<()> {
        if env_id >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "env_id out of range",
            ));
        }
        self.envs[env_id].env.render();
        Ok(())
    }
}

/// Python module definition
#[pymodule]
fn tetris_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TetrisEnv>()?;
    m.add_class::<VecTetrisEnv>()?;
    Ok(())
}
