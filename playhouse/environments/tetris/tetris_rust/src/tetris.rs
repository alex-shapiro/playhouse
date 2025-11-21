use crate::tetromino::{
    NUM_ROTATIONS, NUM_TETROMINOES, SIZE, TETROMINO_COLORS, TETROMINO_FILL_COLS,
    TETROMINO_FILL_ROWS, TETROMINOES,
};
use once_cell::sync::OnceCell;
use rand::{Rng, SeedableRng};
use raylib::prelude::*;
use std::thread;

const HALF_LINEWIDTH: i32 = 1;
const SQUARE_SIZE: i32 = 32;

// Store the main thread ID to ensure rendering only happens on main thread
static MAIN_THREAD_ID: OnceCell<thread::ThreadId> = OnceCell::new();
const DECK_SIZE: usize = 2 * NUM_TETROMINOES; // To implement the 7-bag system
const NUM_PREVIEW: usize = 2;
const NUM_FLOAT_OBS: usize = 6;

#[repr(u8)]
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    #[default]
    NoOp = 0,
    Left = 1,
    Right = 2,
    Rotate = 3,
    SoftDrop = 4,
    HardDrop = 5,
    Hold = 6,
}

impl From<u8> for Action {
    fn from(value: u8) -> Self {
        match value {
            0 => Action::NoOp,
            1 => Action::Left,
            2 => Action::Right,
            3 => Action::Rotate,
            4 => Action::SoftDrop,
            5 => Action::HardDrop,
            6 => Action::Hold,
            _ => Action::NoOp, // Default to NoOp for invalid values
        }
    }
}

#[allow(dead_code)]
const NUM_ROWS: usize = 20;
#[allow(dead_code)]
const NUM_COLS: usize = 10;

const MAX_TICKS: usize = 10000;
const PERSONAL_BEST: usize = 67890;
const INITIAL_TICKS_PER_FALL: usize = 6; // how many ticks before the tetromino naturally falls down of one square
const GARBAGE_KICKOFF_TICK: usize = 500;
const INITIAL_TICKS_PER_GARBAGE: usize = 100;

const LINES_PER_LEVEL: usize = 10;
// Revisit scoring with level. See https://tetris.wiki/Scoring
const SCORE_SOFT_DROP: usize = 1;
#[allow(dead_code)]
const REWARD_SOFT_DROP: f32 = 0.0;
const SCORE_HARD_DROP: usize = 2;
const REWARD_HARD_DROP: f32 = 0.02;
const REWARD_ROTATE: f32 = 0.01;
const REWARD_INVALID_ACTION: f32 = 0.0;

const SCORE_COMBO: [i32; 5] = [0, 100, 300, 500, 1000];
const REWARD_COMBO: [f32; 5] = [0.0, 0.1, 0.3, 0.5, 1.0];

#[derive(Default, Clone, Copy)]
pub struct Log {
    pub perf: f32,
    pub score: f32,
    pub ep_length: f32,
    pub ep_return: f32,
    pub lines_deleted: f32,
    pub avg_combo: f32,
    pub atn_frac_soft_drop: f32,
    pub atn_frac_hard_drop: f32,
    pub atn_frac_rotate: f32,
    pub atn_frac_hold: f32,
    pub game_level: f32,
    pub ticks_per_line: f32,
    pub n: f32,
}

impl std::ops::Add for Log {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Log {
            perf: self.perf + other.perf,
            score: self.score + other.score,
            ep_length: self.ep_length + other.ep_length,
            ep_return: self.ep_return + other.ep_return,
            lines_deleted: self.lines_deleted + other.lines_deleted,
            avg_combo: self.avg_combo + other.avg_combo,
            atn_frac_soft_drop: self.atn_frac_soft_drop + other.atn_frac_soft_drop,
            atn_frac_hard_drop: self.atn_frac_hard_drop + other.atn_frac_hard_drop,
            atn_frac_rotate: self.atn_frac_rotate + other.atn_frac_rotate,
            atn_frac_hold: self.atn_frac_hold + other.atn_frac_hold,
            game_level: self.game_level + other.game_level,
            ticks_per_line: self.ticks_per_line + other.ticks_per_line,
            n: self.n + other.n,
        }
    }
}

struct Client {
    total_cols: i32,
    total_rows: i32,
    ui_rows: i32,
    deck_rows: i32,
    rl: RaylibHandle,
    thread: RaylibThread,
}

pub struct Tetris {
    #[allow(dead_code)]
    client: Option<Client>,
    pub log: Log,
    pub observations: Vec<f32>,
    pub rewards: f32,
    #[allow(dead_code)]
    actions: Action,
    pub terminals: bool,
    n_rows: usize,
    n_cols: usize,
    use_deck_obs: bool,
    n_noise_obs: usize,
    n_init_garbage: usize,
    grid: Vec<i32>,
    rng: rand::rngs::SmallRng,
    tick: usize,
    tick_fall: usize,
    ticks_per_fall: usize,
    tick_garbage: usize,
    ticks_per_garbage: usize,
    score: usize,
    can_swap: bool,
    tetromino_deck: [usize; DECK_SIZE],
    hold_tetromino: Option<usize>,
    cur_position_in_deck: usize,
    cur_tetromino: usize,
    cur_tetromino_row: usize,
    cur_tetromino_col: usize,
    cur_tetromino_rot: usize,
    ep_return: f32,
    lines_deleted: u32,
    count_combos: u32,
    game_level: u32,
    atn_count_hard_drop: u32,
    atn_count_soft_drop: u32,
    atn_count_rotate: u32,
    atn_count_hold: u32,
    tetromino_counts: [u32; NUM_TETROMINOES],
}

impl Tetris {
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        use_deck_obs: bool,
        n_noise_obs: usize,
        n_init_garbage: usize,
    ) -> Self {
        let dim_obs =
            n_cols * n_rows + NUM_FLOAT_OBS + NUM_TETROMINOES * (NUM_PREVIEW + 2) + n_noise_obs;
        let mut tetris = Self {
            client: None,
            log: Log::default(),
            observations: vec![0.0; dim_obs],
            rewards: 0.0,
            actions: Action::default(),
            terminals: false,
            n_rows,
            n_cols,
            use_deck_obs,
            n_noise_obs,
            n_init_garbage,
            grid: vec![0; n_rows * n_cols],
            rng: rand::rngs::SmallRng::seed_from_u64(rand::rng().random()),
            tick: 0,
            tick_fall: 0,
            ticks_per_fall: INITIAL_TICKS_PER_FALL,
            tick_garbage: 0,
            ticks_per_garbage: INITIAL_TICKS_PER_GARBAGE,
            score: 0,
            can_swap: true,
            tetromino_deck: [0; DECK_SIZE],
            hold_tetromino: None,
            cur_position_in_deck: 0,
            cur_tetromino: 0,
            cur_tetromino_row: 0,
            cur_tetromino_col: 0,
            cur_tetromino_rot: 0,
            ep_return: 0.0,
            lines_deleted: 0,
            count_combos: 0,
            game_level: 1,
            atn_count_hard_drop: 0,
            atn_count_soft_drop: 0,
            atn_count_rotate: 0,
            atn_count_hold: 0,
            tetromino_counts: [0; NUM_TETROMINOES],
        };
        tetris.reset();
        tetris
    }

    pub fn add_log(&mut self) {
        let score = self.score as f32;
        let tick = self.tick as f32;

        self.log.score += score;
        self.log.perf += score / (PERSONAL_BEST as f32);
        self.log.ep_length += tick;
        self.log.ep_return += self.ep_return;
        self.log.lines_deleted += self.lines_deleted as f32;
        self.log.avg_combo += if self.count_combos > 0 {
            (self.lines_deleted as f32) / (self.count_combos as f32)
        } else {
            1.0
        };
        self.log.atn_frac_hard_drop += (self.atn_count_hard_drop as f32) / tick;
        self.log.atn_frac_soft_drop += (self.atn_count_soft_drop as f32) / tick;
        self.log.atn_frac_rotate += (self.atn_count_rotate as f32) / tick;
        self.log.atn_frac_hold += (self.atn_count_hold as f32) / tick;
        self.log.game_level += self.game_level as f32;
        self.log.ticks_per_line += if self.lines_deleted > 0 {
            tick / (self.lines_deleted as f32)
        } else {
            tick
        };
        self.log.n += 1.0;
    }

    #[allow(clippy::needless_range_loop)]
    pub fn compute_observations(&mut self) {
        // content of the grid: 0 for empty, 1 for placed blocks, 2 for the current tetromino
        for i in 0..(self.n_cols * self.n_rows) {
            self.observations[i] = if self.grid[i] != 0 { 1.0 } else { 0.0 };
        }

        for r in 0..SIZE {
            for c in 0..SIZE {
                if TETROMINOES[self.cur_tetromino][self.cur_tetromino_rot][r][c] == 1 {
                    let idx =
                        (self.cur_tetromino_row + r) * self.n_cols + c + self.cur_tetromino_col;
                    self.observations[idx] = 2.0;
                }
            }
        }

        let mut offset = self.n_cols * self.n_rows;
        self.observations[offset] = (self.tick as f32) / (MAX_TICKS as f32);
        self.observations[offset + 1] = (self.tick_fall as f32) / (self.ticks_per_fall as f32);
        self.observations[offset + 2] = (self.cur_tetromino_row as f32) / (self.n_rows as f32);
        self.observations[offset + 3] = (self.cur_tetromino_col as f32) / (self.n_cols as f32);
        self.observations[offset + 4] = self.cur_tetromino_rot as f32;
        self.observations[offset + 5] = if self.can_swap { 1.0 } else { 0.0 };
        offset += NUM_FLOAT_OBS;

        // Zero out the one-hot encoded part of the observations for deck and hold.
        let range_start = offset;
        let range_end = offset + NUM_TETROMINOES * (NUM_PREVIEW + 2);
        self.observations[range_start..range_end].fill(0.0);

        if self.use_deck_obs {
            // Deck, one hot encoded
            for j in 0..(NUM_PREVIEW + 1) {
                let tetromino_id = self.tetromino_deck[(self.cur_position_in_deck + j) % DECK_SIZE];
                self.observations[offset + tetromino_id] = 1.0;
                offset += NUM_TETROMINOES;
            }

            // Hold, one hot encoded
            if let Some(held) = self.hold_tetromino {
                self.observations[offset + held] = 1.0;
            }
            offset += NUM_TETROMINOES;
        } else {
            offset += NUM_TETROMINOES * (NUM_PREVIEW + 2);
        }

        // Turn off noise bits, one-by-one.
        if self.n_noise_obs > 0 {
            let noise_idx = self.rng.random_range(0..self.n_noise_obs);
            self.observations[offset + noise_idx] = 0.0;
        }
    }

    fn restore_grid(&mut self) {
        self.grid.fill(0);
    }

    fn refill_and_shuffle(array: &mut [usize], rng: &mut rand::rngs::SmallRng) {
        // Hold can change the deck distribution, so need to refill
        for (i, item) in array.iter_mut().enumerate() {
            *item = i;
        }

        // Fisher-Yates shuffle
        for i in (1..NUM_TETROMINOES).rev() {
            let j = rng.random_range(0..=i);
            array.swap(i, j);
        }
    }

    fn initialize_deck(&mut self) {
        // Implements a 7-bag system. The deck is composed of two bags.
        Self::refill_and_shuffle(&mut self.tetromino_deck[0..NUM_TETROMINOES], &mut self.rng); // First bag
        Self::refill_and_shuffle(
            &mut self.tetromino_deck[NUM_TETROMINOES..DECK_SIZE],
            &mut self.rng,
        ); // Second bag
        self.cur_position_in_deck = 0;
        self.cur_tetromino = self.tetromino_deck[self.cur_position_in_deck];
    }

    fn spawn_new_tetromino(&mut self) {
        self.cur_position_in_deck = (self.cur_position_in_deck + 1) % DECK_SIZE;
        self.cur_tetromino = self.tetromino_deck[self.cur_position_in_deck];
        self.cur_tetromino_rot = 0;

        if self.cur_position_in_deck == 0 {
            // Now using the first bag, so shuffle the second bag
            Self::refill_and_shuffle(
                &mut self.tetromino_deck[NUM_TETROMINOES..DECK_SIZE],
                &mut self.rng,
            );
        } else if self.cur_position_in_deck == NUM_TETROMINOES {
            // Now using the second bag, so shuffle the first bag
            Self::refill_and_shuffle(&mut self.tetromino_deck[0..NUM_TETROMINOES], &mut self.rng);
        }

        self.cur_tetromino_col = self.n_cols / 2;
        self.cur_tetromino_row = 0;
        self.tick_fall = 0;
        self.tetromino_counts[self.cur_tetromino] += 1;
    }

    // This is only used to check if the game is done
    #[allow(clippy::needless_range_loop)]
    fn can_spawn_new_tetromino(&self) -> bool {
        let next_pos = (self.cur_position_in_deck + 1) % DECK_SIZE;
        let next_tetromino = self.tetromino_deck[next_pos];
        for c in 0..(TETROMINO_FILL_COLS[next_tetromino][0] as usize) {
            for r in 0..(TETROMINO_FILL_ROWS[next_tetromino][0] as usize) {
                if (self.grid[r * self.n_cols + c + self.n_cols / 2] != 0)
                    && (TETROMINOES[next_tetromino][0][r][c] == 1)
                {
                    return false;
                }
            }
        }
        true
    }

    #[allow(clippy::needless_range_loop)]
    fn can_soft_drop(&self) -> bool {
        if self.cur_tetromino_row
            == (self.n_rows
                - TETROMINO_FILL_ROWS[self.cur_tetromino][self.cur_tetromino_rot] as usize)
        {
            return false;
        }
        for c in 0..(TETROMINO_FILL_COLS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
            for r in 0..(TETROMINO_FILL_ROWS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
                if (self.grid
                    [(r + self.cur_tetromino_row + 1) * self.n_cols + c + self.cur_tetromino_col]
                    != 0)
                    && (TETROMINOES[self.cur_tetromino][self.cur_tetromino_rot][r][c] == 1)
                {
                    return false;
                }
            }
        }
        true
    }

    #[allow(clippy::needless_range_loop)]
    fn can_go_left(&self) -> bool {
        if self.cur_tetromino_col == 0 {
            return false;
        }
        for c in 0..(TETROMINO_FILL_COLS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
            for r in 0..(TETROMINO_FILL_ROWS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
                if (self.grid
                    [(r + self.cur_tetromino_row) * self.n_cols + c + self.cur_tetromino_col - 1]
                    != 0)
                    && (TETROMINOES[self.cur_tetromino][self.cur_tetromino_rot][r][c] == 1)
                {
                    return false;
                }
            }
        }
        true
    }

    #[allow(clippy::needless_range_loop)]
    fn can_go_right(&self) -> bool {
        if self.cur_tetromino_col
            == (self.n_cols
                - TETROMINO_FILL_COLS[self.cur_tetromino][self.cur_tetromino_rot] as usize)
        {
            return false;
        }

        for c in 0..(TETROMINO_FILL_COLS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
            for r in 0..(TETROMINO_FILL_ROWS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
                if (self.grid
                    [(r + self.cur_tetromino_row) * self.n_cols + c + self.cur_tetromino_col + 1]
                    != 0)
                    && (TETROMINOES[self.cur_tetromino][self.cur_tetromino_rot][r][c] == 1)
                {
                    return false;
                }
            }
        }

        true
    }

    #[allow(clippy::needless_range_loop)]
    fn can_hold(&self) -> bool {
        if !self.can_swap {
            return false;
        }
        let Some(held) = self.hold_tetromino else {
            return true;
        };
        for c in 0..(TETROMINO_FILL_COLS[held][self.cur_tetromino_rot] as usize) {
            for r in 0..(TETROMINO_FILL_ROWS[held][self.cur_tetromino_rot] as usize) {
                if (self.grid
                    [(r + self.cur_tetromino_row) * self.n_cols + c + self.cur_tetromino_col]
                    != 0)
                    && (TETROMINOES[held][self.cur_tetromino_rot][r][c] == 1)
                {
                    return false;
                }
            }
        }
        true
    }

    #[allow(clippy::needless_range_loop)]
    fn can_rotate(&self) -> bool {
        let next_rot = (self.cur_tetromino_rot + 1) % NUM_ROTATIONS;
        if self.cur_tetromino_col
            > (self.n_cols - TETROMINO_FILL_COLS[self.cur_tetromino][next_rot] as usize)
        {
            return false;
        }
        if self.cur_tetromino_row
            > (self.n_rows - TETROMINO_FILL_ROWS[self.cur_tetromino][next_rot] as usize)
        {
            return false;
        }
        for c in 0..(TETROMINO_FILL_COLS[self.cur_tetromino][next_rot] as usize) {
            for r in 0..(TETROMINO_FILL_ROWS[self.cur_tetromino][next_rot] as usize) {
                if (self.grid
                    [(r + self.cur_tetromino_row) * self.n_cols + c + self.cur_tetromino_col]
                    != 0)
                    && (TETROMINOES[self.cur_tetromino][next_rot][r][c] == 1)
                {
                    return false;
                }
            }
        }
        true
    }

    fn is_full_row(&self, row: usize) -> bool {
        for c in 0..self.n_cols {
            if self.grid[row * self.n_cols + c] == 0 {
                return false;
            }
        }
        true
    }

    fn clear_row(&mut self, row: usize) {
        for r in (1..=row).rev() {
            for c in 0..self.n_cols {
                self.grid[r * self.n_cols + c] = self.grid[(r - 1) * self.n_cols + c];
            }
        }
        for c in 0..self.n_cols {
            self.grid[c] = 0;
        }
    }

    fn add_garbage_lines(&mut self, num_lines: usize, num_holes: usize) {
        // Check if adding garbage would cause an immediate game over
        for r in 0..num_lines {
            for c in 0..self.n_cols {
                if self.grid[r * self.n_cols + c] != 0 {
                    self.terminals = true; // Game over
                    return;
                }
            }
        }

        // Shift the existing grid up by num_lines
        for r in 0..(self.n_rows - num_lines) {
            for c in 0..self.n_cols {
                self.grid[r * self.n_cols + c] = self.grid[(r + num_lines) * self.n_cols + c];
            }
        }

        // Add new garbage lines at the bottom
        for r in (self.n_rows - num_lines)..self.n_rows {
            // First, fill the entire row with garbage
            for c in 0..self.n_cols {
                let garbage_val = -(self.rng.random_range(0..NUM_TETROMINOES) as i32 + 1);
                self.grid[r * self.n_cols + c] = garbage_val;
            }

            // Create holes by selecting distinct columns
            // Use a fixed-size array since n_cols is typically 10
            let mut cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
            // Shuffle column indices (Fisher-Yates)
            for i in (1..self.n_cols).rev() {
                let j = self.rng.random_range(0..=i);
                cols.swap(i, j);
            }
            for &col in cols.iter().take(num_holes) {
                self.grid[r * self.n_cols + col] = 0;
            }
        }

        // Move the current piece up as well
        self.cur_tetromino_row = self.cur_tetromino_row.saturating_sub(num_lines);
    }

    pub fn reset(&mut self) {
        self.score = 0;
        self.hold_tetromino = None;
        self.tick = 0;
        self.game_level = 1;
        self.ticks_per_fall = INITIAL_TICKS_PER_FALL;
        self.tick_fall = 0;
        self.ticks_per_garbage = INITIAL_TICKS_PER_GARBAGE;
        self.tick_garbage = 0;
        self.can_swap = true;

        self.ep_return = 0.0;
        self.count_combos = 0;
        self.lines_deleted = 0;
        self.atn_count_hard_drop = 0;
        self.atn_count_soft_drop = 0;
        self.atn_count_rotate = 0;
        self.atn_count_hold = 0;
        self.tetromino_counts.fill(0);

        self.restore_grid();
        // This acts as a learning curriculum, exposing agents to garbage lines later
        self.add_garbage_lines(self.n_init_garbage, 9);

        // Noise obs effectively jitters the action.
        // The agents will eventually learn to ignore these.
        for i in 0..self.n_noise_obs {
            self.observations[234 + i] = 1.0;
        }

        self.initialize_deck();
        self.spawn_new_tetromino();
        self.compute_observations();
    }

    #[allow(clippy::needless_range_loop)]
    fn place_tetromino(&mut self) {
        let mut row_to_check = self.cur_tetromino_row
            + TETROMINO_FILL_ROWS[self.cur_tetromino][self.cur_tetromino_rot] as usize
            - 1;
        let mut lines_deleted = 0;
        self.can_swap = true;

        // Fill the main grid with the tetromino
        for c in 0..(TETROMINO_FILL_COLS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
            for r in 0..(TETROMINO_FILL_ROWS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
                if TETROMINOES[self.cur_tetromino][self.cur_tetromino_rot][r][c] == 1 {
                    self.grid
                        [(r + self.cur_tetromino_row) * self.n_cols + c + self.cur_tetromino_col] =
                        (self.cur_tetromino + 1) as i32;
                }
            }
        }

        // Proceed to delete the complete rows
        for _ in 0..(TETROMINO_FILL_ROWS[self.cur_tetromino][self.cur_tetromino_rot] as usize) {
            if self.is_full_row(row_to_check) {
                self.clear_row(row_to_check);
                lines_deleted += 1;
            } else {
                row_to_check = row_to_check.saturating_sub(1);
            }
        }

        if lines_deleted > 0 {
            self.count_combos += 1;
            self.lines_deleted += lines_deleted;
            self.score += SCORE_COMBO[lines_deleted as usize] as usize;
            self.rewards += REWARD_COMBO[lines_deleted as usize];
            self.ep_return += REWARD_COMBO[lines_deleted as usize];

            // These determine the game difficulty. Consider making them args.
            self.game_level = 1 + self.lines_deleted / LINES_PER_LEVEL as u32;
            self.ticks_per_fall =
                (INITIAL_TICKS_PER_FALL as i32 - self.game_level as i32 / 4).max(3) as usize;
            self.ticks_per_garbage =
                ((INITIAL_TICKS_PER_GARBAGE as f64 - 7.0 * (self.game_level as f64).sqrt()) as i32)
                    .max(40) as usize;
        }

        if self.can_spawn_new_tetromino() {
            self.spawn_new_tetromino();
        } else {
            self.terminals = true; // Game over
        }
    }

    pub fn step(&mut self, action: Action) {
        self.terminals = false;
        self.rewards = 0.0;
        self.tick += 1;
        self.tick_fall += 1;
        self.tick_garbage += 1;

        match action {
            Action::Left => {
                if self.can_go_left() {
                    self.cur_tetromino_col -= 1;
                } else {
                    self.rewards += REWARD_INVALID_ACTION;
                    self.ep_return += REWARD_INVALID_ACTION;
                }
            }
            Action::Right => {
                if self.can_go_right() {
                    self.cur_tetromino_col += 1;
                } else {
                    self.rewards += REWARD_INVALID_ACTION;
                    self.ep_return += REWARD_INVALID_ACTION;
                }
            }
            Action::Rotate => {
                self.atn_count_rotate += 1;
                if self.can_rotate() {
                    self.cur_tetromino_rot = (self.cur_tetromino_rot + 1) % NUM_ROTATIONS;
                    self.rewards += REWARD_ROTATE;
                    self.ep_return += REWARD_ROTATE;
                } else {
                    self.rewards += REWARD_INVALID_ACTION;
                    self.ep_return += REWARD_INVALID_ACTION;
                }
            }
            Action::SoftDrop => {
                self.atn_count_soft_drop += 1;
                if self.can_soft_drop() {
                    self.cur_tetromino_row += 1;
                    self.score += SCORE_SOFT_DROP;
                } else {
                    self.rewards += REWARD_INVALID_ACTION;
                    self.ep_return += REWARD_INVALID_ACTION;
                }
            }
            Action::Hold => {
                self.atn_count_hold += 1;
                if self.can_hold() {
                    let t1 = self.cur_tetromino;
                    match self.hold_tetromino {
                        None => {
                            self.spawn_new_tetromino();
                            self.hold_tetromino = Some(t1);
                            self.can_swap = false;
                        }
                        Some(t2) => {
                            self.cur_tetromino = t2;
                            self.tetromino_deck[self.cur_position_in_deck] = t2;
                            self.hold_tetromino = Some(t1);
                            self.can_swap = false;
                            self.cur_tetromino_rot = 0;
                            self.cur_tetromino_col = self.n_cols / 2;
                            self.cur_tetromino_row = 0;
                            self.tick_fall = 0;
                        }
                    }
                } else {
                    self.rewards += REWARD_INVALID_ACTION;
                    self.ep_return += REWARD_INVALID_ACTION;
                }
            }
            Action::HardDrop => {
                self.atn_count_hard_drop += 1;
                while self.can_soft_drop() {
                    self.cur_tetromino_row += 1;
                    // NOTE: this seems to be a super effective reward trick
                    self.rewards += REWARD_HARD_DROP;
                    self.ep_return += REWARD_HARD_DROP;
                }
                self.score += SCORE_HARD_DROP;
                self.place_tetromino();
            }
            Action::NoOp => {} // No operation
        }

        if self.tick_fall >= self.ticks_per_fall {
            self.tick_fall = 0;
            if self.can_soft_drop() {
                self.cur_tetromino_row += 1;
            } else {
                self.place_tetromino();
            }
        }

        if self.tick >= GARBAGE_KICKOFF_TICK && self.tick_garbage >= self.ticks_per_garbage {
            self.tick_garbage = 0;
            let num_holes = (self.game_level / 8).clamp(1, 5) as usize;
            self.add_garbage_lines(1, num_holes);
        }

        if self.terminals || (self.tick >= MAX_TICKS) {
            self.add_log();
            self.reset();
        }

        self.compute_observations();
    }

    pub fn render(&mut self) {
        // Ensure we're on the main thread
        let main_thread_id = MAIN_THREAD_ID.get_or_init(|| thread::current().id());
        assert_eq!(
            *main_thread_id,
            thread::current().id(),
            "Rendering must be called from the main thread"
        );

        // Initialize client/window if needed
        if self.client.is_none() {
            let ui_rows = 1;
            let deck_rows = SIZE as i32;
            let total_rows = 1 + ui_rows + 1 + deck_rows + 1 + self.n_rows as i32 + 1;
            let total_cols = (1 + self.n_cols + 1).max(1 + 3 * NUM_PREVIEW) as i32;

            let (rl, thread) = raylib::init()
                .size(SQUARE_SIZE * total_cols, SQUARE_SIZE * total_rows)
                .title("Tetris")
                .build();

            self.client = Some(Client {
                total_cols,
                total_rows,
                ui_rows,
                deck_rows,
                rl,
                thread,
            });
        }

        let client = self.client.as_mut().unwrap();

        // Check for window close or escape key
        if client.rl.window_should_close() || client.rl.is_key_down(KeyboardKey::KEY_ESCAPE) {
            return;
        }

        // Toggle fullscreen with TAB
        if client.rl.is_key_pressed(KeyboardKey::KEY_TAB) {
            client.rl.toggle_fullscreen();
        }

        // Colors
        let border_color = Color::new(100, 100, 100, 255);
        let dash_color = Color::new(80, 80, 80, 255);
        let dash_color_bright = Color::new(150, 150, 150, 255);
        let dash_color_dark = Color::new(50, 50, 50, 255);

        let mut d = client.rl.begin_drawing(&client.thread);
        d.clear_background(Color::BLACK);

        // Draw outer grid border
        for r in 0..client.total_rows {
            for c in 0..client.total_cols {
                let x = c * SQUARE_SIZE;
                let y = r * SQUARE_SIZE;

                if (c == 0)
                    || (c == client.total_cols - 1)
                    || ((r > 1 + client.ui_rows) && (r < 1 + client.ui_rows + 1 + client.deck_rows))
                    || ((r > 1 + client.ui_rows + client.deck_rows + 1)
                        && (c >= self.n_rows as i32))
                    || (r == 0)
                    || (r == 1 + client.ui_rows)
                    || (r == 1 + client.ui_rows + 1 + client.deck_rows)
                    || (r == client.total_rows - 1)
                {
                    d.draw_rectangle(
                        x + HALF_LINEWIDTH,
                        y + HALF_LINEWIDTH,
                        SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                        SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                        border_color,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        2 * HALF_LINEWIDTH,
                        dash_color_dark,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y + SQUARE_SIZE - HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        2 * HALF_LINEWIDTH,
                        dash_color_dark,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        2 * HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        dash_color_dark,
                    );
                    d.draw_rectangle(
                        x + SQUARE_SIZE - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        2 * HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        dash_color_dark,
                    );
                }
            }
        }

        // Draw main grid
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                let x = (c + 1) as i32 * SQUARE_SIZE;
                let y = (1 + client.ui_rows + 1 + client.deck_rows + 1 + r as i32) * SQUARE_SIZE;
                let block_id = self.grid[r * self.n_cols + c];

                let color = if block_id == 0 {
                    Color::BLACK
                } else if block_id < 0 {
                    TETROMINO_COLORS[(-block_id - 1) as usize]
                } else {
                    TETROMINO_COLORS[(block_id - 1) as usize]
                };

                d.draw_rectangle(
                    x + HALF_LINEWIDTH,
                    y + HALF_LINEWIDTH,
                    SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                    SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                    color,
                );
                d.draw_rectangle(
                    x - HALF_LINEWIDTH,
                    y - HALF_LINEWIDTH,
                    SQUARE_SIZE,
                    2 * HALF_LINEWIDTH,
                    dash_color,
                );
                d.draw_rectangle(
                    x - HALF_LINEWIDTH,
                    y + SQUARE_SIZE - HALF_LINEWIDTH,
                    SQUARE_SIZE,
                    2 * HALF_LINEWIDTH,
                    dash_color,
                );
                d.draw_rectangle(
                    x - HALF_LINEWIDTH,
                    y - HALF_LINEWIDTH,
                    2 * HALF_LINEWIDTH,
                    SQUARE_SIZE,
                    dash_color,
                );
                d.draw_rectangle(
                    x + SQUARE_SIZE - HALF_LINEWIDTH,
                    y - HALF_LINEWIDTH,
                    2 * HALF_LINEWIDTH,
                    SQUARE_SIZE,
                    dash_color,
                );
            }
        }

        // Draw current tetromino
        for r in 0..SIZE {
            for c in 0..SIZE {
                if TETROMINOES[self.cur_tetromino][self.cur_tetromino_rot][r][c] == 1 {
                    let x = (c + self.cur_tetromino_col + 1) as i32 * SQUARE_SIZE;
                    let y = (1
                        + client.ui_rows
                        + 1
                        + client.deck_rows
                        + 1
                        + r as i32
                        + self.cur_tetromino_row as i32)
                        * SQUARE_SIZE;
                    let color = TETROMINO_COLORS[self.cur_tetromino];

                    d.draw_rectangle(
                        x + HALF_LINEWIDTH,
                        y + HALF_LINEWIDTH,
                        SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                        SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                        color,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        2 * HALF_LINEWIDTH,
                        dash_color,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y + SQUARE_SIZE - HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        2 * HALF_LINEWIDTH,
                        dash_color,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        2 * HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        dash_color,
                    );
                    d.draw_rectangle(
                        x + SQUARE_SIZE - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        2 * HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        dash_color,
                    );
                }
            }
        }

        // Draw deck preview (next pieces)
        for i in 0..NUM_PREVIEW {
            let deck_idx = (self.cur_position_in_deck + 1 + i) % DECK_SIZE;
            let tetromino_id = self.tetromino_deck[deck_idx];
            for r in 0..SIZE {
                for c in 0..2 {
                    let x = (c + 1 + 3 * i) as i32 * SQUARE_SIZE;
                    let y = (1 + client.ui_rows + 1 + r as i32) * SQUARE_SIZE;
                    let r_offset = SIZE - TETROMINO_FILL_ROWS[tetromino_id][0] as usize;

                    let color = if r < r_offset {
                        Color::BLACK
                    } else if TETROMINOES[tetromino_id][0][r - r_offset][c] == 0 {
                        Color::BLACK
                    } else {
                        TETROMINO_COLORS[tetromino_id]
                    };

                    d.draw_rectangle(
                        x + HALF_LINEWIDTH,
                        y + HALF_LINEWIDTH,
                        SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                        SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                        color,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        2 * HALF_LINEWIDTH,
                        dash_color_bright,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y + SQUARE_SIZE - HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        2 * HALF_LINEWIDTH,
                        dash_color_bright,
                    );
                    d.draw_rectangle(
                        x - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        2 * HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        dash_color_bright,
                    );
                    d.draw_rectangle(
                        x + SQUARE_SIZE - HALF_LINEWIDTH,
                        y - HALF_LINEWIDTH,
                        2 * HALF_LINEWIDTH,
                        SQUARE_SIZE,
                        dash_color_bright,
                    );
                }
            }
        }

        // Draw hold tetromino
        for r in 0..SIZE {
            for c in 0..2 {
                let x = (client.total_cols - 3 + c as i32) * SQUARE_SIZE;
                let y = (1 + client.ui_rows + 1 + r as i32) * SQUARE_SIZE;

                let color = if let Some(hold_id) = self.hold_tetromino {
                    let r_offset = SIZE - TETROMINO_FILL_ROWS[hold_id][0] as usize;
                    if r < r_offset || TETROMINOES[hold_id][0][r - r_offset][c] == 0 {
                        Color::BLACK
                    } else {
                        TETROMINO_COLORS[hold_id]
                    }
                } else {
                    Color::BLACK
                };

                d.draw_rectangle(
                    x + HALF_LINEWIDTH,
                    y + HALF_LINEWIDTH,
                    SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                    SQUARE_SIZE - 2 * HALF_LINEWIDTH,
                    color,
                );
                d.draw_rectangle(
                    x - HALF_LINEWIDTH,
                    y - HALF_LINEWIDTH,
                    SQUARE_SIZE,
                    2 * HALF_LINEWIDTH,
                    dash_color_bright,
                );
                d.draw_rectangle(
                    x - HALF_LINEWIDTH,
                    y + SQUARE_SIZE - HALF_LINEWIDTH,
                    SQUARE_SIZE,
                    2 * HALF_LINEWIDTH,
                    dash_color_bright,
                );
                d.draw_rectangle(
                    x - HALF_LINEWIDTH,
                    y - HALF_LINEWIDTH,
                    2 * HALF_LINEWIDTH,
                    SQUARE_SIZE,
                    dash_color_bright,
                );
                d.draw_rectangle(
                    x + SQUARE_SIZE - HALF_LINEWIDTH,
                    y - HALF_LINEWIDTH,
                    2 * HALF_LINEWIDTH,
                    SQUARE_SIZE,
                    dash_color_bright,
                );
            }
        }

        // Draw UI text
        d.draw_text(
            &format!("Score: {}", self.score),
            SQUARE_SIZE + 4,
            SQUARE_SIZE + 4,
            28,
            Color::new(255, 160, 160, 255),
        );
        d.draw_text(
            &format!("Lvl: {}", self.game_level),
            (client.total_cols - 4) * SQUARE_SIZE,
            SQUARE_SIZE + 4,
            28,
            Color::new(160, 255, 160, 255),
        );
    }
}
