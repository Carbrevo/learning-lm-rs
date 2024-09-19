#![allow(non_snake_case)]
#![allow(dead_code)]

use crate::tensor::Tensor;
use std::iter::zip;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    //todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")

    assert!(
        [y.shape().len(), x.shape().len()] == [2; 2],
        "y={:?} x={:?}",
        y.shape(),
        x.shape()
    );
    assert!(y.shape() == x.shape());
    assert!(w.shape().len() == 1);
    assert!(w.shape()[0] == x.shape()[1]);

    let M = x.shape()[0];
    let N = x.shape()[1];
    let mean = |i| {
        ((0..N)
            .map(|j| (x.data()[i * N + j] as f32).powi(2))
            .sum::<f32>()
            / (N as f32)
            + epsilon)
            .sqrt()
    };
    let rv_iter = (0..M).map(|i| {
        let mean = mean(i);
        (0..N).map(move |j| w.data()[j] / mean * x.data()[i * N + j])
    });
    let data = rv_iter.flatten().collect::<Vec<f32>>();
    *y = Tensor::<f32>::new(data, y.shape());
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    // let len = y.size();
    // assert!(len == x.size());

    // let _y = unsafe { y.data_mut() };
    // let _x = x.data();

    //todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
    assert!(y.shape() == x.shape());
    //let silu =
    unsafe {
        y.data_mut()
            .iter_mut()
            .zip(x.data())
            .for_each(|(ye, xe)| *ye = *xe * *ye / (1.0 + (-*xe).exp()));
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    //todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
    assert!(
        [a.shape().len(), b.shape().len(), c.shape().len()] == [2; 3],
        "a{:?} b{:?} c{:?}",
        a.shape(),
        b.shape(),
        c.shape()
    );
    assert!(a.shape()[1] == b.shape()[1]);
    assert!(c.shape() == &vec![a.shape()[0], b.shape()[0]]);

    let K = a.shape()[1];
    let M = c.shape()[0];
    let N = c.shape()[1];
    let a_mi = |m, i| a.data()[m * K + i];
    let b_in = |i, n| b.data()[n * K + i];
    let ab_data = (0..M * N)
        .map(|x| (0..K).map(|i| a_mi(x / N, i) * b_in(i, x % N)).sum())
        .collect();
    let ab = Tensor::<f32>::new(ab_data, c.shape());
    unsafe {
        c.data_mut()
            .iter_mut()
            .zip(ab.data())
            .for_each(|(ce, abe)| *ce = beta * *ce + alpha * *abe);
    }
}

pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    zip(x, y).fold(0.0, |acc, (a, b)| acc + a * b)
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

pub fn add(x: &mut Tensor<f32>, y: &Tensor<f32>) {
    assert!(
        x.shape() == y.shape(),
        "x={:?}, y={:?}",
        x.shape(),
        y.shape()
    );

    unsafe {
        x.data_mut()
            .iter_mut()
            .zip(y.data())
            .for_each(|(xe, ye)| *xe = *xe + *ye);
    }
}

pub fn mul(x: &mut Tensor<f32>, scale: f32) {
    unsafe {
        x.data_mut().iter_mut().for_each(|xe| *xe = *xe * scale);
    }
}

pub fn scaled_dot_prod_qk(
    att_scores: &mut Tensor<f32>, // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,              // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,              // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    //score = Q @ K.T / sqrt(dim)
    eprintln!("attention scores {}", att_scores);
    eprintln!("attention q {}", q);
    eprintln!("attention k {}", k);
    for h in 0..n_kv_h {
        for g in 0..n_groups {
            for t in 0..seq_len {
                let q_off = t * n_kv_h * n_groups * dqkv + h * n_groups * dqkv + g * dqkv;
                for r in 0..total_seq_len {
                    let k_off = r * n_kv_h * dqkv + h * dqkv;
                    let att_off = h * n_groups * seq_len * total_seq_len
                        + g * seq_len * total_seq_len
                        + t * total_seq_len
                        + r;

                    //OP::matmul_transb(att_scores, 0.0, q, k, 1.0/(q.shape().len() as f32).sqrt());
                    unsafe {
                        att_scores.data_mut()[att_off] = q.data()[q_off..q_off + dqkv]
                            .iter()
                            .zip(k.data()[k_off..k_off + dqkv].iter())
                            //.inspect(|(x, y)|println!("[{}]@[{},{},{},{}] {} * {}", att_off, h, g, t, r, x, y))
                            .fold(0.0f32, |acc, (x, y)| acc + x * y)
                            / (dqkv as f32).sqrt();
                    }
                }
            }
        }
    }
}

pub fn scaled_dot_prod_attn(
    attn: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    scores: &Tensor<f32>,   // (n_kv_h, n_groups, seq, total_seq)
    v: &Tensor<f32>,        // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    //x = attn @ V
    eprintln!("attention V, scores {}", scores);
    eprintln!("attention V, V {}", v);
    for t in 0..seq_len {
        for h in 0..n_kv_h {
            for g in 0..n_groups {
                let score_off = h * n_groups * seq_len * total_seq_len
                    + g * seq_len * total_seq_len
                    + t * total_seq_len;
                for d in 0..dqkv {
                    let v_step = n_kv_h * dqkv;
                    let attn_off =
                        t * n_kv_h * n_groups * dqkv + h * n_groups * dqkv + g * dqkv + d;

                    //OP::matmul_transb(&mut x, 0.0, &attn, v, 1.0);
                    unsafe {
                        attn.data_mut()[attn_off] = scores.data()
                            [score_off..score_off + total_seq_len]
                            .iter()
                            .zip(v.data()[h * dqkv + d..].iter().step_by(v_step))
                            .fold(0.0f32, |acc, (x, y)| acc + x * y);
                    }
                }
            }
        }
    }
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

/*
att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
n_kv_h: usize,
n_groups: usize,
seq_len: usize,
total_seq_len: usize,
dqkv: usize,
*/
#[test]
fn test_scaled_dot_prod_qk_1() {
    let n_kv_h = 1usize;
    let n_groups = 1usize;
    let seq_len = 1usize;
    let total_seq_len = 1usize;
    let dpkv = 1;
    let mut att_scores =
        Tensor::<f32>::new(vec![1.1], &vec![n_kv_h, n_groups, seq_len, total_seq_len]);
    let q = Tensor::<f32>::new(vec![2.2], &vec![seq_len, n_kv_h * n_groups * dpkv]);
    let k = Tensor::<f32>::new(vec![3.3], &vec![total_seq_len, n_kv_h * dpkv]);
    scaled_dot_prod_qk(
        &mut att_scores,
        &q,
        &k,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dpkv,
    );
    assert!(att_scores.close_to(
        &Tensor::<f32>::new(
            vec![2.2 * 3.3],
            &vec![n_kv_h, n_groups, seq_len, total_seq_len]
        ),
        1e-3
    ));
}

/*
att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
n_kv_h: usize,
n_groups: usize,
seq_len: usize,
total_seq_len: usize,
dqkv: usize,
*/
#[test]
fn test_scaled_dot_prod_qk_2() {
    let n_kv_h = 1usize;
    let n_groups = 1usize;
    let seq_len = 1usize;
    let total_seq_len = 2usize;
    let dpkv = 1;
    let mut att_scores = Tensor::<f32>::new(
        vec![0.0; 2],
        &vec![n_kv_h, n_groups, seq_len, total_seq_len],
    );
    let q = Tensor::<f32>::new(vec![3.3], &vec![seq_len, n_kv_h * n_groups * dpkv]);
    let k = Tensor::<f32>::new(vec![4.4, 5.5], &vec![total_seq_len, n_kv_h * dpkv]);
    scaled_dot_prod_qk(
        &mut att_scores,
        &q,
        &k,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dpkv,
    );
    let target = Tensor::<f32>::new(
        vec![3.3 * 4.4, 3.3 * 5.5],
        &vec![n_kv_h, n_groups, seq_len, total_seq_len],
    );
    assert!(
        att_scores.close_to(&target, 1e-3),
        "result={:?}\ntarget={:?}",
        att_scores.data(),
        target.data()
    );
}

/*
att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
n_kv_h: usize,
n_groups: usize,
seq_len: usize,
total_seq_len: usize,
dqkv: usize,
*/
#[test]
fn test_scaled_dot_prod_qk_23() {
    let n_kv_h = 1usize;
    let n_groups = 1usize;
    let seq_len = 2usize;
    let total_seq_len = 3usize;
    let dpkv = 1;
    let mut att_scores = Tensor::<f32>::new(
        vec![0.0; 6],
        &vec![n_kv_h, n_groups, seq_len, total_seq_len],
    );
    let q = Tensor::<f32>::new(vec![2.2, 3.3], &vec![seq_len, n_kv_h * n_groups * dpkv]);
    let k = Tensor::<f32>::new(vec![4.4, 5.5, 6.6], &vec![total_seq_len, n_kv_h * dpkv]);
    scaled_dot_prod_qk(
        &mut att_scores,
        &q,
        &k,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dpkv,
    );
    assert!(att_scores.close_to(
        &Tensor::<f32>::new(
            vec![
                2.2 * 4.4,
                2.2 * 5.5,
                2.2 * 6.6,
                3.3 * 4.4,
                3.3 * 5.5,
                3.3 * 6.6
            ],
            &vec![n_kv_h, n_groups, seq_len, total_seq_len]
        ),
        1e-3
    ));
}

/*
att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
*/
#[test]
fn test_scaled_dot_prod_qk_2223_2() {
    let n_kv_h = 2usize;
    let n_groups = 2usize;
    let seq_len = 2usize;
    let total_seq_len = 3usize;
    let dqkv = 2;
    let mut att_scores = Tensor::<f32>::new(
        vec![0.0; 24],
        &vec![n_kv_h, n_groups, seq_len, total_seq_len],
    );
    let q = Tensor::<f32>::new(
        vec![
            11.11, 11.12, 11.21, 11.22, 12.11, 12.12, 12.21, 12.22, 21.11, 21.12, 21.21, 21.22,
            22.11, 22.12, 22.21, 22.22,
        ],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );
    let k = Tensor::<f32>::new(
        vec![
            1.11, 1.12, 1.21, 1.22, 2.11, 2.12, 2.21, 2.22, 3.11, 3.12, 3.21, 3.22,
        ],
        &vec![total_seq_len, n_kv_h * dqkv],
    );
    scaled_dot_prod_qk(
        &mut att_scores,
        &q,
        &k,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );
    let mut target = Tensor::<f32>::new(
        vec![
            11.11 * 1.11 + 11.12 * 1.12,
            11.11 * 2.11 + 11.12 * 2.12,
            11.11 * 3.11 + 11.12 * 3.12,
            21.11 * 1.11 + 21.12 * 1.12,
            21.11 * 2.11 + 21.12 * 2.12,
            21.11 * 3.11 + 21.12 * 3.12,
            11.21 * 1.11 + 11.22 * 1.12,
            11.21 * 2.11 + 11.22 * 2.12,
            11.21 * 3.11 + 11.22 * 3.12,
            21.21 * 1.11 + 21.22 * 1.12,
            21.21 * 2.11 + 21.22 * 2.12,
            21.21 * 3.11 + 21.22 * 3.12,
            12.11 * 1.21 + 12.12 * 1.22,
            12.11 * 2.21 + 12.12 * 2.22,
            12.11 * 3.21 + 12.12 * 3.22,
            22.11 * 1.21 + 22.12 * 1.22,
            22.11 * 2.21 + 22.12 * 2.22,
            22.11 * 3.21 + 22.12 * 3.22,
            12.21 * 1.21 + 12.22 * 1.22,
            12.21 * 2.21 + 12.22 * 2.22,
            12.21 * 3.21 + 12.22 * 3.22,
            22.21 * 1.21 + 22.22 * 1.22,
            22.21 * 2.21 + 22.22 * 2.22,
            22.21 * 3.21 + 22.22 * 3.22,
        ],
        &vec![n_kv_h, n_groups, seq_len, total_seq_len],
    );
    mul(&mut target, 1.0 / (dqkv as f32).sqrt());
    assert!(
        att_scores.close_to(&target, 1e-3),
        "result={:?}\ntarget={:?}",
        att_scores.data(),
        target.data()
    );
}

#[test]
fn test_close() {
    let x = Tensor::<f32>::new(
        vec![1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7],
        &vec![8],
    );
    assert!(x.close_to(
        &Tensor::<f32>::new(
            vec![1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7],
            &vec![8]
        ),
        1e-3
    ));

    assert!(x.close_to(
        &Tensor::<f32>::new(
            vec![1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7],
            &vec![8]
        ),
        1e-3
    ));
}
/*
attn: &mut Tensor<f32>,          // (seq, n_kv_h * n_groups * dqkv)
scores: &Tensor<f32>,            // (n_kv_h, n_groups, seq, total_seq)
v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
*/
#[test]
fn test_scaled_dot_prod_attn_2223_2() {
    let n_kv_h = 2usize;
    let n_groups = 2usize;
    let seq_len = 2usize;
    let total_seq_len = 3usize;
    let dqkv = 2;
    let mut attn = Tensor::<f32>::new(vec![0.0; 16], &vec![seq_len, n_kv_h * n_groups * dqkv]);
    let scores = Tensor::<f32>::new(
        vec![
            11.11, 11.12, 11.13, 11.21, 11.22, 11.23, 12.11, 12.12, 12.13, 12.21, 12.22, 12.23,
            21.11, 21.12, 21.13, 21.21, 21.22, 21.23, 22.11, 22.12, 22.13, 22.21, 22.22, 22.23,
        ],
        &vec![n_kv_h, n_groups, seq_len, total_seq_len],
    );
    let v = Tensor::<f32>::new(
        vec![
            1.11, 1.12, 1.21, 1.22, 2.11, 2.12, 2.21, 2.22, 3.11, 3.12, 3.21, 3.22,
        ],
        &vec![total_seq_len, n_kv_h * dqkv],
    );
    scaled_dot_prod_attn(
        &mut attn,
        &scores,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );
    let target = Tensor::<f32>::new(
        vec![
            11.11 * 1.11 + 11.12 * 2.11 + 11.13 * 3.11,
            11.11 * 1.12 + 11.12 * 2.12 + 11.13 * 3.12,
            12.11 * 1.11 + 12.12 * 2.11 + 12.13 * 3.11,
            12.11 * 1.12 + 12.12 * 2.12 + 12.13 * 3.12,
            //--
            21.11 * 1.21 + 21.12 * 2.21 + 21.13 * 3.21,
            21.11 * 1.22 + 21.12 * 2.22 + 21.13 * 3.22,
            22.11 * 1.21 + 22.12 * 2.21 + 22.13 * 3.21,
            22.11 * 1.22 + 22.12 * 2.22 + 22.13 * 3.22,
            11.21 * 1.11 + 11.22 * 2.11 + 11.23 * 3.11,
            11.21 * 1.12 + 11.22 * 2.12 + 11.23 * 3.12,
            12.21 * 1.11 + 12.22 * 2.11 + 12.23 * 3.11,
            12.21 * 1.12 + 12.22 * 2.12 + 12.23 * 3.12,
            //--
            21.21 * 1.21 + 21.22 * 2.21 + 21.23 * 3.21,
            21.21 * 1.22 + 21.22 * 2.22 + 21.23 * 3.22,
            22.21 * 1.21 + 22.22 * 2.21 + 22.23 * 3.21,
            22.21 * 1.22 + 22.22 * 2.22 + 22.23 * 3.22,
        ],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );
    assert!(
        attn.close_to(&target, 1e-3),
        "result={:?}\ntarget={:?}",
        attn.data(),
        target.data()
    );
}
