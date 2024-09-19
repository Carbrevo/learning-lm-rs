#![allow(dead_code)]
#![allow(unused)]

use std::fs::File;
use std::{default, vec};

use crate::config::LlamaConfigJson;
use crate::ctxgen::{GenerateIterator, Generator};
use crate::kvcache::{self, KVCache};
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use std::path::Path;
use std::fmt:: { Debug };
pub struct Llama<T> 
where T: Debug 
{
    vocab: usize,           // vocab size
    pub(crate) n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    pub(crate) n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    pub(crate) dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    pub(crate) eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            //todo!("self_attention(...)");
            self_attention(&mut hidden_states, &mut att_scores, &q, &full_k, &full_v, &self.params.wo[layer],
                            self.n_kv_h, n_groups, seq_len, total_seq_len, self.dqkv);
                            eprintln!("L{}: after attention:\nhidden_states {}", layer, hidden_states);
            //todo!("down_proj matmul and add residual");
            //residual = x + residual    
            OP::add(&mut residual, &mut hidden_states);

            //todo!("mlp(...)");
            mlp(&mut residual, &mut hidden_states, &mut gate_buf, &mut up_buf, 
                &self.params.w_up[layer], &self.params.w_down[layer], 
                &self.params.w_gate[layer], &self.params.rms_ffn_w[layer], self.eps);
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let mut residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            residual.reshape(&vec![1, residual.shape()[0]]),
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    attn: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    wo: &Tensor<f32>,
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    //todo!("Implement self_attention");

    //x = rms_norm(residual)
    //Q = RoPE(x @ Q_weight.T)
    //K = RoPE(x @ K_weight.T)
    //V = x @ V_weight.T
    //K = cat(K_cache, K)
    //V = cat(V_cache, V)

    //### 以下是你需要实现的部分
    assert!(hidden_states.shape() == &vec![ seq_len, n_kv_h * n_groups * dqkv]);
    assert!(attn.shape() == &vec![ n_kv_h, n_groups, seq_len, total_seq_len]);
    assert!(q.shape() == &vec![ seq_len, n_kv_h * n_groups, dqkv ],
            "q {} \n [{}, {}, {}]", q, seq_len, n_kv_h * n_groups, dqkv );
    assert!(k.shape() == &vec![ total_seq_len, n_kv_h * dqkv ], 
            "k {} \n [ {}, {} ]", k, total_seq_len, n_kv_h * dqkv);
    assert!(v.shape() == &vec![ total_seq_len, n_kv_h * dqkv ]);

    //score = Q @ K.T / sqrt(dim)
    OP::scaled_dot_prod_qk(attn, q, k, n_kv_h, n_groups, seq_len, total_seq_len, dqkv);

    //attn = softmax(score)
    //TODO: Is it right?
    OP::masked_softmax(attn);
    
    //hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    //att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    //q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    //k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    //v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    //x = attn @ V
    let mut attn_x = Tensor::default(hidden_states.shape());
    OP::scaled_dot_prod_attn(&mut attn_x, attn, v, n_kv_h, n_groups, seq_len, total_seq_len, dqkv);


    //x = x @ O_weight.T
    OP::matmul_transb(hidden_states, 0.0, &attn_x, wo, 1.0);    
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    //todo!("Implement mlp");

    //hidden = rms_norm(residual)
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    //gate = hidden @ gate_weight.T
    OP::matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    //up = hidden @ up_weight.T
    OP::matmul_transb(up, 0.0, hidden_states, w_up, 1.0);
    //itermediate = gate * sigmoid(gate) * up ## silu
    let mut intermediate = up;
    OP::silu(&mut intermediate, gate);
    //output = itermediate @ down_weight.T
    let mut output = Tensor::<f32>::default(residual.shape());
    OP::matmul_transb(&mut output, 0.0, &intermediate, w_down, 1.0);
    //residual = output + residual
    OP::add(residual, &output);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
