use std::mem::{ size_of, };
use std::fmt::Debug;
use crate::config::LlamaConfigJson;
use crate::tensor::{ Tensor, };
use safetensors::{ SafeTensors, View };

fn attn(layer:i32, field: &str) -> String {
    format!("model.layers.{}.self_attn.{}_proj.weight", layer, field.strip_prefix("w").unwrap())
}

fn mlp(layer:i32, field: &str) -> String {
    format!("model.layers.{}.mlp.{}_proj.weight", layer, field.strip_prefix("w_").unwrap())
}

fn layernorm(layer:i32, field: &str) -> String {
    let name = match field {
        "rms_att_w" => "input",
        "rms_ffn_w" => "post_attention",
        _ => panic!("Invalid field '{}'", field),
    };
    format!("model.layers.{}.{}_layernorm.weight", layer, name)
}

macro_rules! hidden_tensors {
    ($safetensor:ident, $class:ident, $field:ident, $ttype:ty, $config:ident) => {
        {
            let mut tensors = Vec::<Tensor<$ttype>>::new();
            
            for layer in 0..$config.num_hidden_layers as i32 {
                let tensorview = $safetensor.tensor($class(layer, stringify!($field)).as_str()).unwrap();
                let blen = View::data_len(&tensorview);
                let tensor_data = (0..blen/size_of::<$ttype>())
                                            .map(|x|{ let data = tensorview.data(); Convert::<$ttype>::convert(&data[x*size_of::<$ttype>()..(x+1)*size_of::<$ttype>()]) })
                                            .collect();
                tensors.push(Tensor::<$ttype>::new(tensor_data, &Vec::from(tensorview.shape())));                
            }
            tensors
        }
    }
}

macro_rules! norm_tensor {
    ($safetensor:ident, $ttype:ty, $config:ident) => {
        {
            let name = "model.norm.weight";
            
            let tensorview = $safetensor.tensor(name)
                                //.inspect(|x|println!("{}.shape={:?}", name, x.shape()))
                                .unwrap();
            let blen = View::data_len(&tensorview);
            let tensor_data = (0..blen/size_of::<$ttype>())
                                            .map(|x|{ let data = tensorview.data(); Convert::<$ttype>::convert(&data[x*size_of::<$ttype>()..(x+1)*size_of::<$ttype>()]) })
                                            .collect();
            Tensor::<$ttype>::new(tensor_data, &Vec::from(tensorview.shape()))
        }
    }
}

macro_rules! embedding_tensor {
    ($safetensor:ident, $class:ident, $ttype:ty, $config:ident) => {
        {
            let name = format!("{}.weight", stringify!($class));
            
            let tensorview = $safetensor.tensor(&name).or_else(|e|{
                if $config.tie_word_embeddings {
                    let name = format!("{}.weight", ["lm_head", "embedding_table"].iter().find(|x|{**x!=stringify!($class)}).unwrap());
                    $safetensor.tensor(&name)
                } else {
                    Err(e)
                }
            }).unwrap();

            let blen = View::data_len(&tensorview);
            let tensor_data = (0..blen/size_of::<$ttype>())
                                            .map(|x|{ let data = tensorview.data(); Convert::<$ttype>::convert(&data[x*size_of::<$ttype>()..(x+1)*size_of::<$ttype>()]) })
                                            .collect();
            Tensor::<$ttype>::new(tensor_data, &Vec::from(tensorview.shape()))    
        }
    }
}

pub trait Convert<T> {
    fn convert(&self ) -> T;
}

impl Convert<f32> for [u8] {
    fn convert(&self ) -> f32 {
        f32::from_le_bytes([self[0], self[1], self[2], self[3]])
    }
}

pub struct LLamaParams<T> 
where T: Debug 
{
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl<T: Copy + Clone + Default + Debug> LLamaParams<T> 
where [u8]: Convert<T>
{
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //todo!("实现从safetensors文件的模型参数加载");

        // /*
        for (name, view) in safetensor.tensors() {
            eprintln!("'{}' = {:?}, {:?}", name, view.dtype(), view.shape());
        }
        eprintln!("{:?}", config);
        //*/

        LLamaParams::<T> {
             embedding_table: embedding_tensor!(safetensor, embedding_table, T, config),
             rms_att_w: hidden_tensors!(safetensor, layernorm, rms_att_w, T, config),
             wq: hidden_tensors!(safetensor, attn, wq, T, config),
             wk: hidden_tensors!(safetensor, attn, wk, T, config),
             wv: hidden_tensors!(safetensor, attn, wv, T, config),
             wo: hidden_tensors!(safetensor, attn, wo, T, config),

             rms_ffn_w: hidden_tensors!(safetensor, layernorm, rms_ffn_w, T, config),
             w_up: hidden_tensors!(safetensor, mlp, w_up, T, config),
             w_gate: hidden_tensors!(safetensor, mlp, w_gate, T, config),
             w_down: hidden_tensors!(safetensor, mlp, w_down, T, config),
             rms_out_w: norm_tensor!(safetensor, T, config),
             lm_head: embedding_tensor!(safetensor, lm_head, T, config),
        }

    }
}
