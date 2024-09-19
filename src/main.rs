mod config;
mod ctxgen;
mod kvcache;
mod logits;
mod model;
mod operators;
mod params;
mod tensor;

//use core::slice::SlicePattern;
use std::path::PathBuf;
use std::{io, io::Write};
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut ctxgen = ctxgen::Generator::new(&llama, &tokenizer, 512, 0.85, 5, 0.85);

    let input = "Once upon a time";
    let geniter = ctxgen.prompt(input);
    let mut spec_txt = String::new();
    #[allow(unused_assignments)]
    let mut insert_txt = String::new();
    for txt in geniter {
        if let Some(out_txt) = match txt {
            ref _t if _t == "<|" => {
                spec_txt = _t.clone();
                None
            }
            ref _t if _t == "|>" => {
                spec_txt += _t;

                match spec_txt.as_str() {
                    "<|end_story|>" => {
                        spec_txt.clear();
                        insert_txt = "".to_string();
                        Some(&insert_txt)
                    }
                    _ => {
                        insert_txt = spec_txt.clone();
                        spec_txt.clear();
                        Some(&insert_txt)
                    }
                }
            }
            ref _t if !spec_txt.is_empty() => {
                spec_txt += _t;
                None
            }
            ref _t => Some(_t),
        } {
            print!("{}", out_txt);
            let _ = io::stdout().flush();
        }
    }
}
