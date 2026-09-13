use candle::quantized::gguf_file;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect_gguf <path_to_gguf>");
        std::process::exit(1);
    }
    let path = &args[1];
    println!("Inspecting: {}", path);
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;

    println!("\n=== Metadata ({} entries) ===", content.metadata.len());
    for (k, v) in &content.metadata {
        println!("  {k} = {:?}", v);
    }

    println!("\n=== Tensors ({} total) ===", content.tensor_infos.len());
    for (k, v) in &content.tensor_infos {
        if k.starts_with("blk.0.") || k.starts_with("blk.1.") || k.starts_with("token_") || k.starts_with("output") {
            println!("  {k}: shape={:?}, dtype={:?}", v.shape, v.ggml_dtype);
        }
    }

    Ok(())
}
