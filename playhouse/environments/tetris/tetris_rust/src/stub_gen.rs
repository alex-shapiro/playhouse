use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();
    let stub = tetris_rust::stub_info()?;
    stub.generate()?;
    Ok(())
}
