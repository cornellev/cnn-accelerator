use std::path::Path;

use marlin::{
    verilator::{VerilatorRuntime, VerilatorRuntimeOptions},
    verilog::prelude::*,
};
use snafu::Whatever;

#[verilog(src = "src/verilog/adder.sv", name = "adder", params = { N: 4 })]
pub struct Adder;

impl<'ctx> Adder<'ctx> {
    fn tick(&mut self) {
        self.clk = 1;
        self.eval();
        self.clk = 0;
        self.eval();
    }
}

fn make_runtime() -> Result<VerilatorRuntime, Whatever> {
    VerilatorRuntime::new2(
        "build",
        &["src/verilog/adder.sv"],
        &[Path::new("src/verilog/")],
        [],
        VerilatorRuntimeOptions::default(),
    )
}

#[test]
#[snafu::report]
fn test_adds_numbers() -> Result<(), Whatever> {
    let runtime = make_runtime()?;
    let mut dut = runtime.create_model_simple::<Adder>()?;

    dut.clk = 0;
    dut.eval();

    dut.a = 10;
    dut.b = 5;
    dut.tick();

    assert_eq!(dut.c, 15);

    Ok(())
}
