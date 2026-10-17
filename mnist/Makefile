# Testbench runner for the MNIST RTL, using Verilator. Run from mnist/.
#
#   make                    build and run every testbench in tb/
#   make convolve_3x3_fsm   build and run one testbench (tb/<name>_tb.sv)
#   make vectors            regenerate the test vectors in tb/vectors
#   make clean              delete the build output in target/

PYTHON ?= python3
BUILD  := target

RTL := $(wildcard rtl/*.sv)
TBS := $(patsubst tb/%_tb.sv,%,$(wildcard tb/*_tb.sv))

# Warnings still print, they just don't stop the build
VERILATOR_FLAGS := --binary -j 0 -Wno-fatal -Wno-DECLFILENAME

.PHONY: all vectors clean $(TBS)

all: $(TBS)

# Run target/<name>/sim, rebuilding it first if the tb or any RTL changed
$(TBS): %: $(BUILD)/%/sim
	./$<

$(BUILD)/%/sim: tb/%_tb.sv $(RTL)
	verilator $(VERILATOR_FLAGS) --top-module $*_tb -Mdir $(BUILD)/$* -o sim $< $(RTL)

vectors:
	$(PYTHON) scripts/convolve_3x3_vectors.py
	$(PYTHON) scripts/winograd_3x3_vectors.py

clean:
	rm -rf $(BUILD)
