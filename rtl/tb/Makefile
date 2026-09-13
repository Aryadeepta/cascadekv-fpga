TOPLEVEL_LANG = verilog
SIM ?= icarus
TOPLEVEL = int4_dot16
COCOTB_TEST_MODULES = test_int4_dot16
VERILOG_SOURCES = $(abspath $(CURDIR)/../int4_dot16.sv)
PYTHONPATH := $(abspath $(CURDIR)/../..):$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim
