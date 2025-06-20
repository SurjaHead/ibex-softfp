# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Generate a baremetal application

# Name of the program $(PROGRAM).c will be added as a source file
PROGRAM = relu_test
PROGRAM_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# Any extra source files to include in the build. Use the upper case .S
# extension for assembly files
SOFTFP_DIR := $(PROGRAM_DIR)/../../../../softfp-2018-09-15
EXTRA_SRC_DIRS := $(SOFTFP_DIR)
EXTRA_SRCS := softfp_all.c softfp_wrappers.c softfp_support.c activations_softfp.c

# Link with the math library
LIBS = -lm -lgcc

# Add include paths
PROGRAM_CFLAGS += -I$(PROGRAM_DIR)/../../../.. \
                 -I$(PROGRAM_DIR)/../../common \
                 -I$(PROGRAM_DIR)/../common \
                 -I$(SOFTFP_DIR)

include ${PROGRAM_DIR}/../common/common.mk
