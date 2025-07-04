// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/**
 * Floating Point Unit (FPU) for Ibex
 * 
 * Implements IEEE 754 single precision floating point addition and multiplication
 * for iterative activation calculations. This is a simplified FPU implementation
 * focused on the core operations needed for neural network computations.
 */

`include "prim_assert.sv"

module ibex_fpu #(
  parameter int unsigned DataWidth = 32
) (
  input  logic                     clk_i,      // unused - FPU is combinational
  input  logic                     rst_ni,     // unused - FPU is combinational

  // Control signals
  input  logic                     fpu_en_i,           // Enable FPU operation
  input  ibex_pkg::fpu_op_e        fpu_operator_i,     // FPU operation
  
  // Operands
  input  logic [DataWidth-1:0]     operand_a_i,        // First operand
  input  logic [DataWidth-1:0]     operand_b_i,        // Second operand
  
  // Result
  output logic [DataWidth-1:0]     result_o,           // FPU result
  output logic                     valid_o,            // Result valid
  
  // Status flags
  output logic                     overflow_o,         // Overflow flag
  output logic                     underflow_o,        // Underflow flag
  output logic                     invalid_o           // Invalid operation flag
);

  import ibex_pkg::*;

  // IEEE 754 Single Precision Format - constants now inlined for clarity
  
  // Unused ports (FPU is combinational)
  logic unused_clk, unused_rst_n;
  assign unused_clk = clk_i;
  assign unused_rst_n = rst_ni;

  // Internal signals
  logic        sign_a, sign_b, sign_result;
  logic [7:0]  exp_a, exp_b, exp_result;
  logic [22:0] mant_a, mant_b, mant_result;
  logic [23:0] mant_a_norm, mant_b_norm;  // Normalized mantissas with implicit 1
  /* verilator lint_off UNUSEDSIGNAL */
  logic [47:0] mult_result;               // Multiplication result (lower 23 bits unused)
  logic [8:0]  exp_add_temp, exp_mult_temp;  // 9-bit for overflow detection (MSB unused in normal cases)
  /* verilator lint_on UNUSEDSIGNAL */
  logic [24:0] add_result;                // Addition result
  logic [7:0]  exp_add, exp_mult;
  logic        overflow, underflow, invalid;
  
  // Extract fields from operands
  assign sign_a = operand_a_i[31];
  assign exp_a  = operand_a_i[30:23];
  assign mant_a = operand_a_i[22:0];
  
  assign sign_b = operand_b_i[31];
  assign exp_b  = operand_b_i[30:23];
  assign mant_b = operand_b_i[22:0];
  
  // Add implicit leading 1 for normalized numbers
  assign mant_a_norm = (exp_a != 8'h00) ? {1'b1, mant_a} : {1'b0, mant_a};
  assign mant_b_norm = (exp_b != 8'h00) ? {1'b1, mant_b} : {1'b0, mant_b};
  
  // Special case detection
  logic zero_a, zero_b, inf_a, inf_b, nan_a, nan_b;
  
  assign zero_a = (exp_a == 8'h00) && (mant_a == 23'h0);
  assign zero_b = (exp_b == 8'h00) && (mant_b == 23'h0);
  assign inf_a  = (exp_a == 8'hFF) && (mant_a == 23'h0);
  assign inf_b  = (exp_b == 8'hFF) && (mant_b == 23'h0);
  assign nan_a  = (exp_a == 8'hFF) && (mant_a != 23'h0);
  assign nan_b  = (exp_b == 8'hFF) && (mant_b != 23'h0);
  
  // FPU operation logic
  always_comb begin
    // Default values
    sign_result = 1'b0;
    exp_result  = 8'h00;
    mant_result = 23'h0;
    overflow    = 1'b0;
    underflow   = 1'b0;
    invalid     = 1'b0;
    
    if (fpu_en_i) begin
      case (fpu_operator_i)
        FPU_OP_ADD: begin
          // Floating point addition
          if (nan_a || nan_b) begin
            // NaN result
            sign_result = 1'b0;
            exp_result  = 8'hFF;
            mant_result = 23'h400000; // Quiet NaN
            invalid     = 1'b1;
          end else if (inf_a || inf_b) begin
            // Infinity cases
            if (inf_a && inf_b && (sign_a != sign_b)) begin
              // Inf - Inf = NaN
              sign_result = 1'b0;
              exp_result  = 8'hFF;
              mant_result = 23'h400000; // Quiet NaN
              invalid     = 1'b1;
            end else if (inf_a) begin
              sign_result = sign_a;
              exp_result  = 8'hFF;
              mant_result = 23'h0;
            end else begin
              sign_result = sign_b;
              exp_result  = 8'hFF;
              mant_result = 23'h0;
            end
          end else if (zero_a && zero_b) begin
            // 0 + 0 = 0
            sign_result = sign_a & sign_b;
            exp_result  = 8'h00;
            mant_result = 23'h0;
          end else if (zero_a) begin
            // 0 + b = b
            sign_result = sign_b;
            exp_result  = exp_b;
            mant_result = mant_b;
          end else if (zero_b) begin
            // a + 0 = a
            sign_result = sign_a;
            exp_result  = exp_a;
            mant_result = mant_a;
          end else begin
            // Normal addition - simplified implementation
            if (exp_a >= exp_b) begin
              exp_add = exp_a;
              if (sign_a == sign_b) begin
                add_result = {1'b0, mant_a_norm} + ({1'b0, mant_b_norm} >> (exp_a - exp_b));
                sign_result = sign_a;
              end else begin
                add_result = {1'b0, mant_a_norm} - ({1'b0, mant_b_norm} >> (exp_a - exp_b));
                sign_result = sign_a;
              end
            end else begin
              exp_add = exp_b;
              if (sign_a == sign_b) begin
                add_result = {1'b0, mant_b_norm} + ({1'b0, mant_a_norm} >> (exp_b - exp_a));
                sign_result = sign_b;
              end else begin
                add_result = {1'b0, mant_b_norm} - ({1'b0, mant_a_norm} >> (exp_b - exp_a));
                sign_result = sign_b;
              end
            end
            
            // Normalize result
            if (add_result[24]) begin
              exp_add_temp = {1'b0, exp_add} + 9'd1;
              exp_result = exp_add_temp[7:0];
              mant_result = add_result[23:1];
            end else if (add_result[23]) begin
              exp_result = exp_add;
              mant_result = add_result[22:0];
            end else begin
              // Result needs left normalization (not implemented in this simple version)
              exp_result = exp_add;
              mant_result = add_result[22:0];
            end
            
            // Check for overflow
            if (exp_result >= 8'hFF) begin
              overflow = 1'b1;
              exp_result = 8'hFF;
              mant_result = 23'h0;
            end
          end
        end
        
        FPU_OP_MUL: begin
          // Floating point multiplication
          if (nan_a || nan_b) begin
            // NaN result
            sign_result = 1'b0;
            exp_result  = 8'hFF;
            mant_result = 23'h400000; // Quiet NaN
            invalid     = 1'b1;
          end else if ((inf_a && zero_b) || (zero_a && inf_b)) begin
            // Inf * 0 = NaN
            sign_result = 1'b0;
            exp_result  = 8'hFF;
            mant_result = 23'h400000; // Quiet NaN
            invalid     = 1'b1;
          end else if (inf_a || inf_b) begin
            // Infinity result
            sign_result = sign_a ^ sign_b;
            exp_result  = 8'hFF;
            mant_result = 23'h0;
          end else if (zero_a || zero_b) begin
            // Zero result
            sign_result = sign_a ^ sign_b;
            exp_result  = 8'h00;
            mant_result = 23'h0;
          end else begin
            // Normal multiplication
            sign_result = sign_a ^ sign_b;
            
            // Multiply mantissas
            mult_result = mant_a_norm * mant_b_norm;
            
            // Add exponents and subtract bias
            exp_mult_temp = {1'b0, exp_a} + {1'b0, exp_b} - 9'd127;
            exp_mult = exp_mult_temp[7:0];
            
            // Normalize result
            if (mult_result[47]) begin
              exp_add_temp = {1'b0, exp_mult} + 9'd1;
              exp_result = exp_add_temp[7:0];
              mant_result = mult_result[46:24];
            end else begin
              exp_result = exp_mult;
              mant_result = mult_result[45:23];
            end
            
            // Check for overflow/underflow
            if (exp_result >= 8'hFF) begin
              overflow = 1'b1;
              exp_result = 8'hFF;
              mant_result = 23'h0;
            end else if (exp_result == 8'h00) begin
              underflow = 1'b1;
              exp_result = 8'h00;
              mant_result = 23'h0;
            end
          end
        end
        
        default: begin
          // Invalid operation
          sign_result = 1'b0;
          exp_result  = 8'h00;
          mant_result = 23'h0;
          invalid     = 1'b1;
        end
      endcase
    end else begin
      // FPU disabled
      sign_result = 1'b0;
      exp_result  = 8'h00;
      mant_result = 23'h0;
    end
  end
  
  // Output assignment
  assign result_o = {sign_result, exp_result, mant_result};
  
  // Status flags
  assign overflow_o  = overflow;
  assign underflow_o = underflow;
  assign invalid_o   = invalid;
  
  // Valid signal - for this simple implementation, result is always valid in the same cycle
  assign valid_o = fpu_en_i;

endmodule
