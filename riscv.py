"""
CSE 140 Project - Single-Cycle RISC-V CPU Simulator
Supports: lw, sw, add, addi, sub, and, andi, or, ori, beq, jal, jalr
"""

import sys

# ─────────────────────────────────────────────
# Register name mapping (ABI names)
# ─────────────────────────────────────────────
REG_NAMES = {
    0: "zero", 1: "ra",  2: "sp",  3: "gp",
    4: "tp",   5: "t0",  6: "t1",  7: "t2",
    8: "s0",   9: "s1",  10: "a0", 11: "a1",
    12: "a2",  13: "a3", 14: "a4", 15: "a5",
    16: "a6",  17: "a7", 18: "s2", 19: "s3",
    20: "s4",  21: "s5", 22: "s6", 23: "s7",
    24: "s8",  25: "s9", 26: "s10",27: "s11",
    28: "t3",  29: "t4", 30: "t5", 31: "t6",
}

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
pc = 0
next_pc = 0
branch_target = 0
alu_zero = 0
total_clock_cycles = 0

# Register file: 32 registers, all zero
rf = [0] * 32

# Data memory: 32 entries (each = 4 bytes), address 0x00–0x7C
d_mem = [0] * 32

# Control signals
RegWrite = 0
ALUSrc   = 0
MemWrite = 0
MemRead  = 0
MemToReg = 0
Branch   = 0
Jump     = 0   # for JAL
JumpR    = 0   # for JALR

# ALU control code
alu_ctrl = 0b0000

# Decoded instruction fields (globals shared across pipeline stages)
opcode   = 0
rd       = 0
rs1      = 0
rs2      = 0
funct3   = 0
funct7   = 0
imm      = 0
rs1_val  = 0
rs2_val  = 0
alu_result = 0
mem_data   = 0
pc_at_decode = 0   # PC value when instruction was fetched (needed for JAL/JALR wb)


# ─────────────────────────────────────────────
# Instruction memory (loaded from file)
# ─────────────────────────────────────────────
instructions = []


def load_program(filename: str):
    """Read binary instruction strings from file."""
    global instructions
    instructions = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                instructions.append(int(line, 2))


# ─────────────────────────────────────────────
# Helper: sign-extend a value from bit_width
# ─────────────────────────────────────────────
def sign_extend(value: int, bit_width: int) -> int:
    if value & (1 << (bit_width - 1)):
        value -= (1 << bit_width)
    return value


# ─────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────
def Fetch() -> int | None:
    """
    Read the instruction pointed to by pc.
    pc is a byte address; instruction index = pc // 4.
    Updates pc for the next cycle.
    Returns the 32-bit instruction word, or None if past end.
    """
    global pc, next_pc, branch_target, alu_zero, Branch, Jump, JumpR

    idx = pc // 4
    if idx >= len(instructions):
        return None

    instr = instructions[idx]

    next_pc = pc + 4

    # After execution, choose next PC:
    #   Branch taken  : branch_target
    #   JAL           : branch_target  (set by Execute)
    #   JALR          : branch_target  (set by Execute)
    #   Otherwise     : next_pc
    if Branch and alu_zero:
        pc = branch_target
    elif Jump or JumpR:
        pc = branch_target
    else:
        pc = next_pc

    return instr


# ─────────────────────────────────────────────
# CONTROL UNIT  (called inside Decode)
# ─────────────────────────────────────────────
def ALUControl(alu_op: int, f3: int, f7_bit: int) -> int:
    """
    Generate 4-bit alu_ctrl from ALUOp + funct3/funct7.
    ALUOp encoding:
        00 → ADD  (lw/sw)
        01 → SUB  (beq)
        10 → R-type / I-type arithmetic
        11 → reserved
    """
    if alu_op == 0b00:
        return 0b0010   # ADD
    if alu_op == 0b01:
        return 0b0110   # SUB
    # ALUOp == 10: decode via funct3 / funct7
    if f3 == 0b000:
        if f7_bit == 1:
            return 0b0110   # SUB
        return 0b0010       # ADD
    if f3 == 0b111:
        return 0b0000       # AND
    if f3 == 0b110:
        return 0b0001       # OR
    # addi / andi / ori (I-type) fall through here
    return 0b0010           # default ADD


def ControlUnit(op: int, f3: int, f7: int):
    """
    Set global control signals from 7-bit opcode.
    """
    global RegWrite, ALUSrc, MemWrite, MemRead, MemToReg
    global Branch, Jump, JumpR, alu_ctrl

    # Reset all signals
    RegWrite = ALUSrc = MemWrite = MemRead = MemToReg = 0
    Branch   = Jump   = JumpR   = 0

    f7_bit = (f7 >> 5) & 1   # bit 30 of instruction → funct7[5]

    if op == 0b0110011:       # R-type
        RegWrite = 1
        alu_ctrl = ALUControl(0b10, f3, f7_bit)

    elif op == 0b0010011:     # I-type arithmetic (addi, andi, ori)
        RegWrite = 1
        ALUSrc   = 1
        alu_ctrl = ALUControl(0b10, f3, 0)   # no funct7 for I-type

    elif op == 0b0000011:     # lw
        RegWrite = 1
        ALUSrc   = 1
        MemRead  = 1
        MemToReg = 1
        alu_ctrl = ALUControl(0b00, f3, 0)

    elif op == 0b0100011:     # sw
        ALUSrc   = 1
        MemWrite = 1
        alu_ctrl = ALUControl(0b00, f3, 0)

    elif op == 0b1100011:     # beq
        Branch   = 1
        alu_ctrl = ALUControl(0b01, f3, 0)

    elif op == 0b1101111:     # jal
        RegWrite = 1
        Jump     = 1
        alu_ctrl = 0b0010     # ADD (branch target calc)

    elif op == 0b1100111:     # jalr
        RegWrite = 1
        JumpR    = 1
        ALUSrc   = 1
        alu_ctrl = 0b0010     # ADD


# ─────────────────────────────────────────────
# DECODE
# ─────────────────────────────────────────────
def Decode(instr: int):
    """
    Decode a 32-bit instruction into global fields.
    Reads register values from rf[].
    Calls ControlUnit().
    """
    global opcode, rd, rs1, rs2, funct3, funct7
    global imm, rs1_val, rs2_val, pc_at_decode

    pc_at_decode = next_pc - 4   # the PC that fetched this instruction

    opcode = instr & 0x7F
    rd     = (instr >> 7)  & 0x1F
    funct3 = (instr >> 12) & 0x7
    rs1    = (instr >> 15) & 0x1F
    rs2    = (instr >> 20) & 0x1F
    funct7 = (instr >> 25) & 0x7F

    # Immediate decode by instruction type
    if opcode in (0b0010011, 0b0000011, 0b1100111):   # I-type
        imm = sign_extend((instr >> 20) & 0xFFF, 12)

    elif opcode == 0b0100011:   # S-type
        imm_11_5 = (instr >> 25) & 0x7F
        imm_4_0  = (instr >> 7)  & 0x1F
        imm = sign_extend((imm_11_5 << 5) | imm_4_0, 12)

    elif opcode == 0b1100011:   # B-type (beq)
        b12  = (instr >> 31) & 1
        b11  = (instr >> 7)  & 1
        b10_5= (instr >> 25) & 0x3F
        b4_1 = (instr >> 8)  & 0xF
        imm = sign_extend((b12 << 12) | (b11 << 11) | (b10_5 << 6) | (b4_1 << 1), 13)

    elif opcode == 0b1101111:   # J-type (jal)
        b20    = (instr >> 31) & 1
        b10_1  = (instr >> 21) & 0x3FF
        b11    = (instr >> 20) & 1
        b19_12 = (instr >> 12) & 0xFF
        imm = sign_extend(
            (b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1), 21
        )
    else:
        imm = 0

    rs1_val = rf[rs1]
    rs2_val = rf[rs2]

    ControlUnit(opcode, funct3, funct7)


# ─────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────
def Execute():
    """
    Perform ALU operation.
    Updates alu_result, alu_zero, branch_target.
    """
    global alu_result, alu_zero, branch_target

    operand_a = rs1_val
    operand_b = imm if ALUSrc else rs2_val

    # ALU operations
    if alu_ctrl == 0b0000:       # AND
        alu_result = operand_a & operand_b
    elif alu_ctrl == 0b0001:     # OR
        alu_result = operand_a | operand_b
    elif alu_ctrl == 0b0010:     # ADD
        alu_result = operand_a + operand_b
    elif alu_ctrl == 0b0110:     # SUB
        alu_result = operand_a - operand_b
    else:
        alu_result = operand_a + operand_b

    alu_zero = 1 if alu_result == 0 else 0

    # Branch target: PC + (imm << 1) — note: B-imm already has bit0=0
    # For beq: branch_target = PC_of_instruction + imm (imm encodes offset*2 already)
    # For jal: branch_target = PC_of_instruction + imm
    # For jalr: branch_target = (rs1_val + imm) & ~1
    if JumpR:
        branch_target = (rs1_val + imm) & ~1
    else:
        branch_target = pc_at_decode + imm


# ─────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────
def Mem() -> tuple[int, bool]:
    """
    Perform memory access.
    Returns (data_read, memory_was_written).
    """
    global d_mem
    addr = alu_result
    mem_idx = addr // 4

    if MemRead:   # lw
        return d_mem[mem_idx], False

    if MemWrite:  # sw
        d_mem[mem_idx] = rs2_val
        return 0, True

    return 0, False


# ─────────────────────────────────────────────
# WRITEBACK
# ─────────────────────────────────────────────
def Writeback(mem_read_data: int, mem_written: bool, mem_addr: int) -> list[str]:
    """
    Write result back to register file.
    Increments total_clock_cycles.
    Returns list of output strings for this cycle.
    """
    global rf, total_clock_cycles

    output_lines = []

    # Register writeback
    if RegWrite and rd != 0:
        if Jump or JumpR:
            # JAL / JALR: rd ← PC+4 (next sequential PC)
            write_val = pc_at_decode + 4
        elif MemToReg:
            write_val = mem_read_data
        else:
            write_val = alu_result

        rf[rd] = write_val
        output_lines.append(f"x{rd} ({REG_NAMES[rd]}) is modified to {hex(write_val & 0xFFFFFFFF)}")

    # Memory write output
    if mem_written:
        output_lines.append(f"memory {hex(alu_result)} is modified to {hex(rs2_val & 0xFFFFFFFF)}")

    total_clock_cycles += 1
    return output_lines


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run(filename: str, init_rf: dict = None, init_dmem: dict = None,
        abi_names: bool = True):
    """
    Load and run a program.
    init_rf  : {reg_index: value} overrides
    init_dmem: {byte_address: value} overrides
    """
    global pc, next_pc, branch_target, alu_zero, total_clock_cycles
    global rf, d_mem
    global RegWrite, ALUSrc, MemWrite, MemRead, MemToReg, Branch, Jump, JumpR

    # Reset state
    pc = 0; next_pc = 0; branch_target = 0; alu_zero = 0; total_clock_cycles = 0
    rf    = [0] * 32
    d_mem = [0] * 32
    RegWrite = ALUSrc = MemWrite = MemRead = MemToReg = 0
    Branch = Jump = JumpR = 0

    # Apply initial values
    if init_rf:
        for idx, val in init_rf.items():
            rf[idx] = val
    if init_dmem:
        for addr, val in init_dmem.items():
            d_mem[addr // 4] = val

    load_program(filename)

    while True:
        # ── Fetch ──────────────────────────────
        # Reset jump signals so they don't persist
        Branch = Jump = JumpR = 0

        instr = Fetch()
        if instr is None:
            break

        # ── Decode ─────────────────────────────
        Decode(instr)

        # ── Execute ────────────────────────────
        Execute()

        # ── Memory ─────────────────────────────
        mem_data_out, mem_written = Mem()
        mem_addr_out = alu_result

        # ── Writeback ──────────────────────────
        output = Writeback(mem_data_out, mem_written, mem_addr_out)

        # ── Print cycle results ─────────────────
        print(f"\ntotal_clock_cycles {total_clock_cycles} :")
        for line in output:
            print(f"  {line}")
        print(f"  pc is modified to {hex(pc)}")

    print(f"\nprogram terminated:")
    print(f"total execution time is {total_clock_cycles} cycles")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    filename = input("Enter the program file name to run:\n").strip()

    # Determine which part based on file name (part2 uses different init)
    if "part2" in filename.lower():
        # Part 2 register init: s0=0x20, a0=0x5, a1=0x2, a2=0xa, a3=0xf
        init_rf = {8: 0x20, 10: 0x5, 11: 0x2, 12: 0xa, 13: 0xf}
        init_dmem = {}
    else:
        # Part 1 register init: x1=0x20, x2=0x5, x10=0x70, x11=0x4
        # d_mem init: 0x70=0x5, 0x74=0x10
        init_rf   = {1: 0x20, 2: 0x5, 10: 0x70, 11: 0x4}
        init_dmem = {0x70: 0x5, 0x74: 0x10}

    run(filename, init_rf=init_rf, init_dmem=init_dmem)