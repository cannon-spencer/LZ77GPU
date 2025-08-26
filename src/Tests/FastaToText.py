#!/usr/bin/env python3
import sys

if len(sys.argv) != 3:
    print("Usage: python fasta_to_txt.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]


num_head = 0
with open(input_file, "r") as f, open(output_file, "wb") as out_file:
    for line in f:
        if line.startswith(">"):
            if num_head == 0:
                num_head += 1
                continue
            # replace the title with null
            else:
                out_file.write(bytes("\x00", "utf-8"))
        else:
            # write the data
            out_file.write(bytes(line.strip().replace("\n", ""), "utf-8"))
    out_file.write(bytes("\x00", "utf-8"))