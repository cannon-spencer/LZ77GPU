#!/usr/bin/env python3
# filepath: [verify_lz77.py](http://_vscodecontentref_/0)

"""
LZ77 Compression Verification Tool
Decompresses LZ77 binary files and verifies against original files.
"""

import struct
import sys

def decompress_lz77(lz77_file):
    """Decompress data from LZ77 binary file"""
    decompressed = bytearray()
    factor_count = 0
    
    print("Starting LZ77 decompression...")
    
    with open(lz77_file, 'rb') as f:
        while True:
            pos_data = f.read(8)  # size_t = 8 bytes
            len_data = f.read(8)  # size_t = 8 bytes
            
            if len(pos_data) != 8 or len(len_data) != 8:
                break
                
            pos = struct.unpack('<Q', pos_data)[0]  # unsigned long long
            length = struct.unpack('<Q', len_data)[0]
            
            factor_count += 1
            
            # Print first 20 factors for debugging
            if factor_count <= 20:
                print(f"Factor {factor_count}: pos={pos}, len={length}")
            
            if length == 0:
                # Literal value: position contains the actual byte
                char_value = pos & 0xFF
                decompressed.append(char_value)
                
                # Print literal character details for first 20 factors
                if factor_count <= 20:
                    if 32 <= char_value <= 126:  # Printable ASCII
                        print(f"  -> Added literal: '{chr(char_value)}' (byte: {char_value})")
                    else:
                        print(f"  -> Added literal: non-printable byte {char_value} (0x{char_value:02x})")
                        
            else:
                # Copy reference: copy 'length' bytes starting from 'pos'
                # Handle self-referencing patterns correctly
                start_pos = pos
                bytes_before_copy = len(decompressed)
                
                if factor_count <= 20:
                    print(f"  -> Copying {length} bytes from position {pos}")
                
                for i in range(length):
                    copy_index = start_pos + i
                    if copy_index < len(decompressed):
                        copied_byte = decompressed[copy_index]
                        decompressed.append(copied_byte)
                        
                        # Show first few copied bytes for first few factors
                        if factor_count <= 10 and i < 10:
                            if 32 <= copied_byte <= 126:
                                print(f"    Byte {i}: '{chr(copied_byte)}' from pos {copy_index}")
                            else:
                                print(f"    Byte {i}: 0x{copied_byte:02x} from pos {copy_index}")
                    else:
                        # This should not happen in valid LZ77 data
                        print(f"Warning: Invalid reference at position {copy_index}, current length {len(decompressed)}")
                        break
                
                # Show what was copied for first few factors
                if factor_count <= 20:
                    copied_bytes = decompressed[bytes_before_copy:bytes_before_copy + length]
                    if len(copied_bytes) > 0:
                        # Try to show as text if mostly printable
                        printable_text = ""
                        for b in copied_bytes[:30]:  # Show first 30 bytes max
                            if 32 <= b <= 126:
                                printable_text += chr(b)
                            else:
                                printable_text += f"\\x{b:02x}"
                        if len(copied_bytes) > 30:
                            printable_text += "..."
                        print(f"  -> Copied text: \"{printable_text}\"")
            
            # Print current decompressed content (first 100 characters) after each factor
            if len(decompressed) >= 1 and factor_count <= 50:
                current_text = ""
                char_count = 0
                
                for b in decompressed:
                    if char_count >= 100:  # Limit to first 100 characters
                        current_text += "..."
                        break
                    
                    if 32 <= b <= 126:  # Printable ASCII
                        current_text += chr(b)
                    else:
                        current_text += f"\\x{b:02x}"
                    char_count += 1
                
                print(f"  Current output ({len(decompressed)} bytes): \"{current_text}\"")
                print()  # Empty line for readability
            
            # Progress for large files
            if factor_count % 5000 == 0:
                print(f"Processed {factor_count} factors, current output size: {len(decompressed)} bytes")
    
    print(f"\nDecompression completed!")
    print(f"Total factors processed: {factor_count}")
    print(f"Final decompressed size: {len(decompressed)} bytes")
    
    # Print final first 100 characters
    print("\n" + "="*60)
    print("FINAL FIRST 100 CHARACTERS OF DECOMPRESSED DATA:")
    print("="*60)
    
    if len(decompressed) > 0:
        final_text = ""
        for i, b in enumerate(decompressed):
            if i >= 100:  # Show first 100 characters
                final_text += f"... (and {len(decompressed) - 100} more bytes)"
                break
            
            if 32 <= b <= 126:  # Printable ASCII
                final_text += chr(b)
            else:
                final_text += f"\\x{b:02x}"
        
        print(f"Text: \"{final_text}\"")
        
        # Also show hex representation of first 50 bytes
        print("\nFirst 50 bytes in hex:")
        hex_output = ""
        for i in range(min(50, len(decompressed))):
            hex_output += f"{decompressed[i]:02x} "
            if (i + 1) % 16 == 0:  # New line every 16 bytes
                hex_output += "\n"
        print(hex_output)
    else:
        print("No data decompressed!")
    
    print("="*60)
    
    return bytes(decompressed)

# def decompress_lz77(lz77_file):
#     """Decompress data from LZ77 binary file"""
#     decompressed = bytearray()
    
#     with open(lz77_file, 'rb') as f:
#         while True:
#             pos_data = f.read(8)  # size_t = 8 bytes
#             len_data = f.read(8)  # size_t = 8 bytes
            
#             if len(pos_data) != 8 or len(len_data) != 8:
#                 break
                
#             pos = struct.unpack('<Q', pos_data)[0]  # unsigned long long
#             length = struct.unpack('<Q', len_data)[0]
            
#             if length == 0:
#                 # Literal value: position contains the actual byte
#                 decompressed.append(pos & 0xFF)
#             else:
#                 # Copy reference: copy 'length' bytes starting from 'pos'
#                 # Handle self-referencing patterns correctly
#                 start_pos = pos
#                 for i in range(length):
#                     copy_index = start_pos + i
#                     if copy_index < len(decompressed):
#                         decompressed.append(decompressed[copy_index])
#                     else:
#                         # This should not happen in valid LZ77 data
#                         print(f"Warning: Invalid reference at position {copy_index}, current length {len(decompressed)}")
#                         break
    
#     return bytes(decompressed)

def verify_compression(original_file, lz77_file):
    """Verify if compression is correct by comparing original and decompressed data"""
    # Read original file
    with open(original_file, 'rb') as f:
        original_data = f.read()
    
    # Decompress LZ77 file
    decompressed_data = decompress_lz77(lz77_file)
    
    # Compare data
    if original_data == decompressed_data:
        print(f"✓ Verification successful! Files match perfectly")
        print(f"Original file size: {len(original_data)} bytes")
        print(f"Decompressed size: {len(decompressed_data)} bytes")
        
        # Calculate compression ratio
        if len(original_data) > 0:
            import os
            compressed_size = os.path.getsize(lz77_file)
            ratio = (1.0 - compressed_size / len(original_data)) * 100
            print(f"Compression ratio: {ratio:.2f}% (compressed size: {compressed_size} bytes)")
        
        return True
    else:
        print(f"✗ Verification failed! Files do not match")
        print(f"Original file size: {len(original_data)} bytes")
        print(f"Decompressed size: {len(decompressed_data)} bytes")
        
        # Find first mismatch position
        min_len = min(len(original_data), len(decompressed_data))
        mismatch_count = 0
        
        for i in range(min_len):
            if original_data[i] != decompressed_data[i]:
                if mismatch_count == 0:
                    print(f"First mismatch at position: {i}")
                    print(f"Original: {original_data[i]} (0x{original_data[i]:02x})")
                    print(f"Decompressed: {decompressed_data[i]} (0x{decompressed_data[i]:02x})")
                
                mismatch_count += 1
                if mismatch_count >= 10:  # Limit output for readability
                    print(f"... and {mismatch_count} more mismatches found")
                    break
                elif mismatch_count <= 5:  # Show first few mismatches
                    print(f"Mismatch at {i}: orig={original_data[i]}, decomp={decompressed_data[i]}")
        
        if len(original_data) != len(decompressed_data):
            print(f"Size difference: {abs(len(original_data) - len(decompressed_data))} bytes")
            
        return False

def main():
    """Main entry point"""
    if len(sys.argv) != 3:
        print("Usage: python3 [verify_lz77.py](http://_vscodecontentref_/1) <original_file> <lz77_file>")
        print("Example: python3 [verify_lz77.py](http://_vscodecontentref_/2) input.txt output_lz77.bin")
        sys.exit(1)
    
    original_file = sys.argv[1]
    lz77_file = sys.argv[2]
    
    # Check if files exist
    import os
    if not os.path.exists(original_file):
        print(f"Error: Original file not found: {original_file}")
        sys.exit(1)
    
    if not os.path.exists(lz77_file):
        print(f"Error: LZ77 file not found: {lz77_file}")
        sys.exit(1)
    
    print("Starting LZ77 compression verification...")
    print(f"Original file: {original_file}")
    print(f"LZ77 file: {lz77_file}")
    print("-" * 50)
    
    success = verify_compression(original_file, lz77_file)
    
    print("-" * 50)
    if success:
        print("Verification completed successfully!")
        sys.exit(0)
    else:
        print("Verification failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()