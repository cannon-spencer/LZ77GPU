#pragma once
#include <vector>
#include <cstdint>

uint32_t* build_suffix_array_prefix_doubling(const std::vector<uint8_t>& s, size_t& out_length);

size_t* build_suffix_array_prefix_doubling_64(const std::vector<uint8_t>& s, size_t& out_length);