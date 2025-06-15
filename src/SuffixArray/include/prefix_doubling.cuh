#pragma once
#include <vector>
#include <cstdint>

std::vector<uint32_t> build_suffix_array_prefix_doubling(const std::vector<uint8_t>& text);
