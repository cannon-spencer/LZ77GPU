#pragma once
#include <vector>
#include <cstdint>

#ifndef SA_DEBUG
#define SA_DEBUG 1
#endif

std::vector<uint32_t> build_suffix_array_prefix_doubling(const std::vector<uint8_t>& text);
uint32_t*             build_suffix_array_prefix_doubling_device(const std::vector<uint8_t>& s);
