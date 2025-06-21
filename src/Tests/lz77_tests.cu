#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <vector>

#include "prefix_doubling.cuh"


TEST_CASE("Suffix Array: MISSISSIPPI Example") {
    std::string input = "MISSISSIPPIMISSISSIPPI";
    std::vector<uint8_t> bytes(input.begin(), input.end());
    std::vector<uint32_t> expected_sa = {
        21, 10, 18, 7, 15, 4, 12, 1, 11, 0, 20, 9, 19, 8, 17, 6, 14, 3, 16, 5, 13, 2
    };

    SECTION("Output size is correct") {
        auto sa = build_suffix_array_prefix_doubling(bytes);
        REQUIRE(sa.size() == bytes.size());
    }

    SECTION("Matches reference suffix array") {
        auto sa = build_suffix_array_prefix_doubling(bytes);
        REQUIRE(sa == expected_sa);
    }

    SECTION("Suffixes are in lexicographic order") {
        auto sa = build_suffix_array_prefix_doubling(bytes);
        std::vector<std::string> suffixes;
        for (uint32_t i : sa)
            suffixes.emplace_back(input.substr(i));
        for (size_t i = 1; i < suffixes.size(); ++i)
            REQUIRE(suffixes[i-1] <= suffixes[i]);
    }

    SECTION("ISA is the true inverse") {
        auto sa = build_suffix_array_prefix_doubling(bytes);
        std::vector<uint32_t> isa(sa.size());
        for (size_t i = 0; i < sa.size(); ++i)
            isa[sa[i]] = i;
        for (size_t i = 0; i < sa.size(); ++i)
            REQUIRE(sa[isa[i]] == i);
    }
}
