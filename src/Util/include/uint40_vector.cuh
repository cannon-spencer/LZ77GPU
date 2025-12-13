#ifndef UINT40_VECTOR_CUH
#define UINT40_VECTOR_CUH

#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <omp.h>

/**
 * Memory-efficient vector storing 64-bit values in 5 bytes each
 * Saves 37.5% memory compared to size_t on 64-bit systems
 *
 * Storage format (little-endian):
 * - Element i stored at bytes [i*5 .. i*5+4]
 * - Max value: 2^40 - 1 = 1,099,511,627,775 (~1TB)
 *
 * Thread safety: Reads/writes to different indices are safe
 */
class uint40_vector {
public:
    /**
     * Construct vector with given capacity
     * @param capacity Number of uint40 elements to store
     */
    explicit uint40_vector(size_t capacity)
        : capacity_(capacity)
        , data_(capacity * 5) {
        if (capacity == 0) {
            throw std::invalid_argument("uint40_vector capacity must be > 0");
        }
    }

    /**
     * Get element at index
     * @param idx Index to read
     * @return 64-bit value (fits in 40 bits)
     */
    inline uint64_t get(size_t idx) const {
        if (idx >= capacity_) {
            throw std::out_of_range("uint40_vector index out of range");
        }

        const uint8_t* ptr = &data_[idx * 5];
        uint64_t value = 0;

        // Little-endian read: bytes 0-4
        value |= static_cast<uint64_t>(ptr[0]);
        value |= static_cast<uint64_t>(ptr[1]) << 8;
        value |= static_cast<uint64_t>(ptr[2]) << 16;
        value |= static_cast<uint64_t>(ptr[3]) << 24;
        value |= static_cast<uint64_t>(ptr[4]) << 32;

        return value;
    }

    /**
     * Set element at index
     * @param idx Index to write
     * @param value Value to store (must fit in 40 bits)
     */
    inline void set(size_t idx, uint64_t value) {
        if (idx >= capacity_) {
            throw std::out_of_range("uint40_vector index out of range");
        }
        if (value >= (1ULL << 40)) {
            throw std::overflow_error("Value exceeds 40-bit limit");
        }

        uint8_t* ptr = &data_[idx * 5];

        // Little-endian write: bytes 0-4
        ptr[0] = static_cast<uint8_t>(value);
        ptr[1] = static_cast<uint8_t>(value >> 8);
        ptr[2] = static_cast<uint8_t>(value >> 16);
        ptr[3] = static_cast<uint8_t>(value >> 24);
        ptr[4] = static_cast<uint8_t>(value >> 32);
    }

    /**
     * Pack size_t vector into uint40 format (OpenMP parallel)
     * @param src Source vector with 64-bit values
     */
    void pack_from(const std::vector<size_t>& src) {
        if (src.size() != capacity_) {
            throw std::invalid_argument("Source vector size mismatch");
        }

        #pragma omp parallel for schedule(static, 10240)
        for (size_t i = 0; i < capacity_; ++i) {
            uint64_t value = src[i];
            if (value >= (1ULL << 40)) {
                #pragma omp critical
                {
                    throw std::overflow_error("Value at index " + std::to_string(i) +
                                             " exceeds 40-bit limit: " + std::to_string(value));
                }
            }

            uint8_t* ptr = &data_[i * 5];
            ptr[0] = static_cast<uint8_t>(value);
            ptr[1] = static_cast<uint8_t>(value >> 8);
            ptr[2] = static_cast<uint8_t>(value >> 16);
            ptr[3] = static_cast<uint8_t>(value >> 24);
            ptr[4] = static_cast<uint8_t>(value >> 32);
        }
    }

    /**
     * Unpack range to size_t vector (OpenMP parallel)
     * @param offset Starting index
     * @param count Number of elements
     * @param dst Destination vector (will be resized)
     */
    void unpack_range(size_t offset, size_t count, std::vector<uint64_t>& dst) const {
        if (offset + count > capacity_) {
            throw std::out_of_range("Unpack range exceeds capacity");
        }

        dst.resize(count);

        #pragma omp parallel for schedule(static, 10240)
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* ptr = &data_[(offset + i) * 5];
            uint64_t value = 0;

            value |= static_cast<uint64_t>(ptr[0]);
            value |= static_cast<uint64_t>(ptr[1]) << 8;
            value |= static_cast<uint64_t>(ptr[2]) << 16;
            value |= static_cast<uint64_t>(ptr[3]) << 24;
            value |= static_cast<uint64_t>(ptr[4]) << 32;

            dst[i] = value;
        }
    }

    /**
     * Unpack entire array to size_t vector (OpenMP parallel)
     * @param dst Destination vector (will be resized)
     */
    void unpack_all(std::vector<uint64_t>& dst) const {
        unpack_range(0, capacity_, dst);
    }

    /**
     * Pack range from size_t vector (OpenMP parallel)
     * Useful for writing PSV/NSV results back
     * @param offset Starting index in this uint40_vector
     * @param src Source vector with values to pack
     * @param src_offset Starting index in source vector
     * @param count Number of elements to pack
     */
    void pack_range(size_t offset, const std::vector<uint64_t>& src,
                    size_t src_offset, size_t count) {
        if (offset + count > capacity_) {
            throw std::out_of_range("Pack range exceeds capacity");
        }
        if (src_offset + count > src.size()) {
            throw std::out_of_range("Source range exceeds source size");
        }

        #pragma omp parallel for schedule(static, 10240)
        for (size_t i = 0; i < count; ++i) {
            uint64_t value = src[src_offset + i];
            if (value >= (1ULL << 40)) {
                #pragma omp critical
                {
                    throw std::overflow_error("Value exceeds 40-bit limit");
                }
            }

            uint8_t* ptr = &data_[(offset + i) * 5];
            ptr[0] = static_cast<uint8_t>(value);
            ptr[1] = static_cast<uint8_t>(value >> 8);
            ptr[2] = static_cast<uint8_t>(value >> 16);
            ptr[3] = static_cast<uint8_t>(value >> 24);
            ptr[4] = static_cast<uint8_t>(value >> 32);
        }
    }

    /**
     * Get capacity (number of uint40 elements)
     */
    size_t capacity() const { return capacity_; }

    /**
     * Get raw byte size
     */
    size_t byte_size() const { return data_.size(); }

private:
    size_t capacity_;           // Number of uint40 elements
    std::vector<uint8_t> data_; // Raw bytes (capacity * 5)
};

#endif // UINT40_VECTOR_CUH
