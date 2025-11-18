#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>

namespace lz77gpu {

// Initialize logger - call once at program start
inline void init_logger(const std::string& log_file = "") {
    try {
        std::shared_ptr<spdlog::logger> logger;

        if (log_file.empty()) {
            // Console only
            logger = spdlog::stdout_color_mt("lz77gpu");
        } else {
            // Console + file
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);

            std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
            logger = std::make_shared<spdlog::logger>("lz77gpu", sinks.begin(), sinks.end());
            spdlog::register_logger(logger);
        }

        // Set pattern: no timestamp for cleaner output (similar to original cout)
        logger->set_pattern("%v");
        logger->set_level(spdlog::level::info);
        spdlog::set_default_logger(logger);

    } catch (const spdlog::spdlog_ex& ex) {
        // Fallback to basic logger if initialization fails
        spdlog::set_pattern("%v");
    }
}

// Convenience macros for logging
#define LOG_INFO(...) spdlog::info(__VA_ARGS__)
#define LOG_WARN(...) spdlog::warn(__VA_ARGS__)
#define LOG_ERROR(...) spdlog::error(__VA_ARGS__)
#define LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)

// For progress updates that should stay on same line
#define LOG_PROGRESS(...) spdlog::info(__VA_ARGS__)

} // namespace lz77gpu
