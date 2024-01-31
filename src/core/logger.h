#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <util/string.h>

#include <common.h>

#ifdef _WIN32   // for ansi control sequences
#   define NOMINMAX 
#   include <Windows.h>
#   undef NOMINMAX
#else
#   include <sys/ioctl.h>
#   include <unistd.h>
#endif

NAMESPACE_BEGIN(krr)

namespace Log {

	const string TERMINAL_ESC = "\033";
	const string TERMINAL_RED = "\033[0;31m";
	const string TERMINAL_GREEN = "\033[0;32m";
	const string TERMINAL_LIGHT_GREEN = "\033[1;32m";
	const string TERMINAL_YELLOW = "\033[1;33m";
	const string TERMINAL_BLUE = "\033[0;34m";
	const string TERMINAL_LIGHT_BLUE = "\033[1;34m";
	const string TERMINAL_RESET = "\033[0m";
	const string TERMINAL_DEFAULT = TERMINAL_RESET;
	const string TERMINAL_BOLD = "\033[1;1m";
	const string TERMINAL_LIGHT_RED = "\033[1;31m";

	enum class Level {
		None,
		Fatal,
		Error,
		Warning,
		Success,
		Info,
		Debug,
	};
}

class Logger {
private:
	using Level = Log::Level;
	
	friend void logDebug(const std::string& msg);
	friend void logInfo(const std::string& msg);
	friend void logSuccess(const std::string& msg);
	friend void logWarning(const std::string& msg);
	friend void logError(const std::string& msg, bool terminate);
	friend void logFatal(const std::string& msg, bool terminate);
	template <typename... Args> 
	friend void logMessage(Log::Level level, const std::string &msg, Args &&...payload);

	static void log(Level level, const string& msg, bool terminate = false);
};

inline void logDebug(const std::string& msg) { Logger::log(Logger::Level::Debug, msg); }
inline void logInfo(const std::string& msg) { Logger::log(Logger::Level::Info, msg); }
inline void logSuccess(const std::string& msg) { Logger::log(Logger::Level::Success, msg); }
inline void logWarning(const std::string& msg) { Logger::log(Logger::Level::Warning, msg); }
inline void logError(const std::string& msg, bool terminate = false) { Logger::log(Logger::Level::Error, msg, terminate); }
inline void logFatal(const std::string& msg, bool terminate = true) { Logger::log(Logger::Level::Fatal, msg, terminate); }

template <typename ...Args> 
inline void logMessage(Log::Level level, const std::string &msg, Args &&...args) {
	Logger::log(level, formatString(msg, std::forward<Args>(args)...));
}

#define Log(level, fmt, ...) do{						\
		log##level(formatString(fmt, ##__VA_ARGS__));	\
	} while (0)

namespace Log {
	inline std::string timeToString(const std::string& fmt, time_t time) {
		char timeStr[128];
		if (std::strftime(timeStr, 128, fmt.c_str(), localtime(&time)) == 0) {
			throw std::runtime_error{ "Could not render local time." };
		}
		return timeStr;
	}

	inline std::string nowToString(const std::string& fmt) {
		return timeToString(fmt, std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
	}
}

NAMESPACE_END(krr)
