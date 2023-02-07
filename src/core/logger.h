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

KRR_NAMESPACE_BEGIN

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

// logger class
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

// util functions
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

	inline std::string padFromLeft(std::string str, size_t length, const char paddingChar = ' ') {
		if (length > str.size()) {
			str.insert(0, length - str.size(), paddingChar);
		}
		return str;
	}

	inline std::string padFromRight(std::string str, size_t length, const char paddingChar = ' ') {
		if (length > str.size()) {
			str.resize(length, paddingChar);
		}
		return str;
	}

	template <typename T>
	std::string durationToString(T dur) {
		using namespace std::chrono;

		auto h = duration_cast<hours>(dur);
		auto m = duration_cast<minutes>(dur -= h);
		auto s = duration_cast<seconds>(dur -= m);

		if (h.count() > 0) {
			return
				std::to_string(h.count()) + 'h' +
				padFromLeft(std::to_string(m.count()), 2, '0') + 'm' +
				padFromLeft(std::to_string(s.count()), 2, '0') + 's';
		}
		else if (m.count() > 0) {
			return
				std::to_string(m.count()) + 'm' +
				padFromLeft(std::to_string(s.count()), 2, '0') + 's';
		}
		else {
			return std::to_string(s.count()) + 's';
		}
	}

	using duration_t = std::chrono::microseconds;
	inline std::string progressBar(uint64_t current, uint64_t total, duration_t duration, int width) {
		if (total == 0) {
			throw std::invalid_argument{ "Progress: total must not be zero." };
		}

		if (current > total) {
			throw std::invalid_argument{ "Progress: current must not be larger than total" };
		}

		double fraction = (double)current / total;

		// Percentage display. Looks like so:
		//  69%
		int percentage = (int)std::round(fraction * 100);
		std::string percentageStr = padFromLeft(std::to_string(percentage) + "%", 4);

		// Fraction display. Looks like so:
		// ( 123/1337)
		std::string totalStr = std::to_string(total);
		std::string fractionStr = padFromLeft(std::to_string(current) + "/" + totalStr, totalStr.size() * 2 + 1);

		// Time display. Looks like so:
		//     3s/17m03s
		std::string projectedDurationStr;
		if (current == 0) {
			projectedDurationStr = "inf";
		}
		else {
			auto projectedDuration = duration * (1 / fraction);
			projectedDurationStr = durationToString(projectedDuration);
		}
		std::string timeStr = padFromLeft(durationToString(duration) + "/" + projectedDurationStr, projectedDurationStr.size() * 2 + 1);

		// Put the label together. Looks like so:
		//  69% ( 123/1337)     3s/17m03s
		std::string label = percentageStr + " (" + fractionStr + ") " + timeStr;

		// Build the progress bar itself. Looks like so:
		// [=================>                         ]
		int usableWidth = width - 2 - 1 - (int)label.size();
		usableWidth = usableWidth > 0 ? usableWidth : 0;

		int numFilledChars = (int)std::round(usableWidth * fraction);

		std::string body(usableWidth, ' ');
		if (numFilledChars > 0) {
			for (int i = 0; i < numFilledChars; ++i)
				body[i] = '=';
			if (numFilledChars < usableWidth) {
				body[numFilledChars] = '>';
			}
		}

		// Put everything together. Looks like so:
		// [=================>                         ]  69% ( 123/1337)     3s/17m03s
		return std::string{ "[" } + body + "] " + label;
	}

}

KRR_NAMESPACE_END
