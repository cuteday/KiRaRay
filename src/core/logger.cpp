#include "logger.h"

KRR_NAMESPACE_BEGIN

using namespace Log;

namespace {
#ifdef KRR_DEBUG_BUILD
    static Level sVerbosity = Level::Debug;
#else
    static Level sVerbosity = Level::Info;
#endif
    static bool sLogTime = true;
    static bool sLogToConsole = true;
    static bool sAnsiControlSequence = false;
    static bool sInitialized = false;

    inline string getLogLevelString(Level level) {
        switch (level)
        {
        case Level::Fatal:
            return "[Fatal]";
        case Level::Error:
            return "[Error]";
        case Level::Warning:
            return "[Warning]";
        case Level::Info:
        case Level::Success:
            return "[Info]";
        case Level::Debug:
            return "[Debug]";
        default:
            KRR_SHOULDNT_GO_HERE;
            return nullptr;
        }
    }

    inline string getLevelAnsiColor(Level level) {
        switch (level) {
        case Level::Success:
            return TERMINAL_GREEN;
        case Level::Warning:
            return TERMINAL_YELLOW;
        case Level::Error:
            return TERMINAL_LIGHT_RED;
        case Level::Fatal:
            return TERMINAL_RED;
        default:
            return TERMINAL_DEFAULT;
        }
    }

    static bool enableAnsiControlSequences() {
#ifdef _WIN32
        // Set output mode to handle virtual terminal sequences
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut == INVALID_HANDLE_VALUE) {
            return false;
        }
        DWORD dwMode = 0;
        if (!GetConsoleMode(hOut, &dwMode)) {
            return false;
        }
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if (!SetConsoleMode(hOut, dwMode)) {
            return false;
        }
#endif
        return true;
    }
}

void Logger::log(Level level, const string& msg, bool terminate) {

    if (!sInitialized) {
        sAnsiControlSequence = enableAnsiControlSequences();
        if (sAnsiControlSequence) std::cout << Log::TERMINAL_DEFAULT;
        sInitialized = true;
    }

    if (!msg.length()) return;

    if (level <= sVerbosity) {
        std::string s = "";

        if (sAnsiControlSequence) s += getLevelAnsiColor(level);
        if (sLogTime) s += Log::nowToString("%H:%M:%S ");

        s += getLogLevelString(level) + std::string(" ") + msg + "\n";

        if (sAnsiControlSequence) s += Log::TERMINAL_DEFAULT;
        if (level > Level::Error) {
            std::cout << s;
        }
        else {
            std::cerr << s;
        }
    }

    if (level < Level::Error || terminate) exit(1);
}

KRR_NAMESPACE_END