#include "logger.h"

KRR_NAMESPACE_BEGIN

namespace {
    using Level = Log::Level;

    static Level sVerbosity = Level::Info;
    static bool sLogTime = true;
    static bool sLogToConsole = true;
    static bool sAnsiControlSequence = false;
    static bool sInitialized = false;

}

void Logger::log(Level level, const string& msg, bool terminate) {

    if (!sInitialized) {
        sAnsiControlSequence = Log::enableAnsiControlSequences();
        if (sAnsiControlSequence) std::cout << Log::TERMINAL_DEFAULT;
        sInitialized = true;
    }

    if (level <= sVerbosity) {
        std::string s = "";

        if (sLogTime) s += Log::nowToString("%H:%M:%S ");

        if (sAnsiControlSequence) s += Log::getLevelAnsiColor(level);

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