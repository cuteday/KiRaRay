#include <Windows.h>
#include <delayimp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
FARPROC WINAPI rejectCudaDriver(unsigned notification, PDelayLoadInfo info) {
	if (notification == dliStartProcessing && _stricmp(info->szDll, "nvcuda.dll") == 0) {
		std::fputs("CPU test unexpectedly requested the CUDA driver. Keep GPU initialization lazy.\n", stderr);
		std::exit(EXIT_FAILURE);
	}
	return nullptr;
}
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = rejectCudaDriver;
