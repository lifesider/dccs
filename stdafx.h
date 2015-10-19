// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include <stdio.h>
#include <tchar.h>
#include <windows.h>
#include <algorithm>

// TODO: reference additional headers your program requires here
#define SSE_OPTIMIZE


class clock
{
public:
	clock() { QueryPerformanceFrequency(&fre); }
	void start() { QueryPerformanceCounter(&beg); }
	void stop() { QueryPerformanceCounter(&end); }
	double gettimems() const { return (end.QuadPart-beg.QuadPart)*1000.0/fre.QuadPart; }

private:
	LARGE_INTEGER beg, end, fre;
};
