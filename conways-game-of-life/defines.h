#pragma once




#define GLUE(a, b) a##b
#define GLUE_ADAPTER(a, b) GLUE(a, b)
#define UNIQUE_VAR(pref) GLUE_ADAPTER(pref, __LINE__)
#define __FOO___(a) #a
#define __FOO__(a) __FOO___(a)

#ifdef _DEBUG
#define cuCHECK(func) cudaError_t UNIQUE_VAR(__cuda_error_var_) = func; if (UNIQUE_VAR(__cuda_error_var_) != cudaSuccess) { printf(cudaGetErrorString(UNIQUE_VAR(__cuda_error_var_))); printf(" | " __FILE__ " | Line: " __FOO__(__LINE__) "\n"); throw 1; }
#define sdlCheck(func) int UNIQUE_VAR(__sdl_var_) = func; assert(UNIQUE_VAR(__sdl_var_) == 0);
#else
#define cuCHECK(func) func;
#define sdlCheck(func) func;
#endif // _DEBUG

