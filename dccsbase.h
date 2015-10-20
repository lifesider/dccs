//
//  dccsbase.h
//  DccsBase
//
//  Created by 赖守波 on 15/9/17.
//  Copyright (c) 2015年 Sobey. All rights reserved.
//

#ifndef __DccsBase__dccsbase__
#define __DccsBase__dccsbase__

#include <stdio.h>
#include <stddef.h>
#include <intrin.h>

#ifndef IN
#define IN
#define OUT
#endif

#if defined(_WIN32) || defined(_WINDOWS)
#define decl_align(type, n, var)	__declspec(align(n)) ##type var
#else
#define decl_align(type, n, var)	type var __attribute__((aligned(n)))
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void rgb2gray_s_sse2(OUT unsigned char* gray, IN unsigned char const* rgb, IN int count);
    
void rgb2gray_f_sse2(OUT unsigned char* gray, IN unsigned char const* rgb, IN int count);
  
void get_gr_channel_sse2(unsigned char* green, unsigned char* red, unsigned char const* bgr, int count);
    
void gray2binary(unsigned char* binary, unsigned char const* gray, int threshold, int count);
    
double dotproduct_d(double const* src1, double const* src2, int count);
    
void resample_linear_8_line(unsigned char* des, unsigned char const* src, intptr_t stride, int fy, int kx, int width);

void resample_linear_24_line(unsigned char* des, unsigned char const* src, intptr_t stride, int fy, int kx, int width);

void imfilter_3x3_line2(unsigned char* des, unsigned char const* src, intptr_t stride, short const coef[4], int width);

void imfilter_3x3_line3(unsigned char* des, unsigned char const* src, intptr_t stride, short const coef[4], int width);

size_t calccnt8_eq_sse2(unsigned char* src, int val, size_t count);

void calccnt8_ver_sse2(int* des, unsigned char* src, intptr_t stride, int height);

void ucharnorm2double_sse2(OUT double *des, IN unsigned char const*src, IN size_t count);

void dmemconv(double* des, double const* src, double const* kernel, int kernel_size, int count);

void dmemmul(double* des, double const* src, double weight, int count);

void dmemmad(double* des, double const* src, double weight, int count);

void nsp_filter(OUT double* des,			// 滤波后输出缓冲
				IN double const* src,		// 输入缓冲
				IN double const* kernel,	// 一维核系数缓冲
				IN int kernel_size,			// 一维核长度
				IN int direction,			// 0-水平，1-垂直，2-水平垂直
				IN int width,				// 宽度
				IN int height);				// 高度

#if defined(__cplusplus)
}
#endif

#endif /* defined(__DccsBase__dccsbase__) */

