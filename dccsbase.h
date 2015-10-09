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

#ifndef IN
#define IN
#define OUT
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void rgb2gray_s_sse2(OUT unsigned char* gray, IN unsigned char const* rgb, IN int count);
    
void rgb2gray_f_sse2(OUT unsigned char* gray, IN unsigned char const* rgb, IN int count);
  
void get_gr_channel_sse2(unsigned char* green, unsigned char* red, unsigned char const* bgr, int count);
    
void gray2binary(unsigned char* binary, unsigned char const* gray, int count);
    
double dotproduct_d(double const* src1, double const* src2, int count);
    
void resample_nearest_line(unsigned char* des, unsigned char const* src, int bytes_per_pixel, float coef[4], double step, int width);

void resample_linear_8_line(unsigned char* des, unsigned char const* src, intptr_t stride, int fy, int kx, int width);

void resample_linear_24_line(unsigned char* des, unsigned char const* src, intptr_t stride, int fy, int kx, int width);

#if defined(__cplusplus)
}
#endif

#endif /* defined(__DccsBase__dccsbase__) */

