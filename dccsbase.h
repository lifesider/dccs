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
    
#if defined(__cplusplus)
}
#endif

#endif /* defined(__DccsBase__dccsbase__) */

