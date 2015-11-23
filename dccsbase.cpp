//
//  dccsbase.cpp
//  DccsBase
//
//  Created by 赖守波 on 15/9/17.
//  Copyright (c) 2015年 Sobey. All rights reserved.
//

#include "dccsbase.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <memory>
#include <xutility>
#include <assert.h>
#include <CL/opencl.h>

#pragma comment(lib, "OpenCL")

#pragma warning(disable: 4996)

#ifdef min
#undef min
#undef max
#endif

// OpenCL resources
cl_platform_id		clPlatformID = NULL;
cl_device_id		clDeviceID = NULL;
cl_context			clContext = NULL;
cl_command_queue	clCmdQueue = NULL;
cl_device_type		clDeviceType = CL_DEVICE_TYPE_GPU;
cl_program			clPgmBase = NULL;
cl_mem				clSmooth1D = NULL;
cl_mem				clSmooth2DX = NULL;
cl_mem				clSmooth2DY = NULL;
cl_mem				clFactorRegion = NULL;
cl_mem				clLutArray1 = NULL;
cl_mem				clLutArray2 = NULL;

#define KERNEL(...) #__VA_ARGS__

char const* g_szKernel = KERNEL(

__kernel void GaussianSmoothX(write_only image2d_t des, read_only image2d_t src, global float const* filter, int width, int height, sampler_t sampler)
{
	local float coef[9];
	if(get_local_id(0) < 9 && get_local_id(1) == 0)
		coef[get_local_id(0)] = filter[get_local_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	int2 outCoord = (int2)(get_global_id(0), get_global_id(1));
	if(outCoord.x < width && outCoord.y < height)
	{
		float4 outColor = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		int weight = 0;
		for(int x=outCoord.x-4; x<=outCoord.x+4; ++x, ++weight)
		{
			outColor += read_imagef(src, sampler, (int2)(x, outCoord.y)) * coef[weight];
		}
		write_imagef(des, outCoord, outColor);
	}
}

__kernel void GaussianSmoothY(write_only image2d_t des, read_only image2d_t src, global float const* filter, int width, int height, sampler_t sampler)
{
	local float coef[9];
	if(get_local_id(0) < 9 && get_local_id(1) == 0)
		coef[get_local_id(0)] = filter[get_local_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	int2 outCoord = (int2)(get_global_id(0), get_global_id(1));
	if(outCoord.x < width && outCoord.y < height)
	{
		float4 outColor = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		int weight = 0;
		for(int y=outCoord.y-4; y<=outCoord.y+4; ++y, ++weight)
		{
			outColor += read_imagef(src, sampler, (int2)(outCoord.x, y)) * coef[weight];
		}
		write_imagef(des, outCoord, outColor);
	}
}

__kernel void Gaussian2DReplicate(write_only image2d_t des, read_only image2d_t src, global float const* filter, int width, int height, sampler_t sampler)
{
	local float coef[81];
	if(get_local_id(1) < 5)
	{
		int idx = get_local_id(1)*16 + get_local_id(0);
		coef[idx] = filter[idx];
	}
	if(get_local_id(1) == 5 && get_local_id(0) == 0)
		coef[80] = filter[80];
	barrier(CLK_LOCAL_MEM_FENCE);
	int2 outCoord = (int2)(get_global_id(0), get_global_id(1));
	if(outCoord.x < width && outCoord.y < height)
	{
		int2 startCoord = (int2)(outCoord.x - 4, outCoord.y - 4);
		int2 endCoord = (int2)(outCoord.x + 4, outCoord.y + 4);
		float4 outColor = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		int weight = 0;
		for(int y=startCoord.y; y<=endCoord.y; ++y)
		{
			for(int x=startCoord.x; x<=endCoord.x; ++x)
			{
				outColor += read_imagef(src, sampler, (int2)(x, y)) * coef[weight];
				weight++;
			}
		}
		write_imagef(des, outCoord, outColor);
	}
}

__kernel void GradMagnitude(write_only image2d_t des, read_only image2d_t gradX, read_only image2d_t gradY, int width, int height, sampler_t sampler)
{
	int2 outCoord = (int2)(get_global_id(0), get_global_id(1));
	if(outCoord.x < width && outCoord.y < height)
	{
		float x = read_imagef(gradX, sampler, outCoord).x;
		float y = read_imagef(gradY, sampler, outCoord).x;
		float color = length((float2)(x, y));
		write_imagef(des, outCoord, (float4)(color));
	}
}

__kernel void GradMax(global float* des, read_only image2d_t src, sampler_t sampler, int width, int height)
{
	local float buffer[16][16];
	int2 c = (int2)(get_local_id(0), get_local_id(1));
	buffer[c.y][c.x] = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int y=c.y; y<height; y+=get_local_size(1))
	{
		for(int x=c.x; x<width; x+=get_local_size(0))
			buffer[c.y][c.x] = fmax(buffer[c.y][c.x], read_imagef(src, sampler, (int2)(x, y)).x);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int s=8; s>0; s>>=1)
		if(c.x < s)
			buffer[c.y][c.x] = fmax(buffer[c.y][c.x], buffer[c.y][c.x+s]);
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int s=8; s>0; s>>=1)
		if(c.y < s && c.x == 0)
			buffer[c.y][0] = fmax(buffer[c.y][0], buffer[c.y+s][0]);
	barrier(CLK_LOCAL_MEM_FENCE);
	if(get_local_id(0) == 0 && get_local_id(1) == 0)
		des[0] = buffer[0][0];
}

__kernel void GradNorm(write_only image2d_t des, read_only image2d_t src, int width, int height, sampler_t sampler, global float* maxGrad)
{
	local float G;
	if(get_local_id(0) == 0 && get_local_id(1) == 0)
		G = maxGrad[0];
	barrier(CLK_LOCAL_MEM_FENCE);
	int2 outCoord = (int2)(get_global_id(0), get_global_id(1));
	if(outCoord.x < width && outCoord.y < height)
	{
		write_imagef(des, outCoord, read_imagef(src, sampler, outCoord) / G);
	}
}

__kernel void GradHist(global int* hist, read_only image2d_t grad, global float const* buffer, int width, int height, sampler_t sampler)
{
	local float cbuffer[64];
	local int chist[64];
	if(get_local_id(1) < 4)
	{
		int idx = get_local_id(1)*16 + get_local_id(0);
		cbuffer[idx] = buffer[idx];
		chist[idx] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	int2 outCoord = (int2)(get_global_id(0), get_global_id(1));
	if(outCoord.x < width && outCoord.y < height)
	{
		float v = read_imagef(grad, sampler, outCoord).x;
		for(int i=0; i<64; ++i)
		{
			if(v < cbuffer[i])
			{
				atomic_inc(&chist[i]);
				break;
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(get_local_id(1) < 4)
	{
		int idx = get_local_id(1)*16 + get_local_id(0);
		atomic_add(&hist[idx], chist[idx]);
	}
}

__kernel void CalcThreshold(global float* buffer, float highRatio, float lowRatio, int count)
{
	local int cbuffer[64];
	cbuffer[get_global_id(0)] = ((int*)buffer)[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	float highTh = 0.0f;
	float lowTh = 0.0f;
	float TH = highRatio * count;
	if(get_local_id(0) == 0)
	{
		for(int i=1; i<64; ++i)
			cbuffer[i] += cbuffer[i-1];
		for(int i=0; i<64; ++i)
		{
			if(cbuffer[i] > TH)
			{
				highTh = (i+1)*(1.0f/64);
				break;
			}
		}
		lowTh = highTh * lowRatio;
		buffer[0] = lowTh;
		buffer[1] = highTh;
	}
}

__kernel void NonMaxSuppress(write_only image2d_t des, read_only image2d_t GradX, read_only image2d_t GradY, read_only image2d_t Grad, int width, int height, sampler_t sampler)
{
	int2 outCoord = (int2)(get_global_id(0), get_global_id(1));
	if(outCoord.x < width && outCoord.y < height)
	{
		float4 color = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		if(outCoord.x > 0 && outCoord.x < width-1 && outCoord.y > 0 && outCoord.y < height-1)
		{
			float G = read_imagef(Grad, sampler, outCoord).x;
			if(G != 0)
			{
				float GX = read_imagef(GradX, sampler, outCoord).x;
				float GY = read_imagef(GradY, sampler, outCoord).x;
				float G1 = 0.0f, G2 = 0.0f, G3 = 0.0f, G4 = 0.0f;
				float w = 0.0f;
				if(fabs(GY) > fabs(GX))
				{
					w = fabs(GX) / fabs(GY);
					G2 = read_imagef(Grad, sampler, (int2)(outCoord.x, outCoord.y-1)).x;
					G4 = read_imagef(Grad, sampler, (int2)(outCoord.x, outCoord.y+1)).x;
					if(GX * GY > 0)
					{
						G1 = read_imagef(Grad, sampler, (int2)(outCoord.x-1, outCoord.y-1)).x;
						G3 = read_imagef(Grad, sampler, (int2)(outCoord.x+1, outCoord.y+1)).x;
					}
					else
					{
						G1 = read_imagef(Grad, sampler, (int2)(outCoord.x+1, outCoord.y-1)).x;
						G3 = read_imagef(Grad, sampler, (int2)(outCoord.x-1, outCoord.y+1)).x;
					}
				}
				else
				{
					w = fabs(GY) / fabs(GX);
					G2 = read_imagef(Grad, sampler, (int2)(outCoord.x+1, outCoord.y)).x;
					G4 = read_imagef(Grad, sampler, (int2)(outCoord.x-1, outCoord.y)).x;
					if(GX * GY > 0)
					{
						G1 = read_imagef(Grad, sampler, (int2)(outCoord.x+1, outCoord.y+1)).x;
						G3 = read_imagef(Grad, sampler, (int2)(outCoord.x-1, outCoord.y-1)).x;
					}
					else
					{
						G1 = read_imagef(Grad, sampler, (int2)(outCoord.x+1, outCoord.y-1)).x;
						G3 = read_imagef(Grad, sampler, (int2)(outCoord.x-1, outCoord.y+1)).x;
					}
				}
				float Tmp1 = w * G1 + (1.0f - w) * G2;
				float Tmp2 = w * G3 + (1.0f - w) * G4;
				if(G >= Tmp1 && G >= Tmp2)
					color = (float4)(0.5f, 0.5f, 0.5f, 0.5f);
			}
		}
		write_imagef(des, outCoord, color);
	}
}

__kernel void Hysteresis(write_only image2d_t des, read_only image2d_t src, read_only image2d_t grad, sampler_t sampler, int width, int height, global float* thresh)
{
	local float buffer[2];
	if(get_local_id(1) == 0 && get_local_id(0) < 2)
		buffer[get_local_id(0)] = thresh[get_local_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if(coord.x < width && coord.y < height)
	{
		float c = read_imagef(src, sampler, coord).x;
		float g = read_imagef(grad, sampler, coord).x;
		if(fabs(c-0.5f)<0.01f && g >= buffer[1])
			write_imagef(des, coord, (float4)(1.0f));
		else if(fabs(c-0.5f)<0.01f && g >= buffer[0])
			write_imagef(des, coord, (float4)(0.5f));
		else
			write_imagef(des, coord, (float4)(0.0f));
	}
}

__kernel void TransposeSet(write_only image2d_t des, read_only image2d_t src, sampler_t sampler, int width, int height)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if(coord.x < width && coord.y < height)
	{
		float c = read_imagef(src, sampler, (int2)(coord.y, coord.x)).x;
		write_imagef(des, coord, (float4)(c == 1.f ? 1.f : 0.f));
	}
}

__kernel void Transpose(write_only image2d_t des, read_only image2d_t src, sampler_t sampler, int width, int height)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if(coord.x < width && coord.y < height)
	{
		write_imagef(des, coord, read_imagef(src, sampler, (int2)(coord.y, coord.x)));
	}
}

constant int nWeights3[3][3] = { {1, 8, 64}, {2, 16, 128}, {4, 32, 256} };
__kernel void ApplyLut(write_only image2d_t des, read_only image2d_t src, global unsigned char* lut, sampler_t sampler, int width, int height)
{
	local int buffer[512];
	int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
	buffer[tid] = lut[tid];
	buffer[tid+256] = lut[tid+256];
	barrier(CLK_LOCAL_MEM_FENCE);
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if(coord.x < width && coord.y < height)
	{
		int minR = coord.x == 0 ? 1 : 0;
		int maxR = coord.x == width-1 ? 1 : 2;
		int minC = coord.y == 0 ? 1 : 0;
		int maxC = coord.y == height-1 ? 1 : 2;
		int result = 0;
		for(int rr=minR; rr<=maxR; ++rr)
		{
			for(int cc=minC; cc<=maxC; ++cc)
			{
				result += nWeights3[rr][cc] * (read_imagef(src, sampler, (int2)(coord.x+rr-1, coord.y+cc-1)).x != 0);
			}
		}
		write_imagef(des, coord, (float4)(buffer[result] == 0 ? 0.0f : 1.0f));
	}
}

);

#if defined(_M_X64) || defined(__x86_64__)

#define eax_ptr			rax
#define ebx_ptr			rbx
#define ecx_ptr			rcx
#define edx_ptr			rdx
#define edi_ptr			rdi
#define esi_ptr			rsi
#define ptrword			qword
#define PTR_BYTES_1		0x08
#define push_eflags		pushfq		// Push EFLAGS Register onto the Stack
#define pop_eflags		popfq		// Pop Stack into EFLAGS Register
#define __x64_platform__
#define movsxd			movsxd

#else

#define eax_ptr			eax
#define ebx_ptr			ebx
#define ecx_ptr			ecx
#define edx_ptr			edx
#define edi_ptr			edi
#define esi_ptr			esi
#define ptrword			dword
#define PTR_BYTES_1		0x04
#define push_eflags		pushfd		// Push EFLAGS Register onto the Stack
#define pop_eflags		popfd		// Pop Stack into EFLAGS Register
#define movsxd			mov

#endif

// scale by 1024
decl_align(short, 16, coefBGR[8]) = { 117, 601, 306, 0, 0, 117, 601, 306};
void rgb2gray_s_sse2(OUT unsigned char* gray, IN unsigned char const* rgb, IN int count)
{
    __asm
    {
        movsxd		eax_ptr, count;
        mov			esi_ptr, rgb;
        mov			edi_ptr, gray;
        movdqa		xmm0, coefBGR;
        pxor		xmm7, xmm7;
        sub         eax_ptr, 8;
        jl			loop_1_pre;
    loop_8:
        movdqu		xmm3, [esi_ptr];
        movsd		xmm4, [esi_ptr + 0x10];
        movdqa		xmm5, xmm3;
        pshufd		xmm6, xmm3, 1001b;
        punpcklbw	xmm5, xmm7;
        punpcklbw	xmm6, xmm7;
        shufps		xmm5, xmm5, 10010100b;
        shufps		xmm6, xmm6, 11101001b;
        pmaddwd		xmm5, xmm0;
        pmaddwd		xmm6, xmm0;
        shufpd		xmm3, xmm4, 1;
        movdqa		xmm4, xmm5;
        shufps		xmm5, xmm6, 0x88;
        shufps		xmm4, xmm6, 11011101b;
        pshufd		xmm6, xmm3, 1001b;
        paddd		xmm4, xmm5;
        punpcklbw	xmm6, xmm7;
        punpckhbw	xmm3, xmm7;
        shufps		xmm6, xmm6, 10010100b;
        shufps		xmm3, xmm3, 11101001b;
        pmaddwd		xmm6, xmm0;
        pmaddwd		xmm3, xmm0;
        movdqa		xmm5, xmm6;
        shufps		xmm6, xmm3, 0x88;
        shufps		xmm5, xmm3, 11011101b;
        paddd		xmm5, xmm6;
        psrad		xmm4, 10;
        psrad		xmm5, 10;
        packssdw	xmm4, xmm5;
        packuswb	xmm4, xmm4;
        movsd		[edi_ptr], xmm4;
        add			esi_ptr, 0x18;
        add			edi_ptr, 8;
        sub			eax_ptr, 8;
        jge			loop_8;
    loop_1_pre:
        add         eax_ptr, 8;
        jz          loop_end;
    loop_1:
        pinsrw      xmm4, [esi_ptr], 0;
        pinsrw      xmm4, [esi_ptr + 1], 1;
        punpcklbw	xmm4, xmm7;
        pshuflw     xmm4, xmm4, 110100b;
        pmaddwd		xmm4, xmm0;
        pshufd		xmm6, xmm4, 1;
        paddd		xmm4, xmm6;
        psrad		xmm4, 10;
        packssdw	xmm4, xmm4;
        packuswb	xmm4, xmm4;
        movd		edx_ptr, xmm4;
        mov			[edi_ptr], dl;
        add         esi_ptr, 3;
        inc         edi_ptr;
        dec         eax_ptr;
        jnz         loop_1;
    loop_end:
    }
}

decl_align(float, 16, coefBGR_f[4]) = { 0.114020904255103f, 0.587043074451121f, 0.298936021293775f, 0.f};
decl_align(float, 16, coefRound_f[4]) = { 0.5f, 0.5f, 0.5f, 0.5f};
void rgb2gray_f_sse2(OUT unsigned char* gray, IN unsigned char const* rgb, IN int count)
{
    __asm
    {
        movsxd      eax_ptr, count;
        mov         edi_ptr, gray;
        mov         esi_ptr, rgb;
        pxor        xmm0, xmm0;
        movaps      xmm6, coefBGR_f;
        movaps      xmm7, coefRound_f;
        sub         eax_ptr, 4;
        jl          loop_1_pre;
    loop_4:
        movsd       xmm1, [esi_ptr];
        movd        xmm2, [esi_ptr + 8];
        punpcklbw   xmm1, xmm0;
        punpcklbw   xmm2, xmm0;
        movdqa      xmm4, xmm1;
        pshufd      xmm3, xmm1, 1001b;
        shufps      xmm4, xmm2, 11b;
        pshuflw     xmm3, xmm3, 111001b;
		shufps		xmm4, xmm4, 8;
        pshuflw     xmm2, xmm2, 111001b;
        punpcklbw   xmm1, xmm0;
        punpcklbw   xmm3, xmm0;
        punpcklbw   xmm4, xmm0;
        punpcklbw   xmm2, xmm0;
        cvtdq2ps    xmm1, xmm1;
        cvtdq2ps    xmm3, xmm3;
        cvtdq2ps    xmm4, xmm4;
        cvtdq2ps    xmm2, xmm2;
        mulps       xmm1, xmm6;
        mulps       xmm3, xmm6;
        mulps       xmm4, xmm6;
        mulps       xmm2, xmm6;
        movaps      xmm5, xmm1;
        shufpd      xmm1, xmm3, 0;
        shufpd      xmm5, xmm3, 3;
        addps       xmm1, xmm5;
        movaps      xmm3, xmm4;
        shufpd      xmm4, xmm2, 0;
        shufpd      xmm3, xmm2, 3;
        addps       xmm4, xmm3;
        movaps      xmm2, xmm1;
        shufps      xmm1, xmm4, 0x88;
        shufps      xmm2, xmm4, 0xdd;
        addps       xmm1, xmm2;
        addps       xmm1, xmm7;
        cvttps2dq   xmm1, xmm1;
        packssdw    xmm1, xmm1;
        packuswb    xmm1, xmm1;
        movd        [edi_ptr], xmm1;
        add         esi_ptr, 12;
        add         edi_ptr, 4;
        sub         eax_ptr, 4;
        jge         loop_4;
    loop_1_pre:
        add         eax_ptr, 4;
        jz          loop_end;
    loop_1:
        pinsrw      xmm1, [esi_ptr], 0;
        pinsrw      xmm1, [esi_ptr + 1], 1;
        punpcklbw   xmm1, xmm0;
        pshuflw     xmm1, xmm1, 110100b;
        punpcklwd   xmm1, xmm0;
        cvtdq2ps    xmm1, xmm1;
        mulps       xmm1, xmm6;
        movhlps     xmm2, xmm1;
        addps       xmm1, xmm2;
        pshufd      xmm2, xmm1, 1;
        addss       xmm1, xmm7;
        addss       xmm1, xmm2;
        cvttss2si   ecx_ptr, xmm1;
        mov         [edi_ptr], cl;
        add         esi_ptr, 3;
        inc         edi_ptr;
        dec         eax_ptr;
        jnz         loop_1;
    loop_end:
    }
}

void get_gr_channel_sse2(unsigned char* green, unsigned char* red, unsigned char const* bgr, int count)
{
    __asm
    {
        movsxd      eax_ptr, count;
        mov         ebx_ptr, green;
        mov         ecx_ptr, red;
        mov         esi_ptr, bgr;
        pcmpeqb     xmm0, xmm0;
        psrld       xmm0, 24;
        sub         eax_ptr, 8;
        jl          loop_1_pre;
    loop_8:
        movsd       xmm1, [esi_ptr];
        movdqu      xmm2, [esi_ptr + 8];
        shufpd      xmm1, xmm2, 0;
        shufps      xmm1, xmm1, 0x94;
        shufps      xmm2, xmm2, 0xe9;
        pshuflw     xmm1, xmm1, 0x94;
        pshuflw     xmm2, xmm2, 0x94;
        pshufhw     xmm1, xmm1, 0xe9;
        pshufhw     xmm2, xmm2, 0xe9;
        movaps      xmm3, xmm1;
        shufps      xmm1, xmm2, 0x88;
        shufps      xmm3, xmm2, 0xdd;
        movaps      xmm2, xmm1;
        movaps      xmm4, xmm3;
        psrld       xmm1, 8;
        psrld       xmm2, 16;
        psrld       xmm3, 16;
        psrld       xmm4, 24;
        pand        xmm1, xmm0;
        pand        xmm2, xmm0;
        pand        xmm3, xmm0;
        pand        xmm4, xmm0;
		packssdw	xmm1, xmm2;
		packssdw	xmm3, xmm4;
		psllw		xmm3, 8;
		por			xmm1, xmm3;
        movsd       [ebx_ptr], xmm1;
        movhps      [ecx_ptr], xmm1;
        add         esi_ptr, 24;
        add         ebx_ptr, 8;
        add         ecx_ptr, 8;
        sub         eax_ptr, 8;
        jge         loop_8;
    loop_1_pre:
        add         eax_ptr, 8;
        jz          loop_end;
    loop_1:
        movzx       edx_ptr, word ptr [esi_ptr];
        mov         [ebx_ptr], dh;
        movzx       edx_ptr, byte ptr [esi_ptr + 2];
        mov         [ecx_ptr], dl;
        add         esi_ptr, 3;
        inc         ebx_ptr;
        inc         ecx_ptr;
        dec         eax_ptr;
        jnz         loop_1;
    loop_end:
    }
}

void gray2binary(unsigned char* binary, unsigned char const* gray, int threshold, int count)
{
    __asm
    {
        movsxd      eax_ptr, count;
        mov         edi_ptr, binary;
        mov         esi_ptr, gray;
        movd        xmm0, threshold;
        pxor        xmm7, xmm7;
        shufps      xmm0, xmm0, 0;
        packssdw    xmm0, xmm0;
        packuswb    xmm0, xmm0;
        sub         eax_ptr, 16;
        jl          loop_1_pre;
    loop_16:
        movdqu      xmm1, [esi_ptr];
        movdqa      xmm2, xmm0;
        psubusb     xmm2, xmm1;
        pcmpeqb     xmm2, xmm7;
        movdqu      [edi_ptr], xmm2;
        add         esi_ptr, 16;
        add         edi_ptr, 16;
        sub         eax_ptr, 16;
        jge         loop_16;
    loop_1_pre:
        add         eax_ptr, 16;
        jz          loop_end;
        mov         ebx_ptr, 255;
    loop_1:
        movzx       ecx_ptr, byte ptr [esi_ptr];
        xor         edx_ptr, edx_ptr;
        cmp         ecx, threshold;
        cmovae      edx_ptr, ebx_ptr;
        mov         [edi_ptr], dl;
        inc         esi_ptr;
        inc         edi_ptr;
        dec         eax_ptr;
        jnz         loop_1;
    loop_end:
    }
}

double dotproduct_d(double const* src1, double const* src2, int count)
{
    double val = 0;
    __asm
    {
        movsxd      eax_ptr, count;
        mov         esi_ptr, src1;
        mov         edi_ptr, src2;
        xorpd       xmm0, xmm0;
        sub         eax_ptr, 4;
        jl          loop_1_pre;
    loop_4:
        movupd      xmm1, [esi_ptr];
        movupd      xmm2, [edi_ptr];
        movupd      xmm3, [esi_ptr + 0x10];
        movupd      xmm4, [edi_ptr + 0x10];
        mulpd       xmm1, xmm2;
        mulpd       xmm3, xmm4;
        addpd       xmm0, xmm1;
        addpd       xmm0, xmm3;
        add         esi_ptr, 0x20;
        add         edi_ptr, 0x20;
        sub         eax_ptr, 4;
        jge         loop_4;
    loop_1_pre:
        add         eax_ptr, 4;
        jz          loop_end;
        movsd       xmm1, [esi_ptr];
        mulsd       xmm1, [edi_ptr];
        addsd       xmm0, xmm1;
    loop_end:
        movhlps     xmm1, xmm0;
        addsd       xmm0, xmm1;
        movsd       val, xmm0;
    }
    return val;
}

decl_align(long long, 16, resampleIdx[2]) = { 0, 1 };
decl_align(long long, 16, resampleStep[2]) = { 2, 2 };
void resample_linear_8_line(unsigned char* des, unsigned char const* src, intptr_t stride, int fy, int kx, int width)
{
	__asm
	{
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		movsxd		eax_ptr, width;
		movd		xmm0, kx;
		movd		xmm1, fy;
		movdqa		xmm2, resampleIdx;
		shufpd		xmm0, xmm0, 0;
		pshuflw		xmm1, xmm1, 0;
		sub			eax_ptr, 2;
		jl			loop_1;
	loop_2:
		movdqa		xmm3, xmm2;
		pmuludq		xmm3, xmm0;
		movdqa		xmm4, xmm3;
		psrlq		xmm3, 15;
		pextrw		ecx_ptr, xmm3, 0;
		pextrw		edx_ptr, xmm3, 4;
		psllq		xmm3, 15;
		psubq		xmm4, xmm3;
		pxor		xmm5, xmm5;
		pinsrw		xmm6, [esi_ptr + ecx_ptr], 0;
		pinsrw		xmm6, [esi_ptr + edx_ptr], 1;
		add			ecx_ptr, stride;
		add			edx_ptr, stride;
		pinsrw		xmm7, [esi_ptr + ecx_ptr], 0;
		pinsrw		xmm7, [esi_ptr + edx_ptr], 1;
		punpcklbw	xmm6, xmm5;
		punpcklbw	xmm7, xmm5;
		shufps		xmm4, xmm4, 8;
		pshuflw		xmm3, xmm4, 0xa0;	// fx
		psubsw		xmm7, xmm6;
		movdqa		xmm4, xmm3;
		psllw		xmm4, 1;
		pmulhuw		xmm4, xmm1;	// fxy
		psllw		xmm7, 7;
		psllw		xmm6, 7;
		movdqa		xmm5, xmm7;
		pmulhw		xmm7, xmm4;
		pmulhw		xmm5, xmm1;
		movdqa		xmm4, xmm7;
		psrld		xmm7, 16;
		psubsw		xmm5, xmm4;
		paddsw		xmm5, xmm7;
		movdqa		xmm7, xmm6;
		psrld		xmm6, 16;
		psubsw		xmm6, xmm7;
		psrlw		xmm7, 1;
		pmulhw		xmm6, xmm3;
		paddsw		xmm5, xmm6;
		paddsw		xmm5, xmm7;
		psraw		xmm5, 6;
		pshuflw		xmm5, xmm5, 8;
		packuswb	xmm5, xmm5;
		movd		edx_ptr, xmm5;
		mov			[edi_ptr], dx;
		paddq		xmm2, resampleStep;
		add			edi_ptr, 2;
		sub			eax_ptr, 2;
		jge			loop_2;
	loop_1:
		add			eax_ptr, 2;
		jz			loop_end;
		movdqa		xmm3, xmm2;
		pmuludq		xmm3, xmm0;
		movdqa		xmm4, xmm3;
		psrlq		xmm3, 15;
		pextrw		ecx_ptr, xmm3, 0;
		psllq		xmm3, 15;
		psubq		xmm4, xmm3;
		pxor		xmm5, xmm5;
		pinsrw		xmm6, [esi_ptr + ecx_ptr], 0;
		add			ecx_ptr, stride;
		pinsrw		xmm7, [esi_ptr + ecx_ptr], 0;
		punpcklbw	xmm6, xmm5;
		punpcklbw	xmm7, xmm5;
		shufps		xmm4, xmm4, 8;
		pshuflw		xmm3, xmm4, 0xa0;	// fx
		psubsw		xmm7, xmm6;
		movdqa		xmm4, xmm3;
		psllw		xmm4, 1;
		pmulhuw		xmm4, xmm1;	// fxy
		psllw		xmm7, 7;
		psllw		xmm6, 7;
		movdqa		xmm5, xmm7;
		pmulhw		xmm7, xmm4;
		pmulhw		xmm5, xmm1;
		movdqa		xmm4, xmm7;
		psrld		xmm7, 16;
		psubsw		xmm5, xmm4;
		paddsw		xmm5, xmm7;
		movdqa		xmm7, xmm6;
		psrld		xmm6, 16;
		psubsw		xmm6, xmm7;
		psrlw		xmm7, 1;
		pmulhw		xmm6, xmm3;
		paddsw		xmm5, xmm6;
		paddsw		xmm5, xmm7;
		psraw		xmm5, 6;
		pshuflw		xmm5, xmm5, 8;
		packuswb	xmm5, xmm5;
		movd		edx_ptr, xmm5;
		mov			[edi_ptr], dl;
	loop_end:
	}
}

void resample_linear_24_line(unsigned char* des, unsigned char const* src, intptr_t stride, int fy, int kx, int width)
{
	__asm
	{
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		xor			ebx, ebx;
		movd		xmm0, fy;
		pshuflw		xmm0, xmm0, 0;
		shufpd		xmm0, xmm0, 0;
		pxor		xmm7, xmm7;
	loop_1:
		cmp			ebx, width;
		jae			loop_end;
		mov			eax, ebx;
		mul			kx;
		mov			ecx, eax;
		shr			eax, 15;
		movsxd		edx_ptr, eax;
		shl			eax, 15;
		sub			ecx, eax;
		mov			eax, ecx;
		push		edx_ptr;
		mul			fy;
		pop			edx_ptr;
		shr			eax, 15;
		movd		xmm3, eax;
		movd		xmm4, fy;
		movd		xmm5, ecx;
		mov			eax_ptr, edx_ptr;
		add			edx_ptr, edx_ptr;
		add			edx_ptr, eax_ptr; 
		movd		xmm1, [esi_ptr + edx_ptr];
		pinsrw		xmm1, [esi_ptr + edx_ptr + 4], 2;
		add			edx_ptr, stride;
		movd		xmm2, [esi_ptr + edx_ptr];
		pinsrw		xmm2, [esi_ptr + edx_ptr + 4], 2;
		punpcklbw	xmm1, xmm7;
		punpcklbw	xmm2, xmm7;
		shufps		xmm1, xmm1, 0x94;
		shufps		xmm2, xmm2, 0x94;
		pshufhw		xmm1, xmm1, 111001b;
		pshufhw		xmm2, xmm2, 111001b;
		psllw		xmm1, 7;
		psllw		xmm2, 7;
		pshuflw		xmm3, xmm3, 0;
		pshuflw		xmm4, xmm4, 0;
		pshuflw		xmm5, xmm5, 0;
		psubsw		xmm2, xmm1;
		shufpd		xmm3, xmm3, 0;
		movhlps		xmm6, xmm1;
		pmulhw		xmm4, xmm2;
		pmulhw		xmm3, xmm2;
		psubsw		xmm6, xmm1;
		movhlps		xmm2, xmm3;
		pmulhw		xmm6, xmm5;
		psrlw		xmm1, 1;
		psubsw		xmm2, xmm3;
		paddsw		xmm2, xmm4;
		paddsw		xmm2, xmm6;
		paddsw		xmm1, xmm2;
		psraw		xmm1, 6;
		packuswb	xmm1, xmm1;
		movd		eax_ptr, xmm1;
		mov			[edi_ptr], ax;
		shr			eax_ptr, 16;
		mov			[edi_ptr + 2], al;
		inc			ebx;
		add			edi_ptr, 3;
		jmp			loop_1;
	loop_end:
	}
}

decl_align(short, 16, round5[8]) = { 16, 16, 16, 16, 16, 16, 16, 16 };
void imfilter_3x3_line2(unsigned char* des, unsigned char const* src, intptr_t stride, short const coef[4], int width)
{
	__asm
	{
		movsxd		eax_ptr, width;
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		mov			ecx_ptr, stride;
		mov			ebx_ptr, coef;
		movsd		xmm0, [ebx_ptr];
		pshuflw		xmm1, xmm0, 0x55;
		pshuflw		xmm0, xmm0, 0;
		shufpd		xmm1, xmm1, 0;
		shufpd		xmm0, xmm0, 0;
		pxor		xmm7, xmm7;
		sub			eax_ptr, 8;
		jl			loop_1_pre;
loop_8:
		movsd		xmm3, [esi_ptr];
		movsd		xmm4, [esi_ptr + ecx_ptr];
		punpcklbw	xmm3, xmm7;
		punpcklbw	xmm4, xmm7;
		psllw		xmm3, 7;
		psllw		xmm4, 7;
		pmulhw		xmm3, xmm0;
		pmulhw		xmm4, xmm1;
		paddsw		xmm3, xmm4;
		paddsw		xmm3, round5;
		psraw		xmm3, 5;
		packuswb	xmm3, xmm3;
		movsd		[edi_ptr], xmm3;
		add			esi_ptr, 8;
		add			edi_ptr, 8;
		sub			eax_ptr, 8;
		jge			loop_8;
loop_1_pre:
		add			eax_ptr, 8;
		jz			loop_end;
loop_1:
		movzx		ebx_ptr, byte ptr [esi_ptr];
		movzx		edx_ptr, byte ptr [esi_ptr + ecx_ptr];
		movd		xmm3, ebx_ptr;
		movd		xmm4, edx_ptr;
		psllw		xmm3, 7;
		psllw		xmm4, 7;
		pmulhw		xmm3, xmm0;
		pmulhw		xmm4, xmm1;
		paddsw		xmm3, xmm4;
		paddsw		xmm3, round5;
		psraw		xmm3, 5;
		packuswb	xmm3, xmm3;
		movd		edx_ptr, xmm3;
		mov			[edi_ptr], dl;
		add			esi_ptr, 1;
		add			edi_ptr, 1;
		sub			eax_ptr, 1;
		jnz			loop_1;
loop_end:

	}
}

void imfilter_3x3_line3(unsigned char* des, unsigned char const* src, intptr_t stride, short const coef[4], int width)
{
	__asm
	{
		movsxd		eax_ptr, width;
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		mov			ecx_ptr, stride;
		mov			ebx_ptr, coef;
		movsd		xmm0, [ebx_ptr];
		pshuflw		xmm2, xmm0, 0;
		pshuflw		xmm1, xmm0, 0x55;
		pshuflw		xmm0, xmm0, 0xaa;
		shufpd		xmm2, xmm2, 0;
		shufpd		xmm1, xmm1, 0;
		shufpd		xmm0, xmm0, 0;
		pxor		xmm7, xmm7;
		sub			eax_ptr, 8;
		jl			loop_1_pre;
	loop_8:
		movsd		xmm3, [esi_ptr];
		movsd		xmm4, [esi_ptr + ecx_ptr];
		movsd		xmm5, [esi_ptr + ecx_ptr * 2];
		punpcklbw	xmm3, xmm7;
		punpcklbw	xmm4, xmm7;
		punpcklbw	xmm5, xmm7;
		psllw		xmm3, 7;
		psllw		xmm4, 7;
		psllw		xmm5, 7;
		pmulhw		xmm3, xmm0;
		pmulhw		xmm4, xmm1;
		pmulhw		xmm5, xmm2;
		paddsw		xmm3, xmm4;
		paddsw		xmm3, xmm5;
		paddsw		xmm3, round5;
		psraw		xmm3, 5;
		packuswb	xmm3, xmm3;
		movsd		[edi_ptr], xmm3;
		add			esi_ptr, 8;
		add			edi_ptr, 8;
		sub			eax_ptr, 8;
		jge			loop_8;
loop_1_pre:
		add			eax_ptr, 8;
		jz			loop_end;
loop_1:
		movzx		ebx_ptr, byte ptr [esi_ptr];
		movzx		edx_ptr, byte ptr [esi_ptr + ecx_ptr];
		movd		xmm3, ebx_ptr;
		movd		xmm4, edx_ptr;
		movzx		ebx_ptr, byte ptr [esi_ptr + ecx_ptr * 2];
		movd		xmm5, ebx_ptr;
		psllw		xmm3, 7;
		psllw		xmm4, 7;
		psllw		xmm5, 7;
		pmulhw		xmm3, xmm0;
		pmulhw		xmm4, xmm1;
		pmulhw		xmm5, xmm2;
		paddsw		xmm3, xmm4;
		paddsw		xmm3, xmm5;
		paddsw		xmm3, round5;
		psraw		xmm3, 5;
		packuswb	xmm3, xmm3;
		movd		edx_ptr, xmm3;
		mov			[edi_ptr], dl;
		add			esi_ptr, 1;
		add			edi_ptr, 1;
		sub			eax_ptr, 1;
		jnz			loop_1;
	loop_end:
	}
}

decl_align(int, 16, bit0Mask[4]) = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
size_t calccnt8_eq_sse2(unsigned char* src, int val, size_t count)
{
	size_t cnt;
	__asm
	{
		mov			esi_ptr, src;
		mov			eax_ptr, count;
		xor			ecx_ptr, ecx_ptr;
		movd		xmm0, val;
		pxor		xmm7, xmm7;
		pshuflw		xmm0, xmm0, 0;
		shufpd		xmm0, xmm0, 0;
		packuswb	xmm0, xmm0;
		sub			eax_ptr, 16;
		jl			loop_1_pre;
loop_16:
		movdqu		xmm1, [esi_ptr];
		pcmpeqb		xmm1, xmm0;
		pand		xmm1, bit0Mask;
		movhlps		xmm3, xmm1;
		paddb		xmm1, xmm3;
		pshufd		xmm3, xmm1, 1;
		paddb		xmm1, xmm3;
		pshuflw		xmm3, xmm1, 1;
		paddb		xmm1, xmm3;
		movd		edx_ptr, xmm1;
		add			dl, dh;
		and			edx_ptr, 0xff;
		add			ecx_ptr, edx_ptr;
		add			esi_ptr, 16;
		sub			eax_ptr, 16;
		jge			loop_16;
loop_1_pre:
		mov			cnt, ecx_ptr;
		add			eax_ptr, 16;
		jz			loop_end;
		mov			ecx_ptr, 1;
loop_1:
		movzx		edx_ptr, byte ptr [esi_ptr];
		xor			edi_ptr, edi_ptr;
		cmp			edx, val;
		cmove		edi_ptr, ecx_ptr;
		add			cnt, edi_ptr;
		inc			esi_ptr;
		dec			eax_ptr;
		jnz			loop_1;
loop_end:
	}
	return cnt;
}

void calccnt8_ver_sse2(int* des, unsigned char* src, intptr_t stride, int height)
{
	__asm
	{
		movsxd		eax_ptr, height;
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		mov			ecx_ptr, stride;
		mov			edx_ptr, ecx_ptr;
		add			ecx_ptr, ecx_ptr;
		pxor		xmm0, xmm0;
		pxor		xmm1, xmm1;
		pxor		xmm7, xmm7;
		sub			eax_ptr, 255;
		jl			loop_1_pre;
loop_255:
		pxor		xmm6, xmm6;
		mov			ebx_ptr, 127;
loop_255_2:
		movdqu		xmm4, [esi_ptr];
		movdqu		xmm5, [esi_ptr + edx_ptr];
		pcmpeqb		xmm4, xmm7;
		pcmpeqb		xmm5, xmm7;
		pandn		xmm4, bit0Mask;
		pandn		xmm5, bit0Mask;
		paddb		xmm6, xmm4;
		paddb		xmm6, xmm5;
		add			esi_ptr, ecx_ptr;
		dec			ebx_ptr;
		jnz			loop_255_2;
		movdqu		xmm4, [esi_ptr];
		pcmpeqb		xmm4, xmm7;
		pandn		xmm4, bit0Mask;
		paddb		xmm6, xmm4;
		movdqa		xmm4, xmm6;
		punpcklbw	xmm4, xmm7;
		punpckhbw	xmm6, xmm7;
		paddw		xmm0, xmm4;
		paddw		xmm1, xmm6;
		add			esi_ptr, edx_ptr;
		sub			eax_ptr, 255;
		jge			loop_255;
loop_1_pre:
		add			eax_ptr, 255;
		jz			loop_end;
		pxor		xmm6, xmm6;
loop_1:
		movdqu		xmm4, [esi_ptr];
		pcmpeqb		xmm4, xmm7;
		pandn		xmm4, bit0Mask;
		paddb		xmm6, xmm4;
		add			esi_ptr, edx_ptr;
		dec			eax_ptr;
		jnz			loop_1;
		movdqa		xmm4, xmm6;
		punpcklbw	xmm4, xmm7;
		punpckhbw	xmm6, xmm7;
		paddw		xmm0, xmm4;
		paddw		xmm1, xmm6;
loop_end:
		movdqa		xmm2, xmm0;
		movdqa		xmm3, xmm1;
		punpcklwd	xmm0, xmm7;
		punpckhwd	xmm2, xmm7;
		punpcklwd	xmm1, xmm7;
		punpckhwd	xmm3, xmm7;
		movdqu		[edi_ptr], xmm0;
		movdqu		[edi_ptr + 16], xmm2;
		movdqu		[edi_ptr + 32], xmm1;
		movdqu		[edi_ptr + 48], xmm3;
	}
}

decl_align(double, 16, dInv255[2]) = {1.0/255, 1.0/255};
void ucharnorm2double_sse2(OUT double* des, IN unsigned char const* src, IN size_t count)
{
	__asm {
		mov			esi_ptr, src;
		mov			ecx_ptr, count;
		mov			edi_ptr, des;
		movapd		xmm0, dInv255;
		xorpd		xmm1, xmm1;
		sub			ecx_ptr, 8;
		jl			loop_1_pre;
loop_8:
		movsd		xmm2, [esi_ptr];
		punpcklbw	xmm2, xmm1;
		movapd		xmm3, xmm2;
		punpcklwd	xmm2, xmm1;
		punpckhwd	xmm3, xmm1;
		movhlps		xmm4, xmm2;
		movhlps		xmm5, xmm3;
		cvtdq2pd	xmm2, xmm2;
		cvtdq2pd	xmm4, xmm4;
		cvtdq2pd	xmm3, xmm3;
		cvtdq2pd	xmm5, xmm5;
		mulpd		xmm2, xmm0;
		mulpd		xmm4, xmm0;
		mulpd		xmm3, xmm0;
		mulpd		xmm5, xmm0;
		movupd		[edi_ptr], xmm2;
		movupd		[edi_ptr + 0x10], xmm4;
		movupd		[edi_ptr + 0x20], xmm3;
		movupd		[edi_ptr + 0x30], xmm5;
		add			esi_ptr, 8;
		add			edi_ptr, 0x40;
		sub			ecx_ptr, 8;
		jge			loop_8;
loop_1_pre:
		add			ecx_ptr, 8;
		jz			loop_end;
loop_1:
		movzx		eax_ptr, byte ptr [esi_ptr];
		cvtsi2sd	xmm2, eax_ptr;
		mulsd		xmm2, xmm0;
		movsd		[edi_ptr], xmm2;
		add			esi_ptr, 1;
		add			edi_ptr, 8;
		dec			ecx_ptr;
		jnz			loop_1;
loop_end:
	}
}

decl_align(float, 16, fInv255[4]) = {1.0f/255, 1.0f/255, 1.0f/255, 1.0f/255};
void ucharnorm2float_sse2(OUT float* des, IN unsigned char const* src, IN size_t count)
{
	__asm {
		mov			esi_ptr, src;
		mov			ecx_ptr, count;
		mov			edi_ptr, des;
		movaps		xmm0, fInv255;
		xorps		xmm1, xmm1;
		sub			ecx_ptr, 8;
		jl			loop_1_pre;
loop_8:
		movsd		xmm2, [esi_ptr];
		punpcklbw	xmm2, xmm1;
		movaps		xmm3, xmm2;
		punpcklwd	xmm2, xmm1;
		punpckhwd	xmm3, xmm1;
		cvtdq2ps	xmm2, xmm2;
		cvtdq2ps	xmm3, xmm3;
		mulps		xmm2, xmm0;
		mulps		xmm3, xmm0;
		movups		[edi_ptr], xmm2;
		movups		[edi_ptr + 0x10], xmm3;
		add			esi_ptr, 8;
		add			edi_ptr, 0x20;
		sub			ecx_ptr, 8;
		jge			loop_8;
loop_1_pre:
		add			ecx_ptr, 8;
		jz			loop_end;
loop_1:
		movzx		eax_ptr, byte ptr [esi_ptr];
		cvtsi2ss	xmm2, eax_ptr;
		mulss		xmm2, xmm0;
		movss		[edi_ptr], xmm2;
		add			esi_ptr, 1;
		add			edi_ptr, 4;
		dec			ecx_ptr;
		jnz			loop_1;
loop_end:
	}
}

void dmemconv(double* des, double const* src, double const* kernel, int kernel_size, int count)
{
	__asm
	{
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		movsxd		ecx_ptr, count;
		test		ecx_ptr, ecx_ptr;
		jz			loop_end;
loop_1:
		xorpd		xmm0, xmm0;
		movsxd		ebx_ptr, kernel_size;
		mov			eax_ptr, kernel;
		mov			edx_ptr, esi_ptr;
		sub			ebx_ptr, 2;
		jl			loop_1_1;
loop_1_2:
		movupd		xmm1, [edx_ptr];
		movupd		xmm2, [eax_ptr];
		mulpd		xmm1, xmm2;
		addpd		xmm0, xmm1;
		add			edx_ptr, 0x10;
		add			eax_ptr, 0x10;
		sub			ebx_ptr, 2;
		jge			loop_1_2;
		movhlps		xmm1, xmm0;
		addsd		xmm0, xmm1;
loop_1_1:
		add			ebx_ptr, 2;
		jz			loop_1_end;
		movsd		xmm1, [edx_ptr];
		movsd		xmm2, [eax_ptr];
		mulsd		xmm1, xmm2;
		addsd		xmm0, xmm1;
loop_1_end:
		movsd		[edi_ptr], xmm0;
		add			esi_ptr, 8;
		add			edi_ptr, 8;
		dec			ecx_ptr;
		jnz			loop_1;
loop_end:
	}
}

void dmemmul(double* des, double const* src, double weight, int count)
{
	__asm
	{
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		movsd		xmm0, weight;
		movsxd		eax_ptr, count;
		shufpd		xmm0, xmm0, 0;
		sub			eax_ptr, 2;
		jl			loop_1;
loop_2:
		movupd		xmm1, [esi_ptr];
		mulpd		xmm1, xmm0;
		movupd		[edi_ptr], xmm1;
		add			esi_ptr, 0x10;
		add			edi_ptr, 0x10;
		sub			eax_ptr, 2;
		jge			loop_2;
loop_1:
		add			eax_ptr, 2;
		jz			loop_end;
		movsd		xmm1, [esi_ptr];
		mulsd		xmm1, xmm0;
		movsd		[edi_ptr], xmm1;
loop_end:
	}
}

void dmemmad(double* des, double const* src, double weight, int count)
{
	__asm
	{
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		movsd		xmm0, weight;
		movsxd		eax_ptr, count;
		shufpd		xmm0, xmm0, 0;
		sub			eax_ptr, 2;
		jl			loop_1;
loop_2:
		movupd		xmm1, [esi_ptr];
		movupd		xmm2, [edi_ptr];
		mulpd		xmm1, xmm0;
		addpd		xmm1, xmm2;
		movupd		[edi_ptr], xmm1;
		add			esi_ptr, 0x10;
		add			edi_ptr, 0x10;
		sub			eax_ptr, 2;
		jge			loop_2;
loop_1:
		add			eax_ptr, 2;
		jz			loop_end;
		movsd		xmm1, [esi_ptr];
		movsd		xmm2, [edi_ptr];
		mulsd		xmm1, xmm0;
		addsd		xmm1, xmm2;
		movsd		[edi_ptr], xmm1;
loop_end:
	}
}

void nsp_filter(OUT double* des,			// 滤波后输出缓冲
				IN double const* src,		// 输入缓冲
				IN double const* kernel,	// 一维核系数缓冲
				IN int kernel_size,			// 一维核长度
				IN int direction,			// 0-水平，1-垂直，2-水平垂直
				IN int width,				// 宽度
				IN int height)				// 高度
{
	int half_ker = (kernel_size - 1) >> 1;
	if(direction == 0)
	{
		for(int i=0; i<height; ++i)
		{
			for(int k=0; k<half_ker; ++k)
			{
				double _Sum = src[0] * std::accumulate(kernel, kernel+half_ker-k+1, 0.0);
				for(int j=half_ker-k+1; j<kernel_size; ++j)
					_Sum += kernel[j]*src[j-half_ker+k];
				des[k] = _Sum;
			}
			dmemconv(des + half_ker, src, kernel, kernel_size, width - kernel_size + 1);
			for(int k=width-kernel_size/2; k<width; ++k)
			{
				double _Sum = (double)0;
				for(int j=0; j<kernel_size; ++j)
					_Sum += kernel[j]*src[std::min(width-1, k+j-half_ker)];
				des[k] = _Sum;
			}
			des += width;
			src += width;
		}
	}
	else if(direction == 1)
	{
		for(int k=0; k<half_ker; ++k)
		{
			double const* _tmp = src;
			dmemmul(des, _tmp, std::accumulate(kernel, kernel+half_ker-k+1, 0.0), width);
			_tmp += width;
			for(int j=half_ker-k+1; j<kernel_size; ++j, _tmp+=width)
				dmemmad(des, _tmp, kernel[j], width);
			des += width;
		}
		for(int k=half_ker; k<height-half_ker; ++k)
		{
			double const* _tmp = src;
			dmemmul(des, _tmp, kernel[0], width);
			_tmp += width;
			for(int j=1; j<kernel_size; ++j, _tmp+=width)
				dmemmad(des, _tmp, kernel[j], width);
			des += width;
			src += width;
		}
		for(int k=height-kernel_size/2; k<height; ++k)
		{
			double const* _tmp = src;
			dmemmul(des, _tmp, kernel[0], width);
			_tmp += width;
			int _Len = std::min(height-1, k+kernel_size/2) - k + half_ker;
			for(int j=1; j<_Len; ++j, _tmp+=width)
				dmemmad(des, _tmp, kernel[j], width);
			dmemmad(des, _tmp, std::accumulate(kernel+_Len, kernel+kernel_size, 0.0), width);
			des += width;
			src += width;
		}
	}
}

unsigned __int64* qmemset_sse2(OUT unsigned __int64* des, IN unsigned __int64 qword_value, IN size_t qword_count)
{
	__asm
	{
		mov			ecx_ptr, qword_count;
		movsd		xmm0, qword_value;
		mov			edi_ptr, des;
		mov			eax_ptr, ecx_ptr;
		shufpd		xmm0, xmm0, 0;
		shr			ecx_ptr, 4;	// 一次计算16个
		jz			loop_8;
		test		edi_ptr, 0x0f;
		jz			loop_16a;
		test		edi_ptr, 7;	// 没有对齐到8字节边界
		jnz			loop_16u;
		dec			eax_ptr;
		movsd		[edi_ptr], xmm0;
		mov			ecx_ptr, eax_ptr;
		add			edi_ptr, 8;
		shr			ecx_ptr, 4;
		jz			loop_8;
loop_16a:
		movntpd		[edi_ptr], xmm0;
		movntpd		[edi_ptr + 0x10], xmm0;
		movntpd		[edi_ptr + 0x20], xmm0;
		movntpd		[edi_ptr + 0x30], xmm0;
		movntpd		[edi_ptr + 0x40], xmm0;
		movntpd		[edi_ptr + 0x50], xmm0;
		movntpd		[edi_ptr + 0x60], xmm0;
		movntpd		[edi_ptr + 0x70], xmm0;
		add			edi_ptr, 0x80;
		dec			ecx_ptr;
		jnz			loop_16a;
		sfence;
		jmp			loop_8;
loop_16u:
		movupd		[edi_ptr], xmm0;
		movupd		[edi_ptr + 0x10], xmm0;
		movupd		[edi_ptr + 0x20], xmm0;
		movupd		[edi_ptr + 0x30], xmm0;
		movupd		[edi_ptr + 0x40], xmm0;
		movupd		[edi_ptr + 0x50], xmm0;
		movupd		[edi_ptr + 0x60], xmm0;
		movupd		[edi_ptr + 0x70], xmm0;
		add			edi_ptr, 0x80;
		dec			ecx_ptr;
		jnz			loop_16u;
loop_8:
		test		eax_ptr, 8;
		jz			loop_4;
		movupd		[edi_ptr], xmm0;
		movupd		[edi_ptr + 0x10], xmm0;
		movupd		[edi_ptr + 0x20], xmm0;
		movupd		[edi_ptr + 0x30], xmm0;
		add			edi_ptr, 0x40;
loop_4:
		test		eax_ptr, 4;
		jz			loop_2;
		movupd		[edi_ptr], xmm0;
		movupd		[edi_ptr + 0x10], xmm0;
		add			edi_ptr, 0x20;
loop_2:
		test		eax_ptr, 2;
		jz			loop_1;
		movupd		[edi_ptr], xmm0;
		add			edi_ptr, 0x10;
loop_1:
		test		eax_ptr, 1;
		jz			loop_end;
		movsd		[edi_ptr], xmm0;
loop_end:
	}
	return des;
}

void dbmemgain_sse2(OUT double *des, IN double *src, IN double dGain, IN size_t count)
{
	__asm {
		mov			ecx_ptr, count;
		mov			esi_ptr, src;
		mov			edi_ptr, des;
		movlps		xmm7, dGain;
		// 预取一部分的源数据到离处理器较近的 Cache 中，减少 Cache 污染，并提高命中
		prefetchnta	byte ptr [esi_ptr];
		movlhps		xmm7, xmm7;
		shr			ecx_ptr, 0x03;		// 一次处理 8 个
		jnz			loop_8;
		jmp			loop_m4;
loop_8:
		// 预取一部分的源数据到离处理器较近的 Cache 中，减少 Cache 污染，并提高命中
		prefetchnta	byte ptr [esi_ptr + 0x40];
		movups		xmm0, [esi_ptr];
		movups		xmm1, [esi_ptr + 0x10];
		movups		xmm2, [esi_ptr + 0x20];
		movups		xmm3, [esi_ptr + 0x30];
		mulpd		xmm0, xmm7;
		mulpd		xmm1, xmm7;
		mulpd		xmm2, xmm7;
		mulpd		xmm3, xmm7;
		movups		[edi_ptr], xmm0;
		movups		[edi_ptr + 0x10], xmm1;
		movups		[edi_ptr + 0x20], xmm2;
		movups		[edi_ptr + 0x30], xmm3;
		add			esi_ptr, 0x40;
		add			edi_ptr, 0x40;
		sub			ecx_ptr, 0x01;
		jnz			loop_8;
loop_m4:
		test		count, 0x04;
		jz			loop_trails;
		movups		xmm0, [esi_ptr];
		movups		xmm1, [esi_ptr + 0x10];
		mulpd		xmm0, xmm7;
		mulpd		xmm1, xmm7;
		movups		[edi_ptr], xmm0;
		movups		[edi_ptr + 0x10], xmm1;
		add			esi_ptr, 0x20;
		add			edi_ptr, 0x20;
loop_trails:
		mov			ecx_ptr, count;
		and			ecx_ptr, 0x03;
		jz			loop_end;
loop_1:
		movlps		xmm0, [esi_ptr];
		mulsd		xmm0, xmm7;
		movlps		[edi_ptr], xmm0;
		add			esi_ptr, 0x08;
		add			edi_ptr, 0x08;
		sub			ecx_ptr, 0x01;
		jnz			loop_1;
loop_end:
	}
}

void nsp_calc_norm_magnitude_d(OUT double* magnitude,	// 归一化梯度
							   IN double const* diffX,	// X 方向导数
							   IN double const* diffY,	// Y 方向导数
							   IN int count)
{
	double max_mag = 0;
	__asm
	{
		mov			edi_ptr, magnitude;
		mov			ecx_ptr, diffX;
		mov			edx_ptr, diffY;
		movsxd		esi_ptr, count;
		xorpd		xmm0, xmm0;
		sub			esi_ptr, 2;
		jl			loop_1;
loop_2:
		movupd		xmm1, [ecx_ptr];
		movupd		xmm2, [edx_ptr];
		mulpd		xmm1, xmm1;
		mulpd		xmm2, xmm2;
		addpd		xmm1, xmm2;
		sqrtpd		xmm1, xmm1;
		maxpd		xmm0, xmm1;
		movupd		[edi_ptr], xmm1;
		add			ecx_ptr, 0x10;
		add			edx_ptr, 0x10;
		add			edi_ptr, 0x10;
		sub			esi_ptr, 2;
		jge			loop_2;
		movhlps		xmm1, xmm0;
		maxpd		xmm0, xmm1;
loop_1:
		add			esi_ptr, 2;
		jz			loop_end;
		movsd		xmm1, [ecx_ptr];
		movsd		xmm2, [edx_ptr];
		mulpd		xmm1, xmm1;
		mulpd		xmm2, xmm2;
		addpd		xmm1, xmm2;
		sqrtpd		xmm1, xmm1;
		maxpd		xmm0, xmm1;
		movsd		[edi_ptr], xmm1;
loop_end:
		movsd		max_mag, xmm0;
	}
	if(max_mag > 0)
		dbmemgain_sse2(magnitude, magnitude, 1.0/max_mag, count);
}

void float2double_sse2(double* des, float const* src, size_t count)
{
	__asm
	{
		mov			ecx_ptr, count;
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		sub			ecx_ptr, 8;
		jl			loop_1_pre;
loop_8:
		movups		xmm0, [esi_ptr];
		movups		xmm1, [esi_ptr + 0x10];
		movhlps		xmm2, xmm0;
		movhlps		xmm3, xmm1;
		cvtps2pd	xmm0, xmm0;
		cvtps2pd	xmm2, xmm2;
		cvtps2pd	xmm1, xmm1;
		cvtps2pd	xmm3, xmm3;
		movupd		[edi_ptr], xmm0;
		movupd		[edi_ptr + 0x10], xmm2;
		movupd		[edi_ptr + 0x20], xmm1;
		movupd		[edi_ptr + 0x30], xmm3;
		add			esi_ptr, 0x20;
		add			edi_ptr, 0x40;
		sub			ecx_ptr, 8;
		jge			loop_8;
loop_1_pre:
		add			ecx_ptr, 8;
		jz			loop_end;
loop_1:
		cvtss2sd	xmm0, [esi_ptr];
		movsd		[edi_ptr], xmm0;
		add			esi_ptr, 4;
		add			edi_ptr, 8;
		dec			ecx_ptr;
		jnz			loop_1;
loop_end:
	}
}

inline void memcmpset(unsigned char* des, unsigned char* src, int count)
{
	__asm
	{
		movsxd		eax_ptr, count;
		mov			edi_ptr, des;
		mov			esi_ptr, src;
		pcmpeqb		xmm0, xmm0;
		sub			eax_ptr, 32;
		jl			loop_1_pre;
loop_32:
		movups		xmm1, [esi_ptr];
		movups		xmm2, [esi_ptr + 16];
		pcmpeqb		xmm1, xmm0;
		pcmpeqb		xmm2, xmm0;
		movntps		[edi_ptr], xmm1;
		movntps		[edi_ptr + 16], xmm2;
		add			esi_ptr, 32;
		add			edi_ptr, 32;
		sub			eax_ptr, 32;
		jge			loop_32;
		sfence;
loop_1_pre:
		add			eax_ptr, 32;
		jz			loop_end;
loop_1:
		movzx		ecx_ptr, byte ptr [esi_ptr];
		sar			cl, 8;
		mov			[edi_ptr], cl;
		inc			esi_ptr;
		inc			edi_ptr;
		dec			eax_ptr;
		jnz			loop_1;
loop_end:
	}
}

decl_align(float, 16, g_pfKernel1D[9]) = {5.33905354532819e-005f, 0.00176805171185202f, 
	0.0215392793018486f, 0.0965323526300539f, 0.159154943091895f, 
	0.0965323526300539f, 0.0215392793018486f, 0.00176805171185202f, 
	5.33905354532819e-005f};

decl_align(float, 16, g_pfKernel2D_X[81]) = { 
	1.43284234626241e-007f,     3.55869164115247e-006f,     2.89024929508973e-005f,     6.47659933817797e-005f, 0 ,   -6.47659933817797e-005f ,   -2.89024929508973e-005f,    -3.55869164115247e-006f, -1.43284234626241e-007f,
	4.7449221882033e-006f,      0.000117847682078385f,      0.000957119116801882f,       0.00214475514239131f,  0 ,     -0.00214475514239131f,     -0.000957119116801882f,     -0.000117847682078385f, -4.7449221882033e-006f,
	5.78049859017946e-005f,       0.00143567867520282f,        0.0116600978601128f,        0.0261284665693698f, 0 ,      -0.0261284665693698f,       -0.0116600978601128f,      -0.00143567867520282f, -5.78049859017946e-005f,
	0.000259063973527119f,       0.00643426542717393f,        0.0522569331387397f,         0.117099663048638f, 0 ,       -0.117099663048638f ,      -0.0522569331387397f,      -0.00643426542717393f,  -0.000259063973527119f,
	0.000427124283626255f,       0.0106083102711121f,        0.0861571172073945f,         0.193064705260108f, 0  ,      -0.193064705260108f ,      -0.0861571172073945f,       -0.0106083102711121f, -0.000427124283626255f,
	0.000259063973527119f,      0.00643426542717393f,        0.0522569331387397f,         0.117099663048638f, 0  ,      -0.117099663048638f,       -0.0522569331387397f,      -0.00643426542717393f, -0.000259063973527119f,
	5.78049859017946e-005f,       0.00143567867520282f,        0.0116600978601128f,        0.0261284665693698f, 0 ,      -0.0261284665693698f ,      -0.0116600978601128f,      -0.00143567867520282f, -5.78049859017946e-005f,
	4.7449221882033e-006f,      0.000117847682078385f,      0.000957119116801882f,       0.00214475514239131f, 0  ,    -0.00214475514239131f,     -0.000957119116801882f,     -0.000117847682078385f, -4.7449221882033e-006f,
	1.43284234626241e-007f,     3.55869164115247e-006f,     2.89024929508973e-005f,     6.47659933817797e-005f, 0,    -6.47659933817797e-005f,    -2.89024929508973e-005f,    -3.55869164115247e-006f, -1.43284234626241e-007f
};

decl_align(float, 16, g_pfKernel2D_Y[81]) = {
	1.43284234626241e-007f,      4.7449221882033e-006f,     5.78049859017946e-005f,      0.000259063973527119f,0.000427124283626255f,      0.000259063973527119f,     5.78049859017946e-005f,      4.7449221882033e-006f,1.43284234626241e-007f,
	3.55869164115247e-006f,      0.000117847682078385f,       0.00143567867520282f,       0.00643426542717393f,0.0106083102711121f,       0.00643426542717393f,       0.00143567867520282f ,     0.000117847682078385f,3.55869164115247e-006f,
	2.89024929508973e-005f,      0.000957119116801882f,        0.0116600978601128f,        0.0522569331387397f,0.0861571172073945f,        0.0522569331387397f,        0.0116600978601128f,      0.000957119116801882f,2.89024929508973e-005f,
	6.47659933817797e-005f,       0.00214475514239131f,        0.0261284665693698f,         0.117099663048638f,0.193064705260108f,         0.117099663048638f ,       0.0261284665693698f,       0.00214475514239131f,6.47659933817797e-005f,
	0,                          0,                           0 ,                        0,                 0,                         0     ,                    0             ,            0,                    0,
	-6.47659933817797e-005f,      -0.00214475514239131f,       -0.0261284665693698f,        -0.117099663048638f,-0.193064705260108f,        -0.117099663048638f,       -0.0261284665693698f,      -0.00214475514239131f,-6.47659933817797e-005f,
	-2.89024929508973e-005f,     -0.000957119116801882f,       -0.0116600978601128f,       -0.0522569331387397f, -0.0861571172073945f,       -0.0522569331387397f,       -0.0116600978601128f,     -0.000957119116801882f,-2.89024929508973e-005f,
	-3.55869164115247e-006f,     -0.000117847682078385f,      -0.00143567867520282f,      -0.00643426542717393f,-0.0106083102711121f,      -0.00643426542717393f,      -0.00143567867520282f,     -0.000117847682078385f,-3.55869164115247e-006f,
	-1.43284234626241e-007f,     -4.7449221882033e-006f,    -5.78049859017946e-005f,     -0.000259063973527119f,-0.000427124283626255f,     -0.000259063973527119f,    -5.78049859017946e-005f ,    -4.7449221882033e-006f,-1.43284234626241e-007f
};

decl_align(float, 16, s_fFactorRegon64[64]) = {
	0.00793650793651f,
	0.02380952380953f,
	0.03968253968255f,
	0.05555555555557f,
	0.07142857142859f,
	0.08730158730161f,
	0.10317460317463f,
	0.11904761904765f,
	0.13492063492067f,
	0.15079365079369f,
	0.16666666666671f,
	0.18253968253973f,
	0.19841269841275f,
	0.21428571428577f,
	0.23015873015879f,
	0.24603174603181f,
	0.26190476190483f,
	0.27777777777785f,
	0.29365079365087f,
	0.30952380952389f,
	0.32539682539691f,
	0.34126984126993f,
	0.35714285714295f,
	0.37301587301597f,
	0.38888888888899f,
	0.40476190476201f,
	0.42063492063503f,
	0.43650793650805f,
	0.45238095238107f,
	0.46825396825409f,
	0.48412698412711f,
	0.50000000000013f,
	0.51587301587315f,
	0.53174603174617f,
	0.54761904761919f,
	0.56349206349221f,
	0.57936507936523f,
	0.59523809523825f,
	0.61111111111127f,
	0.62698412698429f,
	0.64285714285731f,
	0.65873015873033f,
	0.67460317460335f,
	0.69047619047637f,
	0.70634920634939f,
	0.72222222222241f,
	0.73809523809543f,
	0.75396825396845f,
	0.76984126984147f,
	0.78571428571449f,
	0.80158730158751f,
	0.81746031746053f,
	0.83333333333355f,
	0.84920634920657f,
	0.86507936507959f,
	0.88095238095261f,
	0.89682539682563f,
	0.91269841269865f,
	0.92857142857167f,
	0.94444444444469f,
	0.96031746031771f,
	0.97619047619073f,
	0.99206349206375f,
	1.00793650793677f
};

decl_align(unsigned char, 16, bLUTArray1[512]) = 
{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,		
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1		
,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,		
1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1		
,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,		
1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1		
,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,		
1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1		
,1,1,1,1,1,1,1,1,1,1,1,1 };

//基于索引表的细化表2
decl_align(unsigned char, 16, bLUTArray2[512]) = 
{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1 };

// 0-Intel GPU，1-NVIDIA CUDA，2-AMD
bool init_platform(int platformID)
{
	static char const* const platformName[] = {
		"Intel",
		"NVIDIA CUDA",
		"Advanced Micro Devices"
	};

	cl_uint pidcount = 0;
	clGetPlatformIDs(0, NULL, &pidcount);
	if(pidcount == 0)
		return false;

	std::vector<cl_platform_id> pids(pidcount);
	clGetPlatformIDs(pidcount, &pids[0], NULL);

	char strInfo[100];
	if(platformID > 1)
		clDeviceType = CL_DEVICE_TYPE_GPU;
	for(cl_uint i=0; i<pidcount; ++i)
	{
		clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, 100, strInfo, NULL);
		if(strstr(strInfo, platformName[platformID]) != NULL)
		{
			if(clGetDeviceIDs(pids[i], clDeviceType, 1, &clDeviceID, NULL) != CL_SUCCESS)
				continue;
			cl_context_properties pops[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)pids[i], NULL };
			if((clContext = clCreateContextFromType(pops, clDeviceType, NULL, NULL, NULL)) == NULL)
				continue;
			cl_int errcode = CL_SUCCESS;
			clCmdQueue = clCreateCommandQueue(clContext, clDeviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &errcode);
			if(errcode == CL_INVALID_QUEUE_PROPERTIES)
				clCmdQueue = clCreateCommandQueue(clContext, clDeviceID, 0, &errcode);
			if(clCmdQueue == NULL)
				continue;
			clPlatformID = pids[i];
			clPgmBase = clCreateProgramWithSource(clContext, 1, &g_szKernel, NULL, NULL);
			errcode = clBuildProgram(clPgmBase, 1, &clDeviceID, "-cl-fast-relaxed-math", NULL, NULL);
			clSmooth1D = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(g_pfKernel1D), g_pfKernel1D, NULL);
			clSmooth2DX = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(g_pfKernel2D_X), g_pfKernel2D_X, NULL);
			clSmooth2DY = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(g_pfKernel2D_Y), g_pfKernel2D_Y, NULL);
			clFactorRegion = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(s_fFactorRegon64), s_fFactorRegon64, NULL);
			clLutArray1 = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bLUTArray1), bLUTArray1, NULL);
			clLutArray2 = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bLUTArray2), bLUTArray2, NULL);
			assert(clSmooth1D && clSmooth2DX && clSmooth2DY && clFactorRegion && clLutArray1 && clLutArray2);
			break;
		}
	}
	return clPlatformID != NULL;
}

bool release_platform()
{
	if(clCmdQueue != NULL)
		clReleaseCommandQueue(clCmdQueue), clCmdQueue = NULL;
	if(clContext != NULL)
		clReleaseContext(clContext), clContext = NULL;
	if(clPgmBase != NULL)
		clReleaseProgram(clPgmBase), clPgmBase = NULL;
	if(clSmooth1D != NULL)
		clReleaseMemObject(clSmooth1D), clSmooth1D = NULL;
	if(clSmooth2DX != NULL)
		clReleaseMemObject(clSmooth2DX), clSmooth2DX = NULL;
	if(clSmooth2DY != NULL)
		clReleaseMemObject(clSmooth2DY), clSmooth2DY = NULL;
	if(clFactorRegion != NULL)
		clReleaseMemObject(clFactorRegion), clFactorRegion = NULL;
	if(clLutArray1 != NULL)
		clReleaseMemObject(clLutArray1), clLutArray1 = NULL;
	if(clLutArray2 != NULL)
		clReleaseMemObject(clLutArray2), clLutArray2 = NULL;
	return true;
}

template<>
class std::auto_ptr<cl_mem>
{
public:
	explicit auto_ptr(cl_mem _Mem = NULL) : _Mymem(_Mem) {}
	operator cl_mem() const { return _Mymem; }
	~auto_ptr() { if(_Mymem != NULL) clReleaseMemObject(_Mymem); _Mymem = NULL; }

private:
	cl_mem	_Mymem;
};

bool clGetCannyEdge(unsigned char* edge,
					unsigned char const* image,
					int width,
					int height,
					double ratioLow,
					double ratioHigh)
{
	int count = width * height;
	cl_image_format imgFormat = { CL_LUMINANCE, CL_UNORM_INT8 };
	typedef std::auto_ptr<cl_mem> mem_type;
	mem_type clSrc(clCreateImage2D(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &imgFormat, width, height, width, (void*)image, NULL));
	mem_type clEdge(clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormat, width, height, 0, NULL, NULL));
	mem_type clTrans(clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormat, height, width, 0, NULL, NULL));
	mem_type clBackup(clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormat, height, width, 0, NULL, NULL));
	mem_type clTmp(clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormat, height, width, 0, NULL, NULL));
	imgFormat.image_channel_data_type = CL_FLOAT;
	mem_type clSmooth(clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormat, width, height, 0, NULL, NULL));
	mem_type clSmoothY(clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormat, width, height, 0, NULL, NULL));
	mem_type clGradY(clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormat, width, height, 0, NULL, NULL));
	mem_type clGradNorm(clCreateImage2D(clContext, CL_MEM_READ_WRITE, &imgFormat, width, height, 0, NULL, NULL));

	if(!clSrc || !clEdge || !clTrans || !clBackup || !clTmp || !clSmooth || !clSmoothY || !clGradY || !clGradNorm)
		return false;

	// Gaussian Smooth
	cl_int ret_code = CL_SUCCESS;
	cl_sampler sampler = clCreateSampler(clContext, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, NULL);
	cl_kernel kernelGaussianSmooth = clCreateKernel(clPgmBase, "GaussianSmoothX", &ret_code);
	clSetKernelArg(kernelGaussianSmooth, 0, sizeof(cl_mem), &clSmooth);
	clSetKernelArg(kernelGaussianSmooth, 1, sizeof(cl_mem), &clSrc);
	clSetKernelArg(kernelGaussianSmooth, 2, sizeof(cl_mem), &clSmooth1D);
	clSetKernelArg(kernelGaussianSmooth, 3, sizeof(int), &width);
	clSetKernelArg(kernelGaussianSmooth, 4, sizeof(int), &height);
	clSetKernelArg(kernelGaussianSmooth, 5, sizeof(cl_sampler), &sampler);
	size_t local_work_size[] = {16, 16};
	size_t global_work_size[] = {(width+15)&~15, (height+15)&~15};
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelGaussianSmooth, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelGaussianSmooth);

	cl_kernel kernelGaussianSmoothY = clCreateKernel(clPgmBase, "GaussianSmoothY", NULL);
	clSetKernelArg(kernelGaussianSmoothY, 0, sizeof(cl_mem), &clSmoothY);
	clSetKernelArg(kernelGaussianSmoothY, 1, sizeof(cl_mem), &clSmooth);
	clSetKernelArg(kernelGaussianSmoothY, 2, sizeof(cl_mem), &clSmooth1D);
	clSetKernelArg(kernelGaussianSmoothY, 3, sizeof(int), &width);
	clSetKernelArg(kernelGaussianSmoothY, 4, sizeof(int), &height);
	clSetKernelArg(kernelGaussianSmoothY, 5, sizeof(cl_sampler), &sampler);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelGaussianSmoothY, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelGaussianSmoothY);

	// Gaussian Filter
	cl_kernel kernelGaussianFilter = clCreateKernel(clPgmBase, "Gaussian2DReplicate", NULL);
	clSetKernelArg(kernelGaussianFilter, 0, sizeof(cl_mem), &clSmooth);
	clSetKernelArg(kernelGaussianFilter, 1, sizeof(cl_mem), &clSmoothY);
	clSetKernelArg(kernelGaussianFilter, 2, sizeof(cl_mem), &clSmooth2DX);
	clSetKernelArg(kernelGaussianFilter, 3, sizeof(int), &width);
	clSetKernelArg(kernelGaussianFilter, 4, sizeof(int), &height);
	clSetKernelArg(kernelGaussianFilter, 5, sizeof(cl_sampler), &sampler);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelGaussianFilter, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	cl_mem clGradX = clSmooth;

	clSetKernelArg(kernelGaussianFilter, 0, sizeof(cl_mem), &clGradY);
	clSetKernelArg(kernelGaussianFilter, 1, sizeof(cl_mem), &clSmoothY);
	clSetKernelArg(kernelGaussianFilter, 2, sizeof(cl_mem), &clSmooth2DY);
	clSetKernelArg(kernelGaussianFilter, 3, sizeof(int), &width);
	clSetKernelArg(kernelGaussianFilter, 4, sizeof(int), &height);
	clSetKernelArg(kernelGaussianFilter, 5, sizeof(cl_sampler), &sampler);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelGaussianFilter, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelGaussianFilter);

	cl_mem clGrad = clSmoothY;
	cl_kernel kernelGradMagnitude = clCreateKernel(clPgmBase, "GradMagnitude", NULL);
	clSetKernelArg(kernelGradMagnitude, 0, sizeof(cl_mem), &clGrad);
	clSetKernelArg(kernelGradMagnitude, 1, sizeof(cl_mem), &clGradX);
	clSetKernelArg(kernelGradMagnitude, 2, sizeof(cl_mem), &clGradY);
	clSetKernelArg(kernelGradMagnitude, 3, sizeof(int), &width);
	clSetKernelArg(kernelGradMagnitude, 4, sizeof(int), &height);
	clSetKernelArg(kernelGradMagnitude, 5, sizeof(cl_sampler), &sampler);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelGradMagnitude, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelGradMagnitude);

	size_t image_row_pitch = 0;
	size_t origin[] = { 0, 0, 0 }, region[] = { width, height, 1 };
	cl_mem maxGrad = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 64*sizeof(float), NULL, NULL);
	cl_kernel kernelGradMax = clCreateKernel(clPgmBase, "GradMax", NULL);
	clSetKernelArg(kernelGradMax, 0, sizeof(cl_mem), &maxGrad);
	clSetKernelArg(kernelGradMax, 1, sizeof(cl_mem), &clGrad);
	clSetKernelArg(kernelGradMax, 2, sizeof(cl_sampler), &sampler);
	clSetKernelArg(kernelGradMax, 3, sizeof(int), &width);
	clSetKernelArg(kernelGradMax, 4, sizeof(int), &height);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelGradMax, 2, NULL, local_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelGradMax);

	cl_kernel kernelGradInv = clCreateKernel(clPgmBase, "GradNorm", NULL);
	clSetKernelArg(kernelGradInv, 0, sizeof(cl_mem), &clGradNorm);
	clSetKernelArg(kernelGradInv, 1, sizeof(cl_mem), &clGrad);
	clSetKernelArg(kernelGradInv, 2, sizeof(int), &width);
	clSetKernelArg(kernelGradInv, 3, sizeof(int), &height);
	clSetKernelArg(kernelGradInv, 4, sizeof(cl_sampler), &sampler);
	clSetKernelArg(kernelGradInv, 5, sizeof(cl_mem), &maxGrad);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelGradInv, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelGradInv);

	cl_mem hist = maxGrad;
	cl_kernel kernelHist = clCreateKernel(clPgmBase, "GradHist", NULL);
	ret_code = clEnqueueFillBuffer(clCmdQueue, hist, origin, sizeof(size_t), 0, 64*sizeof(int), 0, NULL, NULL);
	clSetKernelArg(kernelHist, 0, sizeof(cl_mem), &hist);
	clSetKernelArg(kernelHist, 1, sizeof(cl_mem), &clGradNorm);
	clSetKernelArg(kernelHist, 2, sizeof(cl_mem), &clFactorRegion);
	clSetKernelArg(kernelHist, 3, sizeof(int), &width);
	clSetKernelArg(kernelHist, 4, sizeof(int), &height);
	clSetKernelArg(kernelHist, 5, sizeof(cl_sampler), &sampler);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelHist, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelHist);

	cl_kernel kernalTH = clCreateKernel(clPgmBase, "CalcThreshold", NULL);
	clSetKernelArg(kernalTH, 0, sizeof(cl_mem), &hist);
	float fHighR = (float)ratioHigh;
	float fLowR = (float)ratioLow;
	clSetKernelArg(kernalTH, 1, sizeof(float), &fHighR);
	clSetKernelArg(kernalTH, 2, sizeof(float), &fLowR);
	clSetKernelArg(kernalTH, 3, sizeof(int), &count);
	size_t work_size = 64;
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernalTH, 1, NULL, &work_size, &work_size, 0, NULL, NULL);
	clReleaseKernel(kernalTH);

	// Non-maximum suppress
	cl_kernel kernelNonMaxSuppress = clCreateKernel(clPgmBase, "NonMaxSuppress", NULL);
	clSetKernelArg(kernelNonMaxSuppress, 0, sizeof(cl_mem), &clEdge);
	clSetKernelArg(kernelNonMaxSuppress, 1, sizeof(cl_mem), &clGradX);
	clSetKernelArg(kernelNonMaxSuppress, 2, sizeof(cl_mem), &clGradY);
	clSetKernelArg(kernelNonMaxSuppress, 3, sizeof(cl_mem), &clGradNorm);
	clSetKernelArg(kernelNonMaxSuppress, 4, sizeof(int), &width);
	clSetKernelArg(kernelNonMaxSuppress, 5, sizeof(int), &height);
	clSetKernelArg(kernelNonMaxSuppress, 6, sizeof(cl_sampler), &sampler);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelNonMaxSuppress, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelNonMaxSuppress);

	// Hysteresis
	cl_kernel kernelHys = clCreateKernel(clPgmBase, "Hysteresis", NULL);
	clSetKernelArg(kernelHys, 0, sizeof(cl_mem), &clSrc);
	clSetKernelArg(kernelHys, 1, sizeof(cl_mem), &clEdge);
	clSetKernelArg(kernelHys, 2, sizeof(cl_mem), &clGradNorm);
	clSetKernelArg(kernelHys, 3, sizeof(cl_sampler), &sampler);
	clSetKernelArg(kernelHys, 4, sizeof(int), &width);
	clSetKernelArg(kernelHys, 5, sizeof(int), &height);
	clSetKernelArg(kernelHys, 6, sizeof(cl_mem), &hist);
	ret_code = clEnqueueNDRangeKernel(clCmdQueue, kernelHys, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelHys);

	ret_code = clEnqueueReadImage(clCmdQueue, clSrc, CL_TRUE, origin, region, 0, 0, edge, 0, NULL, NULL);
	std::vector<unsigned char*> vPtr;
	vPtr.reserve(count);
	unsigned char* _Tmp = edge;
	for(int i=0; i<height; ++i, _Tmp += width)
	{
		for(int j=0; j<width; ++j)
		{
			if(_Tmp[j] == 255)
			{
				vPtr.push_back(&_Tmp[j]);
				while(!vPtr.empty())
				{
					unsigned char* cur = vPtr.back();
					vPtr.pop_back();
					int len = (int)(cur - edge);
					int x = len % width;
					*cur = 255;
					if(len >= width)
					{
						if(x > 0 && cur[-width-1] == 128)
							vPtr.push_back(&cur[-width-1]);
						if(cur[-width] == 128)
							vPtr.push_back(&cur[-width]);
						if(x < width-1 && cur[-width+1] == 128)
							vPtr.push_back(&cur[-width+1]);
					}
					if(x > 0 && cur[-1] == 128)
						vPtr.push_back(&cur[-1]);
					if(x < width-1 && cur[1] == 128)
						vPtr.push_back(&cur[1]);
					if(len <= count - width)
					{
						if(x > 0 && cur[width-1] == 128)
							vPtr.push_back(&cur[width-1]);
						if(cur[width] == 128)
							vPtr.push_back(&cur[width]);
						if(x < width-1 && cur[width+1] == 128)
							vPtr.push_back(&cur[width+1]);
					}
				}
			}
		}
	}

	// Final
	clEnqueueWriteImage(clCmdQueue, clSrc, CL_TRUE, origin, region, 0, 0, edge, 0, NULL, NULL);
	cl_kernel kernelTransSet = clCreateKernel(clPgmBase, "TransposeSet", NULL);
	clSetKernelArg(kernelTransSet, 0, sizeof(cl_mem), &clTrans);
	clSetKernelArg(kernelTransSet, 1, sizeof(cl_mem), &clSrc);
	clSetKernelArg(kernelTransSet, 2, sizeof(cl_sampler), &sampler);
	clSetKernelArg(kernelTransSet, 3, sizeof(int), &height);
	clSetKernelArg(kernelTransSet, 4, sizeof(int), &width);
	std::swap(global_work_size[0], global_work_size[1]);
	cl_int errcode = clEnqueueNDRangeKernel(clCmdQueue, kernelTransSet, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelTransSet);

	bool done = false;
	std::swap(region[0], region[1]);
	cl_kernel kernelApplyLut = clCreateKernel(clPgmBase, "ApplyLut", NULL);
	int iterates = 1;
	bool equalC = true;
	int iterNum = 1;
	while(!done)
	{
#ifdef THINNER_MORE
		errcode = clEnqueueCopyImage(clCmdQueue, clTrans, clBackup, origin, origin, region, 0, NULL, NULL);
#endif
		clSetKernelArg(kernelApplyLut, 0, sizeof(cl_mem), &clTmp);
		clSetKernelArg(kernelApplyLut, 1, sizeof(cl_mem), &clTrans);
		clSetKernelArg(kernelApplyLut, 2, sizeof(cl_mem), &clLutArray1);
		clSetKernelArg(kernelApplyLut, 3, sizeof(cl_sampler), &sampler);
		clSetKernelArg(kernelApplyLut, 4, sizeof(int), &height);
		clSetKernelArg(kernelApplyLut, 5, sizeof(int), &width);
		errcode = clEnqueueNDRangeKernel(clCmdQueue, kernelApplyLut, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

		clSetKernelArg(kernelApplyLut, 0, sizeof(cl_mem), &clTrans);
		clSetKernelArg(kernelApplyLut, 1, sizeof(cl_mem), &clTmp);
		clSetKernelArg(kernelApplyLut, 2, sizeof(cl_mem), &clLutArray2);
		clSetKernelArg(kernelApplyLut, 3, sizeof(cl_sampler), &sampler);
		clSetKernelArg(kernelApplyLut, 4, sizeof(int), &height);
		clSetKernelArg(kernelApplyLut, 5, sizeof(int), &width);
		errcode = clEnqueueNDRangeKernel(clCmdQueue, kernelApplyLut, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

#ifdef THINNER_MORE
		size_t row_pitch0 = 0;
		void* ptr0 = clEnqueueMapImage(clCmdQueue, clBackup, CL_TRUE, CL_MAP_READ, origin, region, &row_pitch0, NULL, 0, NULL, NULL, NULL);
		size_t row_pitch1 = 0;
		void* ptr1 = clEnqueueMapImage(clCmdQueue, clTrans, CL_TRUE, CL_MAP_READ, origin, region, &row_pitch1, NULL, 0, NULL, NULL, NULL);
		if(row_pitch0 == row_pitch1 && row_pitch0 == height)
			equalC = memcmp(ptr0, ptr1, count) == 0;
		else
		{
			void* tmp0 = ptr0;
			void* tmp1 = ptr1;
			for(int i=0; i<width; ++i)
			{
				if(!(equalC = memcmp(tmp0, tmp1, height) != 0))
					break;
				tmp0 = (char*)tmp0 + row_pitch0, tmp1 = (char*)tmp1 + row_pitch1;
			}
		}
		clEnqueueUnmapMemObject(clCmdQueue, clBackup, ptr0, 0, NULL, NULL);
		clEnqueueUnmapMemObject(clCmdQueue, clTrans, ptr1, 0, NULL, NULL);
#endif

		done = (iterates >= iterNum) | equalC;
		iterates++;
	}

	cl_kernel kernelTrans = clCreateKernel(clPgmBase, "Transpose", NULL);
	clSetKernelArg(kernelTrans, 0, sizeof(cl_mem), &clSrc);
	clSetKernelArg(kernelTrans, 1, sizeof(cl_mem), &clTrans);
	clSetKernelArg(kernelTrans, 2, sizeof(cl_sampler), &sampler);
	clSetKernelArg(kernelTrans, 3, sizeof(int), &width);
	clSetKernelArg(kernelTrans, 4, sizeof(int), &height);
	std::swap(global_work_size[0], global_work_size[1]);
	errcode = clEnqueueNDRangeKernel(clCmdQueue, kernelTrans, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	clReleaseKernel(kernelTrans);
	clReleaseKernel(kernelApplyLut);
	clReleaseSampler(sampler);

	std::swap(region[0], region[1]);
	errcode = clEnqueueReadImage(clCmdQueue, clSrc, CL_TRUE, origin, region, 0, 0, edge, 0, NULL, NULL);

	// Done
	return true;
}
