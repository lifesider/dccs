//
//  dccsbase.cpp
//  DccsBase
//
//  Created by 赖守波 on 15/9/17.
//  Copyright (c) 2015年 Sobey. All rights reserved.
//

#include "dccsbase.h"

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
    loop_1:
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
		pand		xmm4, bit0Mask;
		pand		xmm5, bit0Mask;
		paddb		xmm6, xmm4;
		paddb		xmm6, xmm5;
		add			esi_ptr, ecx_ptr;
		dec			ebx_ptr;
		jnz			loop_255_2;
		movdqu		xmm4, [esi_ptr];
		pcmpeqb		xmm4, xmm7;
		pand		xmm4, bit0Mask;
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
		pand		xmm4, bit0Mask;
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

decl_align(float, 16, fInv255[4]) = {1.f/255, 1.f/255, 1.f/255, 1.f/255};
void ucharnorm2double_sse2(OUT double *des, IN unsigned char const*src, IN size_t count)
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
		movhlps		xmm4, xmm2;
		movhlps		xmm5, xmm3;
		cvtps2pd	xmm2, xmm2;
		cvtps2pd	xmm4, xmm4;
		cvtps2pd	xmm3, xmm3;
		cvtps2pd	xmm5, xmm5;
		movups		[edi_ptr], xmm2;
		movups		[edi_ptr + 0x10], xmm4;
		movups		[edi_ptr + 0x20], xmm3;
		movups		[edi_ptr + 0x30], xmm5;
		add			esi_ptr, 8;
		add			edi_ptr, 0x40;
		sub			ecx_ptr, 8;
		jge			loop_8;
loop_1_pre:
		add			ecx_ptr, 8;
		jz			loop_end;
		cvtps2pd	xmm0, xmm0;
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
