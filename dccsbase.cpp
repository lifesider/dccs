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

#if defined(_WIN32) || defined(_WINDOWS)
#define decl_align(type, n, var)	__declspec(align(n)) ##type var
#else
#define decl_align(type, n, var)	type var __attribute__((aligned(n)))
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
        movaps      xmm5, xmm1;
        movaps      xmm6, xmm2;
        unpcklps    xmm1, xmm3;
        unpcklps    xmm2, xmm4;
        unpckhps    xmm5, xmm3;
        unpckhps    xmm6, xmm4;
        packssdw    xmm1, xmm5;
        packssdw    xmm2, xmm6;
        packuswb    xmm1, xmm2;
        movsd       [ebx_ptr], xmm1;
        movhps      [ecx_ptr], xmm2;
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
        jz          loop_1;
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