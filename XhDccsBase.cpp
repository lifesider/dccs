#include "StdAfx.h"
#include "XhDccsBase.h"
#include <assert.h>

#ifdef SSE_OPTIMIZE
#include <intrin.h>
#include "dccsbase.h"
#endif

XhDccsBase::XhDccsBase(void)
{
}


XhDccsBase::~XhDccsBase(void)
{
}

// 彩色图转灰度图，nBpp必须为24，nType = 0 时采用移位方式近似，nType = 1时采用浮点计算
void XhDccsBase::ImgRGB2Gray(const BYTE* pbyImg, const SIZE& szImg, 
                             BYTE* pbyGray, int nBpp, int nType)
{
#ifdef SSE_OPTIMIZE
    if(nType == 0)
        rgb2gray_s_sse2(pbyGray, pbyImg, szImg.cx*szImg.cy);
    else
        rgb2gray_f_sse2(pbyGray, pbyImg, szImg.cx*szImg.cy);
    return;
#endif
    assert(pbyImg);
    assert(nBpp == 24);
    assert(pbyGray);    

    int           nBytes   = (nBpp >> 3);
    const BYTE*   pi       = pbyImg;
    BYTE*         po       = pbyGray;
    int           nBufLen = szImg.cx * szImg.cy;

    if (nType == 0)
    {
        for (int i = 0; i < nBufLen; i++, po++)
        {
            *po  = (pi[0] * B_COEF + pi[1] * G_COEF + pi[2] * R_COEF ) >> 10;      // 右移十位等于除以1024，因此RGB系数也比nType=1时，乘以了1.024倍
            pi	+= nBytes;
        }
    }
    else if (nType == 1)
    {
        for (int i = 0; i < nBufLen; i++, po++)
        {
            double dbVal = (0.114020904255103 * pi[0] + 0.587043074451121 * pi[1] + 0.298936021293775 * pi[2]);
            *po			 = BYTE(dbVal + 0.5);
            pi			+= nBytes;
        }
    }

}

// 分离BGR图像中GR通道
void XhDccsBase::SeparateGR(const BYTE* pbyBGR, const SIZE& szImg, 
                            BYTE* pbyGreen, BYTE* pbyRed)
{
#ifdef SSE_OPTIMIZE
    get_gr_channel_sse2(pbyGreen, pbyRed, pbyBGR, szImg.cx*szImg.cy);
    return;
#endif
    assert(pbyBGR);
    assert(pbyGreen && pbyRed);

    int         nPxlNum = szImg.cx * szImg.cy;
    const BYTE* pi      = pbyBGR + 1;

    for (int i = 0; i < nPxlNum; i++)
    {
        pbyGreen[i] = *pi;
        pbyRed[i]   = pi[1];

        pi          += 3;
    }
}


// 灰度图转二值图
void XhDccsBase::ImgGray2Binary(const BYTE* pbyGray, const SIZE& szGray, BYTE byThresh, BYTE* pbyBinary)
{
#ifdef SSE_OPTIMIZE
    gray2binary(pbyBinary, pbyGray, byThresh, szGray.cx*szGray.cy);
    return;
#endif
    assert(pbyGray);
    assert(pbyBinary);

    int nLimit = szGray.cx * szGray.cy;

    for (int i = 0; i < nLimit; i++)
    {
        if (*pbyGray >= byThresh)
        {
            *pbyBinary = 255;
        }
        else
        {
            *pbyBinary = 0;
        }

        pbyGray++;
        pbyBinary++;
    }
}

// 提取指定区域图像
void XhDccsBase::GetSelectedArea(const BYTE* pbySrc, const SIZE& szSrc, int nBpp, 
                                 const RECT& rtDst, BYTE* pbyDst)
{
    assert(pbySrc);
    assert(pbyDst);
    assert(rtDst.right - rtDst.left + 1 > 0);
    assert(rtDst.bottom - rtDst.top + 1 > 0);

    const BYTE* pi        = pbySrc + (rtDst.top * szSrc.cx + rtDst.left) * (nBpp >> 3);
    BYTE*       po        = pbyDst;
    int         nCopyBuff = (rtDst.right - rtDst.left + 1) * (nBpp >> 3);
    int         nSrcBuff  = szSrc.cx * (nBpp >> 3);

    for (int j = rtDst.top; j <= rtDst.bottom; j++)     // 逐行拷贝限定区域内的像素值
    {
        memcpy(po, pi, nCopyBuff);
        pi += nSrcBuff;
        po += nCopyBuff;
    }
}

// 双线性插值缩放图像
#ifndef NSHIFT
#define NSHIFT  (15)
#endif

void XhDccsBase::RspImgLinear(const BYTE* pbySrc, const SIZE& szSrc, 
                              int nBpp, const SIZE& szDst, BYTE* pbyDst)
{
    int   nSrcW  = szSrc.cx;
    int   nSrcH  = szSrc.cy;
    int   nDstW  = szDst.cx;
    int   nDstH  = szDst.cy;

    assert(pbySrc);
    assert(pbyDst);    
    assert(nBpp == 8 || nBpp == 24);
    assert(nDstW > 0 && nDstH > 0);
    assert(nSrcW > 0 && nSrcH > 0 );

    int   w, h, k;
    int   tmp;
    int   ix, iy;
    int   sum;
    int   kx, ky;
    int   fx, fy, fxy;
    const BYTE* p_src   = pbySrc;
    BYTE*       p_dst   = pbyDst;
    int         nOffset = nBpp >> 3;

    kx = ((nSrcW - 1) << NSHIFT) / nDstW;
    ky = ((nSrcH - 1) << NSHIFT) / nDstH;

    if (nSrcW == 1 && nSrcH == 1)   // 传入图像只有1个点
    {
        int nPixelNum = nDstH * nDstW;

        for (int i = 0; i < nPixelNum; i++)
        {
            CopyMemory(p_dst, p_src, nOffset);
            p_dst += nOffset;
        }
    }
    else if (nSrcW == 1)            // 传入图像只有一列(行数大于1)
    {
        for (h = 0; h < nDstH; h++)
        {
            tmp = h * ky;
            iy  = tmp >> NSHIFT;
            fy  = tmp - (iy << NSHIFT);

            for (w = 0; w < nDstW; w++)
            {
                for (k = 0; k < nOffset; k++)
                {
                    p_src = pbySrc + nSrcW * iy * nOffset + k;

                    sum = ((1 << NSHIFT) - fy) * p_src[0 * nOffset]
                    + fy * p_src[nSrcW * nOffset];

                    p_dst[w * nOffset + k] = (BYTE)(sum >> NSHIFT);
                }
            }

            p_dst += nDstW * nOffset;
        }
    }
    else if (nSrcH == 1)  // 传入图像只有一行(列数大于1)
    {
        for (h = 0; h < nDstH; h++)
        {
            for (w = 0; w < nDstW; w++)
            {
                tmp = w * kx;
                ix  = tmp >> NSHIFT;
                fx  = tmp - (ix << NSHIFT);

                for (k = 0; k < nOffset; k++)
                {
                    p_src = pbySrc + ix * nOffset + k;

                    sum = ((1 << NSHIFT) - fx) * p_src[0 * nOffset]
                    + fx * p_src[1 * nOffset];

                    p_dst[w * nOffset + k] = (BYTE)(sum >> NSHIFT);
                }
            }

            p_dst += nDstW * nOffset;
        }
    }
    else
    {
        for (h = 0; h < nDstH; h++)
        {
            tmp = h * ky;
            iy = tmp >> NSHIFT;
            fy = tmp - (iy << NSHIFT);

#ifdef SSE_OPTIMIZE
			typedef void (*_Func_Ptr)(unsigned char*, unsigned char const*, intptr_t, int, int, int);
			_Func_Ptr resample_linear_line = nBpp == 8 ? resample_linear_8_line : resample_linear_24_line;
			resample_linear_line(p_dst, pbySrc + nSrcW * iy * nOffset, nSrcW * nOffset, fy, kx, nDstW);
#else
            for (w = 0; w < nDstW; w++)
            {
                tmp = w * kx;
                ix  = tmp >> NSHIFT;
                fx  = tmp - (ix << NSHIFT);
                fxy = (fx * fy) >> NSHIFT;

                for (int k = 0; k < nOffset; k++)
                {
                    p_src = pbySrc + nSrcW * iy * nOffset + ix * nOffset + k;

                    sum = ((1 << NSHIFT) - fx - fy + fxy) * p_src[0 * nOffset]
                    + (fx - fxy)                      * p_src[1 * nOffset]
                    + (fy - fxy) * p_src[nSrcW * nOffset + 0 * nOffset]
                    + (fxy)      * p_src[nSrcW * nOffset + 1 * nOffset];

                    p_dst[w * nOffset + k] = (BYTE)(sum >> NSHIFT);

                }
            }
#endif
            p_dst += nDstW * nOffset;
        }
    }
}

// 最近邻法缩放图像
void XhDccsBase::RspImgNN(const BYTE* pbySrc, const SIZE& szSrc, int nBpp,
                          const SIZE& szDst, BYTE* pbyDst)
{
    assert(pbySrc);
    assert(pbyDst);    
    assert(nBpp == 8 || nBpp == 24);
    assert(szDst.cx > 0 && szDst.cy > 0);

    int         nW         = szDst.cx;
    int         nH         = szDst.cy;
    double      dbRX       = (double)szSrc.cx / nW;
    double      dbRY       = (double)szSrc.cy / nH;
    double      dbOffsetY  = nH - 0.5;
    int         nRowBufLen = szSrc.cx * (nBpp >> 3);

    const BYTE* pi         = pbySrc + (szSrc.cy - 1) * nRowBufLen;
    const BYTE* pRow       = pi;
    const BYTE* pCol       = pi;
    BYTE*       po         = pbyDst;			

    int         i, j;

    if (nBpp == 24)
    {
        for (j = 0; j < nH; j++)
        {
            pRow       = pi - (int)(dbRY * (dbOffsetY - j)) * nRowBufLen;   // 最近邻行
#ifdef SSE_OPTIMIZE
			__m128i xmm0 = _mm_set_epi32(3, 2, 1, 0);
			static __m128i xmm1 = _mm_set_epi32(4, 4, 4, 4);
			static __m128 xmm2 = _mm_set_ps(0.5f, 0.5f, 0.5f, 0.5f);
			__m128 xmm7 = _mm_set_ps((float)dbRX, (float)dbRX, (float)dbRX, (float)dbRX);
			for(i=nW-4; i>=0; i-=4)
			{
				__m128i xmm3 = _mm_cvttps_epi32(_mm_mul_ps(_mm_add_ps(_mm_cvtepi32_ps(xmm0), xmm2), xmm7));
				xmm3 = _mm_add_epi32(xmm3, _mm_add_epi32(xmm3, xmm3));
				int offs = _mm_cvtsi128_si32(xmm3);
				xmm3 = _mm_shuffle_epi32(xmm3, 0x39);
				*(unsigned short*)po = *(unsigned short*)(pRow + offs);
				*(po+2) = *(pRow + offs + 2);
				po += 3;
				offs = _mm_cvtsi128_si32(xmm3);
				xmm3 = _mm_shuffle_epi32(xmm3, 9);
				*(unsigned short*)po = *(unsigned short*)(pRow + offs);
				*(po+2) = *(pRow + offs + 2);
				po += 3;
				offs = _mm_cvtsi128_si32(xmm3);
				xmm3 = _mm_shuffle_epi32(xmm3, 1);
				*(unsigned short*)po = *(unsigned short*)(pRow + offs);
				*(po+2) = *(pRow + offs + 2);
				po += 3;
				offs = _mm_cvtsi128_si32(xmm3);
				*(unsigned short*)po = *(unsigned short*)(pRow + offs);
				*(po+2) = *(pRow + offs + 2);
				po += 3;
				xmm0 = _mm_add_epi32(xmm0, xmm1);
			}
			if((i = (nW & 3)) > 0)
			{
				while (i > 0)
				{
					pCol = pRow + (int)(dbRX * (xmm0.m128i_i32[0] + 0.5)) * 3;
					*(short*)po = *(short*)pCol;
					*(po+2) = *(pCol+2);
					po+=3;
				}
			}
#else
            for (i = 0; i < nW; i++)
            {
                pCol   = pRow + (int)(dbRX * (i + 0.5)) * 3;     // 最近邻行与最近邻列交叉处（找到最近邻点）
                CopyMemory(po, pCol, 3);                         // 复制最近邻点的像素值
                po     += 3;
            }
#endif
        }
    }
    else
    {
        for (j = 0; j < nH; j++)
        {
            pRow       = pi - (int)(dbRY * (dbOffsetY - j)) * nRowBufLen;   // 最近邻行

#ifdef SSE_OPTIMIZE
			__m128i xmm0 = _mm_set_epi32(3, 2, 1, 0);
			static __m128i xmm1 = _mm_set_epi32(4, 4, 4, 4);
			static __m128 xmm2 = _mm_set_ps(0.5f, 0.5f, 0.5f, 0.5f);
			__m128 xmm7 = _mm_set_ps((float)dbRX, (float)dbRX, (float)dbRX, (float)dbRX);
			for(i=nW-4; i>=0; i-=4)
			{
				__m128i xmm3 = _mm_cvttps_epi32(_mm_mul_ps(_mm_add_ps(_mm_cvtepi32_ps(xmm0), xmm2), xmm7));
				int offs = _mm_cvtsi128_si32(xmm3);
				xmm3 = _mm_shuffle_epi32(xmm3, 0x39);
				*po++ = *(pRow + offs);
				offs = _mm_cvtsi128_si32(xmm3);
				xmm3 = _mm_shuffle_epi32(xmm3, 9);
				*po++ = *(pRow + offs);
				offs = _mm_cvtsi128_si32(xmm3);
				xmm3 = _mm_shuffle_epi32(xmm3, 1);
				*po++ = *(pRow + offs );
				offs = _mm_cvtsi128_si32(xmm3);
				*po++ = *(pRow + offs);
				xmm0 = _mm_add_epi32(xmm0, xmm1);
			}
			if((i = (nW & 3)) > 0)
			{
				while (i > 0)
				{
					pCol = pRow + (int)(dbRX * (xmm0.m128i_i32[0] + 0.5));
					*po++ = *pCol;
				}
			}
#else
           for (i = 0; i < nW; i++, po++)
            {
                pCol   = pRow + (int)(dbRX * (i + 0.5));    // 最近邻行与最近邻列交叉处（找到最近邻点）
                *po    = *pCol;                             // 复制最近邻点的像素值
            }
#endif
        }
    }
}

// 任意3*3算子滤波, pdbKernel是3*3算子（实际数据为一行九个）
void XhDccsBase::ImgFilter(const BYTE* pbySrc, const SIZE& szSrc, int nBpp, 
                           BYTE* pbyDst, const double* pdbKernel)
{
    assert(pbySrc);
    assert(pbyDst);    
    assert(pdbKernel);
    assert(nBpp == 8 || nBpp == 24);
	
#ifdef SSE_OPTIMIZE
	int channels = nBpp >> 3;
	if (szSrc.cy == 1)
	{
		double coef = pdbKernel[3] + pdbKernel[4] + pdbKernel[5];
		for (int i = channels; i < szSrc.cx*channels; ++i)
			*pbyDst++ = (BYTE)(pbySrc[i] * coef + 0.5);
		for (int i = 0; i < channels; ++i)
			*pbyDst++ = 0;
	}
	else if (szSrc.cy == 2)
	{
		short coef[4];
		coef[0] = (short)((pdbKernel[0] + pdbKernel[1] + pdbKernel[2]) * 16384 + 0.5);
		coef[1] = (short)((pdbKernel[3] + pdbKernel[4] + pdbKernel[5]) * 16384 + 0.5);
		coef[2] = (short)((pdbKernel[6] + pdbKernel[7] + pdbKernel[8]) * 16384 + 0.5);
		int stride = (szSrc.cx - 1) * channels;
		imfilter_3x3_line2(pbyDst, pbySrc + channels, szSrc.cx*channels, coef, stride);
		for (int i = 0; i < channels; ++i)
			pbyDst[stride + i] = 0;
		pbyDst += stride + channels;
		short __coef[4] = { coef[1], coef[2] };
		imfilter_3x3_line2(pbyDst, pbySrc + channels, szSrc.cx*channels, __coef, stride);
		for (int i = 0; i < channels; ++i)
			pbyDst[stride + i] = 0;
	}
	else
	{
		short coef[4];
		coef[0] = (short)((pdbKernel[0] + pdbKernel[1] + pdbKernel[2]) * 16384 + 0.5);
		coef[1] = (short)((pdbKernel[3] + pdbKernel[4] + pdbKernel[5]) * 16384 + 0.5);
		coef[2] = (short)((pdbKernel[6] + pdbKernel[7] + pdbKernel[8]) * 16384 + 0.5);
		int stride = (szSrc.cx - 1) * channels;
		imfilter_3x3_line2(pbyDst, pbySrc+ channels, szSrc.cx*channels, coef, stride);
		for (int i = 0; i < channels; ++i)
			pbyDst[stride + i] = 0;
		pbyDst += stride + channels;
		for (int i = 1; i < szSrc.cy-1; ++i)
		{
			imfilter_3x3_line3(pbyDst, pbySrc + channels, szSrc.cx*channels, coef, stride);
			for (int j = 0; j < channels; ++j)
				pbyDst[stride + j] = 0;
			pbyDst += stride + channels;
			pbySrc += stride + channels;
		}
		short __coef[4] = { coef[1], coef[2] };
		imfilter_3x3_line2(pbyDst, pbySrc + channels, szSrc.cx*channels, __coef, stride);
		for (int i = 0; i < channels; ++i)
			pbyDst[stride + i] = 0;
	}
	return;
#endif

    // 初始化卷积结果图像
    SIZE    szPadded  = {szSrc.cx + 2, szSrc.cy + 2};
    int     nChannels = nBpp >> 3;
    int     nBufLen  = szPadded.cy * szPadded.cx * nChannels;
    double* pdbPadded = new double[nBufLen];
    ZeroMemory(pdbPadded, nBufLen * sizeof(*pdbPadded)); 

    // 临时变量
    const BYTE*  pi         = pbySrc;
    double*      pt         = pdbPadded;    
    int          nTowColOff = (nChannels << 1);
    int          nOneRowOff   = szPadded.cx * nChannels; 	
    int          nTowRowOff   = (nOneRowOff << 1);
    int          i, j, k;
    
    // 初始化偏移（邻域内3*3像素对应相乘的点，相对于左上顶点的位置）
    int* pnSubOff = new int[9 * nChannels];    

    for (k = 0; k < 3; k++)   // 三列
    {
        for (i = 0; i < nChannels; i++)
        {
            pnSubOff[     k  * nChannels + i] = i;                                           // 第一行
            pnSubOff[(3 + k) * nChannels + i] = pnSubOff[k * nChannels + i] + nOneRowOff;    // 第二行
            pnSubOff[(6 + k) * nChannels + i] = pnSubOff[k * nChannels + i] + nTowRowOff;    // 第三行
        }
    }

    // 计算每个点的卷积贡献
    int*     pf         = pnSubOff;

    if (nBpp == 24)
    {
        for (j = 0; j < szSrc.cy; j++)
        {
            for (i = 0; i < szSrc.cx; i++)
            {
                pf  = pnSubOff;

                for (k = 0; k < 9; k++)
                {
                    pt[*pf++] += pi[0] * pdbKernel[k];
                    pt[*pf++] += pi[1] * pdbKernel[k];
                    pt[*pf++] += pi[2] * pdbKernel[k];
                }

                pi += nChannels;
                pt += nChannels;
            }

            pt += nTowColOff;
        }

        // 取出内部区域
        pt	          = pdbPadded + nOneRowOff + nChannels;
        BYTE* po      = pbyDst;

        for (j = 0; j < szSrc.cy; j++)
        {
            for (i = 0; i < szSrc.cx; i++)
            {
                *po++  = BYTE((*pt++) + 0.5);
                *po++  = BYTE((*pt++) + 0.5);
                *po++  = BYTE((*pt++) + 0.5);
            }

            pt += nTowColOff;
        }
    }
    else  // 8位图像
    {
        for (j = 0; j < szSrc.cy; j++)
        {
            for (i = 0; i < szSrc.cx; i++, pi++, pt++)
            {
                pf  = pnSubOff;

                for (k = 0; k < 9; k++)
                {
                    pt[*pf++] += (*pi * pdbKernel[k]);
                }
            }

            pt += nTowColOff;
        }

        // 取出内部区域
        pt	          = pdbPadded + nOneRowOff + nChannels;
        BYTE* po      = pbyDst;

        for (j = 0; j < szSrc.cy; j++)
        {
            for (i = 0; i < szSrc.cx; i++)
            {
                *po++  = BYTE((*pt++) + 0.5);
            }

            pt += nTowColOff;
        }
    }

    MEMO_FREE_AND_NULL_N(pnSubOff);
    MEMO_FREE_AND_NULL_N(pdbPadded);
}

// 连通域标记-区域标注函数，返回标注得到的区域的个数，
//    byIgnore - 不需要标注的像素值，若全部像素值都需要被标注则令为-1（也是默认值）
//    nType   - 区域标注的邻域类型，必须为4邻域或者8邻域，默认为8.
int XhDccsBase::ImgRegionLabel(const BYTE* pbyImg, int* pnOut, 
                               const SIZE& szImg, BYTE byIgnore, 
                               int nType)
{
    assert(pbyImg);
    assert(pnOut);    
    assert(nType);

    int  nNum = -1;

    if (nType == 8)
    {
        nNum = ImgRegionLabel_8(pbyImg, pnOut, szImg, byIgnore);
    }

    if (nType == 4)
    {
        nNum = ImgRegionLabel_4(pbyImg, pnOut, szImg, byIgnore);	
    }

    return      nNum;
}


int XhDccsBase::ImgRegionLabel_8(const BYTE* pbyImg, int* pnOut, 
    const SIZE& szImg, BYTE byIgnore)
{
    int    nBufLen  = szImg.cx * szImg.cy;
    BYTE*  pbyFlag  = new BYTE  [nBufLen];
    BYTE** ppbTmp	= new BYTE* [nBufLen * 8];
    memset(pbyFlag, 0, sizeof(*pbyFlag) * nBufLen);// clear all flag 
    memset(pnOut, 0, sizeof(*pnOut) * nBufLen);

    // 用于边界判断
    BYTE*  pbyFlagRowStart = pbyFlag + szImg.cx;
    BYTE*  pbyFlagRowEnd   = pbyFlag + szImg.cx * (szImg.cy - 1) - 1;
    int	   nLastCol		  = szImg.cx - 1;

    // 点遍历
    const BYTE*  pi = pbyImg; 
    BYTE*  pf       = pbyFlag; 
    int    nNum     = 1; 

    for (int i = 0; i < nBufLen; i++, pi++, pf++)
    {
        if (*pi != byIgnore && (!*pf))
        {
            FillNeighbor_8(pbyImg, szImg, nLastCol, pbyFlag, pbyFlagRowStart, pbyFlagRowEnd,
                ppbTmp, pi, nNum++, pnOut);
        }
    }

    MEMO_FREE_AND_NULL_N(pbyFlag);
    MEMO_FREE_AND_NULL_N(ppbTmp);

    return           nNum; 
}

int XhDccsBase::ImgRegionLabel_4(const BYTE* pbyImg, int* pnOut, 
                                 const SIZE& szImg, BYTE byIgnore)
{
    int    nBufLen  = szImg.cx * szImg.cy;
    BYTE*  pbyFlag  = new BYTE [nBufLen];
    BYTE** ppbTmp	= new BYTE* [nBufLen];
    memset(pbyFlag, 0, sizeof(*pbyFlag) * nBufLen);// clear all flag 
    memset(pnOut, 0, sizeof(*pnOut) * nBufLen);

    // 用于边界判断
    BYTE*  pbyFlagRowStart = pbyFlag + szImg.cx;
    BYTE*  pbyFlagRowEnd   = pbyFlag + szImg.cx * (szImg.cy - 1) - 1;
    int	   nLastCol		  = szImg.cx - 1;

    // 点遍历
    const BYTE*  pi = pbyImg; 
    BYTE*  pf       = pbyFlag; 
    int    nNum     = 1; 

    for (int i = 0; i < nBufLen; i++, pi++, pf++)
    {
        if (*pi != byIgnore && (!*pf))
        {
            FillNeighbor_4(pbyImg, szImg, nLastCol, pbyFlag, pbyFlagRowStart, pbyFlagRowEnd,
                ppbTmp, pi, nNum++, pnOut);
        }
    }

    MEMO_FREE_AND_NULL_N(pbyFlag);
    MEMO_FREE_AND_NULL_N(ppbTmp);

    return           nNum; 
}

// 8邻域区域生长
void XhDccsBase::FillNeighbor_8(const BYTE* pbyImg, const SIZE& szImg, int nLastCol,
                                BYTE* pbyFlag, BYTE* pbyFlagRowStart, 
                                BYTE* pbyFlagRowEnd, BYTE** ppbyStack, 
                                const BYTE* pbyPxl, int nLabelInd, int* pnOut)
{
    BYTE** ppbPointSet = ppbyStack; 
    int    nPointNum   = 0; 
    int    nOffset     = 0;
    int    nX		   = 0;

    ppbPointSet[nPointNum++] = pbyFlag + (pbyPxl - pbyImg);

    int    nOffNW	   = -szImg.cx - 1;
    int    nOffNE	   = -szImg.cx + 1;
    int    nOffSW	   = szImg.cx - 1;
    int    nOffSE	   = szImg.cx + 1;

    while (nPointNum > 0) 
    {
        // pop a point
        BYTE* pf   = ppbPointSet[--nPointNum]; 

        // label it 
        nOffset        = pf - pbyFlag; // offset from origin
        *pf            = 1;  
        pnOut[nOffset] = nLabelInd; 
        const BYTE* px = pbyImg + nOffset;

        // check surroundings
        nX = nOffset % szImg.cx; 

        // 左
        if (nX != 0	&& !pf[-1] && px[-1] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf - 1;
        }

        // 右
        if (nX != nLastCol && !pf[1] && px[1] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf + 1;
        }

        // 上
        if (pf >= pbyFlagRowStart && !pf[-szImg.cx] && px[-szImg.cx] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf - szImg.cx;
        }

        // 下
        if (pf <= pbyFlagRowEnd && !pf[szImg.cx] && px[szImg.cx] == *pbyPxl)  
        {
            ppbPointSet[nPointNum++] = pf + szImg.cx; 
        }

        // 左上
        if (nX != 0 && pf >= pbyFlagRowStart	&& !pf[nOffNW] && px[nOffNW] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf + nOffNW; 
        }

        // 右上
        if (nX != nLastCol && pf >= pbyFlagRowStart && !pf[nOffNE] && px[nOffNE] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf + nOffNE; 
        }

        // 左下
        if (nX != 0 && pf <= pbyFlagRowEnd && !pf[nOffSW] && px[nOffSW] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf + nOffSW; 
        }

        // 右下
        if (nX != nLastCol && pf <= pbyFlagRowEnd && !pf[nOffSE] && px[nOffSE] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf + nOffSE; 
        }

    } // while
}

// 4邻域区域生长
void XhDccsBase::FillNeighbor_4(const BYTE* pbyImg, const SIZE& szImg, int nLastCol,
                                BYTE* pbyFlag, BYTE* pbyFlagRowStart, 
                                BYTE* pbyFlagRowEnd, BYTE** ppbyStack, 
                                const BYTE* pbyPxl, int nLabelInd, int* pnOut)
{
    BYTE** ppbPointSet = ppbyStack; 
    int    nPointNum   = 0; 
    int    nOffset     = 0;
    int    nX		   = 0;

    ppbPointSet[nPointNum++] = pbyFlag + (pbyPxl - pbyImg);

    while (nPointNum > 0) 
    {
        // pop a point
        BYTE* pf   = ppbPointSet[--nPointNum]; 

        // label it 
        nOffset         = pf - pbyFlag; // offset from origin
        *pf			    = 1;  
        pnOut[nOffset]  = nLabelInd; 
        const BYTE* px	= pbyImg + nOffset;

        // check surroundings
        nX = nOffset % szImg.cx; 

        // 左
        if (nX != 0	&& !pf[-1] && px[- 1] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf - 1;
        }

        // 右
        if (nX != nLastCol && !pf[1] && px[1] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf + 1;
        }

        // 上
        if (pf >= pbyFlagRowStart && !pf[-szImg.cx] && px[-szImg.cx] == *pbyPxl)
        {
            ppbPointSet[nPointNum++] = pf - szImg.cx;
        }

        // 下
        if (pf <= pbyFlagRowEnd && !pf[szImg.cx] && px[szImg.cx] == *pbyPxl)  
        {
            ppbPointSet[nPointNum++] = pf + szImg.cx; 
        }
    } // while
}


// sobel算子梯度
void XhDccsBase::SobelPowGrad(const BYTE* pbyGray, const SIZE& szGray, int nBpp, 
                              const BYTE* pbyMask, int* pnGrad, double& dbMeanConv, int nDir)
{
    assert(pbyGray);
    assert(pbyMask);
    assert(pnGrad);
    assert(nBpp == 8);

    // 初始化
    int     nBufLen    = szGray.cx * szGray.cy;	
    memset(pnGrad, 0, nBufLen * sizeof(*pnGrad));

    // 初始化偏移
    int           nLastRow = szGray.cy - 1;
    int           nLastCol = szGray.cx - 1;

    // 四个角落，S-南，W-西，N-北，E-东
    int           nOffSW   = nLastCol;
    int           nOffSE   = szGray.cx + 1;
    int           nOffNW   = -nOffSE;
    int           nOffNE   = -nOffSW;

    // 起始点
    const BYTE*   pi       = pbyGray + nOffSE;
    const BYTE*   pm       = pbyMask + nOffSE;
    int*          pg	   = pnGrad + nOffSE;
    int           nSum     = 0;
    int           i, j;

    // 以下三个变量用于统计卷积平均值，需要计算总合，若浮点求和不会显著增加耗时，可以将nConvSum 改为double型的dbConvSum
    int           nConvSum = 0;
    int           nCount   = 0;    
    int           nIntLimit = - (255 * 4 * 255 * 4);   // int型数据防止溢出边界，[255*（1+2+1）]^2 是单个像素点理论最大梯度值

    for (int i = 0; i < 31; i ++)   
    {
        nIntLimit += (1 << i);
    }

    // 卷积
    if (nDir == DIR_HORIZONTAL)
    {
        // 算子为  -1, -2, -1 
        //          0,  0,  0
        //          1,  2,  1   // 这里负数在上是因为位图格式数据与图像上下颠倒 
        for (j = 1; j < nLastRow; j++)
        {
            for (i = 1; i < nLastCol; i++, pi++, pm++, pg++)
            {
                if (*pm)
                {
	                nSum =  pi[nOffSW] + (pi[szGray.cx] << 1) + pi[nOffSE]
                        - pi[nOffNW] - (pi[-szGray.cx] << 1) - pi[nOffNE];	
	                *pg  = nSum * nSum;
	
	                if (nConvSum < nIntLimit)   // 若nCouvSum采用double型，可避免此类判断
	                {
	                    nConvSum += *pg;
	                    nCount++;               // 若nCouvSum采用double型，可避免累加计算，直接用图像尺寸计算累加点数
	                }   
                }
            }

            pi += 2;
            pm += 2;
            pg += 2;
        }
    }
    else
    {
        // 算子为  -1, 0, 1
        //         -2, 0, 2
        //         -1, 0, 1  
        for (j = 1; j < nLastRow; j++)
        {
            for (i = 1; i < nLastCol; i++, pi++, pm++, pg++)
            {
                if (*pm)
                {
	                nSum  = pi[nOffNE] + (pi[1] << 1) + pi[nOffSE]
	                            - pi[nOffNW] - (pi[-1] << 1) - pi[nOffSW];
	                *pg   = nSum * nSum;
	
	                if (nConvSum < nIntLimit)  // 若nCouvSum采用double型，可避免此类判断
	                {
	                    nConvSum += *pg;
	                    nCount++;              // 若nCouvSum采用double型，可避免累加计算，直接用图像尺寸计算累加点数
	                }
                }
   
            }

            pi += 2;
            pm += 2;
            pg += 2;
        }
    }

    dbMeanConv = (double)nConvSum / nCount;
}

// sobel算子梯度
void XhDccsBase::SobelGrad(const BYTE* pbyGray, const SIZE& szGray, int nBpp, 
                           const BYTE* pbyMask, int* pnGrad, int nDir)
{
    assert(pbyGray);
    assert(pbyMask);
    assert(pnGrad);
    assert(nBpp == 8);

    // 初始化
    int     nBufLen    = szGray.cx * szGray.cy;	
    memset(pnGrad, 0, nBufLen * sizeof(*pnGrad));

    // 初始化偏移
    int           nLastRow = szGray.cy - 1;
    int           nLastCol = szGray.cx - 1;

    // 四个角落，S-南，W-西，N-北，E-东
    int           nOffSW   = nLastCol;
    int           nOffSE   = szGray.cx + 1;
    int           nOffNW   = -nOffSE;
    int           nOffNE   = -nOffSW;

    // 起始点
    const BYTE*   pi       = pbyGray + nOffSE;
    const BYTE*   pm       = pbyMask + nOffSE;
    int*          pg	   = pnGrad + nOffSE;
    int           nSum     = 0;
    int           i, j;


    // 卷积
    if (nDir == DIR_HORIZONTAL)
    {
        // 算子为  -1, -2, -1 
        //          0,  0,  0
        //          1,  2,  1   // 这里负数在上是因为位图格式数据与图像上下颠倒 
        for (j = 1; j < nLastRow; j++)
        {
            for (i = 1; i < nLastCol; i++, pi++, pm++, pg++)
            {
                if (*pm)
                {
                    nSum =  pi[nOffSW] + (pi[szGray.cx] << 1) + pi[nOffSE]
                    - pi[nOffNW] - (pi[-szGray.cx] << 1) - pi[nOffNE];	
                    *pg  = abs(nSum); 
                }
            }

            pi += 2;
            pm += 2;
            pg += 2;
        }
    }
    else
    {
        // 算子为  -1, 0, 1
        //         -2, 0, 2
        //         -1, 0, 1  
        for (j = 1; j < nLastRow; j++)
        {
            for (i = 1; i < nLastCol; i++, pi++, pm++, pg++)
            {
                if (*pm)
                {
                    nSum  = pi[nOffNE] + (pi[1] << 1) + pi[nOffSE]
                    - pi[nOffNW] - (pi[-1] << 1) - pi[nOffSW];
                    *pg   = abs(nSum);
                }
            }

            pi += 2;
            pm += 2;
            pg += 2;
        }
    }
}

// hewitt算子梯度
void XhDccsBase::HewittGrad(const BYTE* pbyGray, const SIZE& szGray, int nBpp, 
                            const BYTE* pbyMask, const BYTE* pbyEnhance, 
                            int* pnGrad, double& dbMeanConv, int nDir)

{
    assert(pbyGray);
    assert(pbyMask);
    assert(pbyEnhance);
    assert(pnGrad);
    assert(nBpp == 8);

    // 初始化
    int     nBufLen    = szGray.cx * szGray.cy;	
    memset(pnGrad, 0, nBufLen * sizeof(*pnGrad));

    // 初始化偏移
    int           nLastRow = szGray.cy - 1;
    int           nLastCol = szGray.cx - 1;

    // 四个角落，S-南，W-西，N-北，E-东
    int           nOffSW   = nLastCol;
    int           nOffSE   = szGray.cx + 1;
    int           nOffNW   = -nOffSE;
    int           nOffNE   = -nOffSW;

    // 起始点
    const BYTE*   pi       = pbyGray + nOffSE;
    const BYTE*   pm       = pbyMask + nOffSE;
    const BYTE*   pc       = pbyEnhance + nOffSE;
    int*          pg	   = pnGrad + nOffSE;
    int           nSum     = 0;
    int           i, j;

    // 以下变量用于统计卷积平均值
    int           nConvSum = 0;        // hewitt梯度，计算卷积后取绝对值而不是平方，因此nConvSum在一般尺寸图像上不会溢出
    int           nCount   = 0;

    // 卷积
    if (nDir == DIR_HORIZONTAL)
    {
        // 算子为 -1, -1, -1 
        //         0,  0,  0
        //         1,  1,  1   // 这里负数在上是因为位图格式数据与图像上下颠倒 
        for (j = 1; j < nLastRow; j++)
        {
            for (i = 1; i < nLastCol; i++, pi++, pm++, pc++, pg++)
            {
                if (*pm)
                {
                    nSum = pi[nOffSW] + pi[szGray.cx] + pi[nOffSE]
                                - pi[nOffNW] - pi[-szGray.cx] - pi[nOffNE]; 

                    if (*pc)
                    {
                        *pg = abs(nSum) * 2;
                    }
                    else
                    {
                        *pg = abs(nSum);
                    }

                    nConvSum += *pg;
                    nCount++;
                }
            }

            pi += 2;
            pm += 2;
            pc += 2;
            pg += 2;
        }
    }
    else
    {
        // 算子为  -1, 0, 1
        //         -1, 0, 1
        //         -1, 0, 1  
        for (j = 1; j < nLastRow; j++)
        {
            for (i = 1; i < nLastCol; i++, pi++, pm++, pc++, pg++)
            {
                if (*pm)
                {
	                nSum      = pi[nOffNE] + pi[1] + pi[nOffSE]
	                            - pi[nOffNW] - pi[-1] - pi[nOffSW];
	
	                if (*pc)
	                {
	                    *pg = abs(nSum) * 2;
	                }
	                else
	                {
	                    *pg = abs(nSum);
	                }
	
	                nConvSum += *pg;
                    nCount++;
                }
            }

            pi += 2;
            pc += 2;
            pg += 2;
        }
    }

    dbMeanConv = (double)nConvSum / nCount;
}

// 由梯度计算边缘图
void XhDccsBase::Grad2Edge(const int* pnGrad, const SIZE& szGrad, const BYTE* pbyMask,
                           double dbMeanGrad, double dbLevel, BYTE* pbyEdge, int nDir)
{
    assert(pnGrad);
    assert(pbyMask);
    assert(pbyEdge);
    assert(nDir);

    int     nBufLen    = szGrad.cx * szGrad.cy;	
    memset(pbyEdge, 0, nBufLen * sizeof(*pbyEdge));

    // 阈值
    int          nThresh  = (int)(dbMeanGrad * dbLevel);  // 化作int型，避免不同类型数据比较

    // 初始化偏移
    int          nLastRow = szGrad.cy - 1;
    int          nLastCol = szGrad.cx - 1;

    // 起始点
    const int*   pg	      = pnGrad + szGrad.cx + 1;    
    const BYTE*  pf       = pbyMask + szGrad.cx + 1;  // 掩膜
    BYTE*        po	      = pbyEdge + szGrad.cx + 1;
    int          i, j;

    // 二值化  
    if (nDir == DIR_HORIZONTAL)
    {
        for (j = 1; j < nLastRow; j++)
        {
            for (i = 1; i < nLastCol; i++, pg++, pf++, po++)
            {
                if (*pg > nThresh && (*pf) && *pg > pg[-szGrad.cx] && *pg >= pg[szGrad.cx])
                {
                    *po = 255;
                }
            }

            pg += 2;
            pf += 2;
            po += 2;
        }
    }
    else
    {
        for (j = 1; j < nLastRow; j++)
        {
            for (i = 1; i < nLastCol; i++, pg++, pf++, po++)
            {
                if (*pg > nThresh && (*pf) && *pg > pg[1] && *pg >= pg[-1])
                {
                    *po = 255;
                }
            }

            pg += 2;
            pf += 2;
            po += 2;
        }
    }
}

// 蓝黄色区域提取
void XhDccsBase::FlagPlateColor(const BYTE* pbyImg, const SIZE& szImg, int nBpp,
                                const BYTE* pbyMask, BYTE* pbyBlue, 
                                BYTE* pbyYellow, BYTE* pbyFlag)
{
    assert(pbyImg);
    assert(pbyBlue);
    assert(pbyYellow);
    assert(pbyFlag);
    assert(nBpp == 24);

    // 初始化
    int nBuflen   = szImg.cx * szImg.cy;	
    memset(pbyBlue,   0, nBuflen * sizeof(*pbyBlue));
    memset(pbyYellow, 0, nBuflen * sizeof(*pbyYellow));
    memset(pbyFlag,   0, nBuflen * sizeof(*pbyFlag));

    // 偏移点
    int         nChannels = (nBpp >> 3);
    int         nFirstCol = 2;
    int         nLastCol  = szImg.cx - 1 - nFirstCol;
    int         nOffGray  = nFirstCol * 2;
    int         nOffColor = nOffGray * nChannels;	   

    // 起始点
    const BYTE* pi        = pbyImg    + nFirstCol * nChannels;  
    const BYTE* pm        = pbyMask   + nFirstCol;
    BYTE*       pob       = pbyBlue   + nFirstCol;    
    BYTE*       poy       = pbyYellow + nFirstCol;  
    int         i, j;

    for (j = 0; j < szImg.cy; j++)
    {
        for (i = nFirstCol; i <= nLastCol; i++, pm++, pob++, poy++)
        {
            if (*pm)
            {
                if (IsBlue(pi))
                {
                    memset(pob - 2, COLOR_BLUE, 5);
                }
                else if (IsYellow(pi))
                {
                    memset(pob - 2, COLOR_YELLOW, 5);	// bug? maybe fix by pob -> poy
                }
            }

            pi += 3;
        }

        pi  += nOffColor;
        pm  += nOffGray;
        pob += nOffGray;
        poy += nOffGray;
    }    

    pob       = pbyBlue;    
    poy       = pbyYellow;  
    BYTE*  pf = pbyFlag;  

#ifdef SSE_OPTIMIZE
	static __m128i __color_exist = {COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST, COLOR_EXIST};
	__m128i xmm7 = _mm_setzero_si128();
	for(nBuflen-=16; nBuflen>=0; nBuflen-=16, pob+=16, poy+=16, pf+=16)
	{
		__m128i xmm0 = _mm_loadu_si128((__m128i*)pob);
		__m128i xmm1 = _mm_loadu_si128((__m128i*)poy);
		xmm0 = _mm_cmpeq_epi8(xmm0, xmm7);
		xmm1 = _mm_cmpeq_epi8(xmm1, xmm7);
		_mm_storeu_si128((__m128i*)pf, _mm_andnot_si128(_mm_and_si128(xmm0, xmm1), __color_exist));
	}
	nBuflen += 16;
#endif
    for (i = 0; i < nBuflen; i++, pob++, poy++, pf++)
    {
        if (*pob || *poy)
        {
            *pf = COLOR_EXIST;
        }
    }
}

// 实现两段数据的点乘，最后返回乘积之和
double XhDccsBase::MultyDot(double* dbSrcA, double* dbSrcB, int nLen)
{
#ifdef SSE_OPTIMIZE
    return dotproduct_d(dbSrcA, dbSrcB, nLen);
#endif
    double dbCnt = 0;

    for (int i = 0; i < nLen; i++)
    {
        dbCnt += dbSrcA[i] * dbSrcB[i];
    }

    return dbCnt;
}

// 判断是否为可能的蓝色点
bool XhDccsBase::IsBlue(const BYTE* pi)
{
    if (*pi > 32 && pi[1] < 224 && pi[2] < 176
        && *pi - pi[2] > 24 && *pi - pi[1] > 24)
    {
        return true;
    }

    return false;
}

// 判断是否为可能的黄色点
bool XhDccsBase::IsYellow(const BYTE* pi)
{
    if (*pi < 96 && pi[1] > 64 && pi[2] > 80
        && pi[2] - *pi > 64 && pi[1] - *pi > 48)
    {
        return true;
    }
    
    return false;
}

// 2*3算子图像膨胀（2行3列，边界不处理，膨胀后的图像尺寸不变）
void XhDccsBase::DilateFix_On2x3(const BYTE* pbyBinary, const SIZE& szBinary, 
                                 int nVal, BYTE* pbyDst)
{
    assert(pbyBinary);
    assert(pbyDst);
    assert(nVal >= 0);

    memset(pbyDst, 0, szBinary.cx * szBinary.cy * sizeof(*pbyDst));

    // 设置起始点
    int          nOffNW  = szBinary.cx + 1;
    const BYTE*  pi      = pbyBinary + nOffNW;
    BYTE*        po      = pbyDst    + nOffNW;
    int          i, j;    

    // 2行分别是当前行，与上面一行
    for (j = 1; j < szBinary.cy; j++)
    {
        for (i = 1; i < szBinary.cx - 1; i++, pi++, po++)
        {
            if (*pi)
            {
                memset(po - nOffNW, nVal, 3);
                memset(po - 1, nVal, 3);
            }
        }

        pi += 2;
        po += 2;
    }
}

// 3*3算子图像膨胀（3行3列，边界不处理，膨胀后的图像尺寸不变）
void XhDccsBase::DilateFix_On3x3(const BYTE* pbyBinary, const SIZE& szBinary, 
                                 int nVal, BYTE* pbyDst)
{
    assert(pbyBinary);
    assert(pbyDst);
    assert(nVal >= 0);

    memset(pbyDst, 0, szBinary.cx * szBinary.cy * sizeof(*pbyDst));

    // 设置起始点
    int          nOffNW  = szBinary.cx + 1;
    int          nOffSW  = szBinary.cx - 1;
    const BYTE*  pi      = pbyBinary + nOffNW;
    BYTE*        po      = pbyDst    + nOffNW;
    int          i, j;    

    // 3行分别是当前行，上面一行与下面一行
    for (j = 1; j < szBinary.cy - 1; j++)
    {
        for (i = 1; i < szBinary.cx - 1; i++, pi++, po++)
        {
            if (*pi)
            {
                memset(po - nOffNW, nVal, 3);
                memset(po - 1, nVal, 3);
                memset(po + nOffSW, nVal, 3);
            }
        }

        pi += 2;
        po += 2;
    }
}

// 3*n算子图像膨胀（3行n列，边界处理，膨胀后的图像尺寸增加）
void XhDccsBase::DilateVary_On3xn(const BYTE* pbyBinary, const SIZE& szBinary,
                                  int nKenelWidth, int nVal, 
                                  const SIZE& szDst, BYTE* pbyDst)
{
    assert(pbyBinary);
    assert(pbyDst);
    assert(nVal >= 0);

    memset(pbyDst, 0, szBinary.cx * szBinary.cy * sizeof(*pbyDst));

    int          nXcoef = (nKenelWidth >> 2);
    const BYTE*  pi     = pbyBinary;	
    BYTE*		 po     = pbyDst;	
    int          nOffD = 2 * nXcoef;
    int          nLenX = nOffD + 1;
    int          i, j;

    for (j = 0; j < szBinary.cy; j++)
    {
        for (i = 0; i < szBinary.cx; i++)
        {
            if (pi[i])
            {
                BYTE*  pTemp = po + i;		// Y方向上下各膨胀两行(nYcoef当前固定为2）
                memset(pTemp, nVal, nLenX);
                pTemp  += szDst.cx; 	
                memset(pTemp, nVal, nLenX);
                pTemp  += szDst.cx; 	
                memset(pTemp, nVal, nLenX);
            }
        }

        pi += szBinary.cx;
        po += szDst.cx;
    }
}

// 5*n算子图像膨胀（5行n列，边界处理，膨胀后的图像尺寸增加）
void XhDccsBase::DilateVary_On5xn(const BYTE* pbyBinary, const SIZE& szBinary,
                                  int nKenelWidth, int nVal, const SIZE& szDst, BYTE* pbyDst)
{
    assert(pbyBinary);
    assert(pbyDst);
    assert(nVal >= 0);

    memset(pbyDst, 0, szBinary.cx * szBinary.cy * sizeof(*pbyDst));

    int          nXcoef = (nKenelWidth >> 2);
    const BYTE*  pi     = pbyBinary;	
    BYTE*		 po     = pbyDst;	
    int          nOffD = 2 * nXcoef;
    int          nLenX = nOffD + 1;
    int          i, j;

    for (j = 0; j < szBinary.cy; j++)
    {
        for (i = 0; i < szBinary.cx; i++)
        {
            if (pi[i])
            {
                BYTE*  pTemp = po + i;		// Y方向上下各膨胀两行(nYcoef当前固定为2）
                memset(pTemp, nVal, nLenX);
                pTemp  += szDst.cx; 	
                memset(pTemp, nVal, nLenX);
                pTemp  += szDst.cx; 	
                memset(pTemp, nVal, nLenX);
                pTemp  += szDst.cx; 	
                memset(pTemp, nVal, nLenX);
                pTemp  += szDst.cx; 	
                memset(pTemp, nVal, nLenX);
            }
        }

        pi += szBinary.cx;
        po += szDst.cx;
    }
}

// 二值图像投影计算
void XhDccsBase::BwProjection(const BYTE* pbyBinary, const SIZE& szBinary, int nProjLen,
                              int* pnProj, int nMode, BYTE byBackPtVal)
{
    assert(pbyBinary);
    assert(pnProj);

    // 计算区域尺寸和起始点
    const BYTE* pCol  = pbyBinary;

    if (nMode ==DIR_HORIZONTAL)
    {
        assert(szBinary.cy == nProjLen);

        for (int j = 0; j < szBinary.cy; j++)
        {
#ifdef SSE_OPTIMIZE
			pnProj[j] = szBinary.cx - (int)calccnt8_eq_sse2((PBYTE)pCol, byBackPtVal, szBinary.cx);
#else
            pnProj[j] = szBinary.cx - std::count(pCol, pCol + szBinary.cx, byBackPtVal);  // 与一行像素个数减去背景点数
#endif
            pCol     += szBinary.cx;
        }
    }
    else
    {
        if (nMode == DIR_VERTICAL)
        {	
            assert(szBinary.cx == nProjLen);
            memset(pnProj, 0, nProjLen * sizeof(*pnProj));
            int			 i, j;     

#ifdef SSE_OPTIMIZE
			for(i=szBinary.cx-16; i>=0; i-=16, pCol+=16, pnProj+=16)
			{
				calccnt8_ver_sse2(pnProj, (PBYTE)pCol, szBinary.cx, szBinary.cy);
			}
			if((i+=16) > 0)
			{
				for(int k=0; k<i; ++k, pCol++)
				{
					PBYTE pi = (PBYTE)pCol;
					for(j=0; j<szBinary.cy; ++j, pi+=szBinary.cx)
						if(*pi)
							pnProj[k]++;
				}
			}
			return;
#endif
            for (i = 0; i < szBinary.cx; i++, pCol++)
            {
                const BYTE*  pi = pCol;

                for (j = 0; j < szBinary.cy; j++)
                {
                    if (*pi)
                    {
                        pnProj[i]++;
                    }

                    pi += szBinary.cx;
                }
            }
        }
    }
}

// 二值图像指定区域内投影
void XhDccsBase::LocProjection(const BYTE* pbyBinary, const SIZE& szBinary, int nProjLen,
                               const RECT& rtLoc, int* pnProj, int nMode, BYTE byBackPtVal)
{	
    assert(pbyBinary);
    assert(pnProj);

    // 计算区域尺寸和起始点
    SIZE        szLoc = {rtLoc.right - rtLoc.left + 1, rtLoc.bottom - rtLoc.top + 1};
    const BYTE* pCol  = pbyBinary + rtLoc.top * szBinary.cx + rtLoc.left;

    if (nMode ==DIR_HORIZONTAL)
    {
        assert(szLoc.cy == nProjLen);

        for (int j = 0; j < szLoc.cy; j++)
        {
#ifdef SSE_OPTIMIZE
			pnProj[j] = szLoc.cx - calccnt8_eq_sse2((unsigned char*)pCol, byBackPtVal, szLoc.cx);
#else
            pnProj[j] = szLoc.cx - std::count(pCol, pCol + szLoc.cx, byBackPtVal);  // 与一行像素个数减去背景点数
#endif
            pCol        += szBinary.cx;
        }
    }
    else
    {
        if (nMode == DIR_VERTICAL)
        {	
            assert(szLoc.cx == nProjLen);
            memset(pnProj, 0, nProjLen * sizeof(*pnProj));
            int			 i, j;     

#ifdef SSE_OPTIMIZE
			for(i=szLoc.cx-16; i>=0; i-=16, pCol+=16, pnProj+=16)
			{
				calccnt8_ver_sse2(pnProj, (PBYTE)pCol, szBinary.cx, szLoc.cy);
			}
			if((i+=16) > 0)
			{
				for(int k=0; k<i; ++k, pCol++)
				{
					PBYTE pi = (PBYTE)pCol;
					for(j=0; j<szLoc.cy; ++j, pi+=szBinary.cx)
						if(*pi)
							pnProj[k]++;
				}
			}
			return;
#endif
           for (i = 0; i < szLoc.cx; i++, pCol++)
            {
                const BYTE*  pi = pCol;

                for (j = 0; j < szLoc.cy; j++)
                {
                    if (*pi)
                    {
                        pnProj[i]++;
                    }

                    pi += szBinary.cx;
                }
            }
        }
    }
}