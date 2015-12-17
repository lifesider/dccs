#pragma once
#include "dccsbase.h"

#define MEMO_FREE_AND_NULL_N(p)     if (p){delete [] p; p = NULL;}
#define MEMO_FREE_AND_NULL(p)       if (p){delete p; p = NULL;}

#define OPENCL_FOR_GRAD_GMM

class xhGradsGMM
{
public:
	xhGradsGMM(void);
	~xhGradsGMM(void);

private:
	// 高斯混合模型结构体
	#define GmmMaxNum		   3
	struct ST_PixelPerGMM
	{
		float		fWeight;
		float       fGradX;
		float       fGradY;

		ST_PixelPerGMM()
		{
			fWeight = 0.0f;
			fGradX  = 0.0f;
			fGradY  = 0.0f;
		}
	};

	struct ST_PixelGMM		
	{
		float			 fVar;
		int			     nGMMUsedNum;				// 每个像素点的高斯模型个数
		ST_PixelPerGMM   pGMM[GmmMaxNum];			// 每个像素点可能会有几个高斯模型   

		ST_PixelGMM()
		{
			nGMMUsedNum = 0;
			fVar		= 0.0f;
		}
	};

public:
	// 重置
	int ReSet();

	//
	int SetCirMode(int nCirMode);

	//传入视频信息
	int SetVideoFrameInfo(int nWidth, int nHeight);

	//传入视频帧
	int SetVideoFrame(const BYTE* pbImage, const BYTE* pbMask, double dDriftTime);

	//得二值图
	int GetGmmBw(BYTE** pbBw);

	int GetDist(BYTE** pbDist);
	int GetBackImg(int** pnBackImg);

private:
	void InitialPixelGmm();

	int  MatchForeJudge(int* pnGrad, ST_PixelGMM* pstPixelGmm, BYTE* pbBw/*, int* pnBackGrad, BYTE* pbDist*/);
	void MatchForeJudge_Skip(int* pnGrad, ST_PixelGMM* pstPixelGmm, BYTE* pbBw/*, int* pnBackGrad, BYTE* pbDist*/);

	void UpdataNoMatchPixelGmm(int* pnGrad, ST_PixelGMM* pstPixelGmm);
	void UpdataMatchPixelGmm(ST_PixelGMM* pstPixelGmm, int nMatchFlag);

	void SortedByKey(ST_PixelPerGMM* pGMM, int nMatchIndex, float fMatchWeight);
	void Sobel(const BYTE* pbImage, const BYTE* pbMask);

private:
	BOOL			m_bInitalFrame;
	double			m_dForDriftTime;
	int				m_nWidth;
	int				m_nHeight;
	int				m_nSize;
	SIZE			m_szImg;

	float			m_fAlphaW;
	float			m_fAlphaM;
	float			m_fAlpha1;
	float			m_fCT;
	float			m_fVarRatio1;
	float			m_fVarRatio2;

	ST_PixelGMM*	m_pstPixelGmm;
	BYTE*			m_pbGmmBw;
	int*			m_pnGrad;
	int*			m_pnBackGrad;
	BYTE*			m_pbDist;

#ifdef OPENCL_FOR_GRAD_GMM
	cl_mem			m_clPixelGmm;
	cl_mem			m_clGmmBw;
	cl_mem			m_clGrad;
	cl_mem			m_clBackGrad;
	cl_mem			m_clDist;
#endif
};
