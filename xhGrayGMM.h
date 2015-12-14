#pragma once

#define MEMO_FREE_AND_NULL_N(p)     if (p){delete [] p; p = NULL;}
#define MEMO_FREE_AND_NULL(p)       if (p){delete p; p = NULL;}

class xhGrayGMM
{
public:
	xhGrayGMM(void);
	~xhGrayGMM(void);

private:
	// 高斯混合模型结构体
	#define GmmMaxNum		   3
	struct ST_PixelPerGMM
	{
		float		fWeight;
		float       frgbValue;

		ST_PixelPerGMM()
		{
			fWeight = 0.0f;
			frgbValue = 0.0f;
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
	int GetBackImg(BYTE** pbBackImg);

private:
	void InitialPixelGmm(const BYTE* pbImage);

	int	 MatchForeJudge(const BYTE* pbImage, ST_PixelGMM* pstPixelGmm, BYTE* pbBw/*, BYTE* pbBackImg, BYTE* pbDist*/);
	void MatchForeJudge_Skip(const BYTE* pbImage, ST_PixelGMM* pstPixelGmm, BYTE* pbBw/*, BYTE* pbBackImg, BYTE* pbDist*/);
	int  MatchForeJudge_sse(const BYTE* pbImage, ST_PixelGMM* pstPixelGmm, BYTE* pbBw);

	void UpdataNoMatchPixelGmm(const BYTE* pbImage, ST_PixelGMM* pstPixelGmm);
	void UpdataMatchPixelGmm(ST_PixelGMM* pstPixelGmm, int nMatchFlag);

	void SortedByKey(ST_PixelPerGMM* pGMM, int nMatchIndex, float fMatchWeight);

private:
	BOOL			m_bInitalFrame;
	double			m_dForDriftTime;
	int				m_nWidth;
	int				m_nHeight;
	int				m_nSize;

	float			m_fAlphaW;
	float			m_fAlphaM;
	float			m_fAlpha1;
	float			m_fCT;
	float			m_fVarRatio1;
	float			m_fVarRatio2;

	BYTE*			m_pbGmmBw;
	BYTE*			m_pbBackImg;
	BYTE*			m_pbDist;
	ST_PixelGMM*	m_pstPixelGmm;
};
