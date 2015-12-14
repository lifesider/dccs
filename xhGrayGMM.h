#pragma once

#define MEMO_FREE_AND_NULL_N(p)     if (p){delete [] p; p = NULL;}
#define MEMO_FREE_AND_NULL(p)       if (p){delete p; p = NULL;}

class xhGrayGMM
{
public:
	xhGrayGMM(void);
	~xhGrayGMM(void);

private:
	// ��˹���ģ�ͽṹ��
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
		int			     nGMMUsedNum;				// ÿ�����ص�ĸ�˹ģ�͸���
		ST_PixelPerGMM   pGMM[GmmMaxNum];			// ÿ�����ص���ܻ��м�����˹ģ��

		ST_PixelGMM()
		{
			nGMMUsedNum = 0;
			fVar		= 0.0f;
		}
	};

public:
	// ����
	int ReSet();

	//
	int SetCirMode(int nCirMode);

	//������Ƶ��Ϣ
	int SetVideoFrameInfo(int nWidth, int nHeight);

	//������Ƶ֡
	int SetVideoFrame(const BYTE* pbImage, const BYTE* pbMask, double dDriftTime);

	//�ö�ֵͼ
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
