#include "StdAfx.h"
#include "xhGradsGMM.h"
#include "dccsbase.h"

#define MinVar			 1.0f
#define MaxVar			 9.0f
#define InitialVar		 4.0f
#define VarRatio1		 /*6.25f*/9.0f
#define VarRatio2		 /*12.25f*/16.0f
#define VarRatio3		 4.0f/*6.25f*/
#define VarRatio4		 9.0f/*12.25f*/
#define Thres			 0.67f
#define DefaultFrameRate 15.0f
#define	WeightThL        0.17    // ??????阈值的设置还待考察
#define SlowWLearnRate   225
#define FastWLearnRate   60
#define MeanLearnRate	 30
#define FOREGROUND		 255
#define BACKGROUND		 0

xhGradsGMM::xhGradsGMM(void)
{
	m_pstPixelGmm	= NULL;
	m_pbGmmBw		= NULL;
	m_pnGrad		= NULL;
	m_pnBackGrad	= NULL;
	m_pbDist		= NULL;

	m_bInitalFrame	= TRUE;
	m_fVarRatio1	= VarRatio1;
	m_fVarRatio2	= VarRatio2;
}

xhGradsGMM::~xhGradsGMM(void)
{
	MEMO_FREE_AND_NULL_N(m_pstPixelGmm);
	MEMO_FREE_AND_NULL_N(m_pbGmmBw);
	MEMO_FREE_AND_NULL_N(m_pnGrad);
	MEMO_FREE_AND_NULL_N(m_pnBackGrad);
	MEMO_FREE_AND_NULL_N(m_pbDist);
}

int xhGradsGMM::ReSet()
{
	m_bInitalFrame	= TRUE;

	return 1;
}

int xhGradsGMM::SetCirMode(int nCirMode)
{
	if (nCirMode == 2)
	{
		m_fVarRatio1 = VarRatio3;
		m_fVarRatio2 = VarRatio4;
	}
	else
	{
		m_fVarRatio1 = VarRatio1;
		m_fVarRatio2 = VarRatio2;
	}
	return 1;
}

int xhGradsGMM::SetVideoFrameInfo(int nWidth, int nHeight)
{
	m_nWidth		= nWidth;
	m_nHeight		= nHeight;
	m_nSize			= m_nWidth * m_nHeight;
	m_szImg.cx		= m_nWidth;
	m_szImg.cy		= m_nHeight;

	MEMO_FREE_AND_NULL_N(m_pstPixelGmm);
	MEMO_FREE_AND_NULL_N(m_pnGrad);
	MEMO_FREE_AND_NULL_N(m_pnBackGrad);
	MEMO_FREE_AND_NULL_N(m_pbDist);
	MEMO_FREE_AND_NULL_N(m_pbGmmBw);
	m_pstPixelGmm	= new ST_PixelGMM[m_nSize + 31];	
	m_pnGrad		= new int[m_nSize * 2 + 31];
	m_pnBackGrad	= new int[m_nSize * 2 + 31];
	m_pbDist        = new BYTE[m_nSize + 31];
	m_pbGmmBw       = new BYTE[m_nSize + 31];
	ZeroMemory(m_pbGmmBw, sizeof(*m_pbGmmBw) * m_nSize);

	return 1;
}

void xhGradsGMM::Sobel(const BYTE* pbImage, const BYTE* pbMask)
{
	ZeroMemory(m_pnGrad, m_nSize * 2 * sizeof(*m_pnGrad));

	int nOffSetLU = -m_nWidth - 1;
	int nOffSetRU = -m_nWidth + 1;
	int nOffSetLB = m_nWidth  - 1;
	int nOffSetRB = m_nWidth  + 1;
	int nRow      = m_nWidth  - 1;
	int i, j;

	const BYTE* pbImgTmp  = pbImage + m_nWidth + 1;
	int*        pnGradTmp = m_pnGrad + (m_nWidth + 1) * 2;
	if (pbMask == NULL)
	{
		for (i = 1; i < m_nHeight - 1; i++)
		{
			for (j = 1; j < m_nWidth - 1; j++)
			{
				pnGradTmp[0] = ((pbImgTmp[nOffSetLB] - pbImgTmp[nOffSetLU]) + 
							2 * (pbImgTmp[m_nWidth]  - pbImgTmp[-m_nWidth]) + 
								(pbImgTmp[nOffSetRB] - pbImgTmp[nOffSetRU])) >> 3;

				pnGradTmp[1] = ((pbImgTmp[nOffSetLU] - pbImgTmp[nOffSetRU]) + 
							2 * (pbImgTmp[-1]        - pbImgTmp[1]) + 
								(pbImgTmp[nOffSetLB] - pbImgTmp[nOffSetRB])) >> 3;
		
				pbImgTmp++;
				pnGradTmp += 2;
			}

			pbImgTmp  += 2;
			pnGradTmp += 4;
		}
	}
	else
	{
		const BYTE* pbMaskTmp = pbMask + m_nWidth + 1;
		for (i = 1; i < m_nHeight - 1; i++)
		{
			for (j = 1; j < m_nWidth - 1; j++)
			{
				if (*pbMaskTmp != 0)
				{
					pnGradTmp[0] = ((pbImgTmp[nOffSetLB] - pbImgTmp[nOffSetLU]) + 
								2 * (pbImgTmp[m_nWidth]  - pbImgTmp[-m_nWidth]) + 
								    (pbImgTmp[nOffSetRB] - pbImgTmp[nOffSetRU])) >> 3;

					pnGradTmp[1] = ((pbImgTmp[nOffSetLU] - pbImgTmp[nOffSetRU]) + 
   								2 * (pbImgTmp[-1]        - pbImgTmp[1]) + 
									(pbImgTmp[nOffSetLB] - pbImgTmp[nOffSetRB])) >> 3;
				}

				pbImgTmp++;
				pbMaskTmp++;
				pnGradTmp += 2;
			}

			pbImgTmp  += 2;
			pbMaskTmp += 2;
			pnGradTmp += 4;
		}
	}
}

int xhGradsGMM::SetVideoFrame(const BYTE* pbImage, const BYTE* pbMask, double dDriftTime)
{
	//sobel梯度计算
	Sobel(pbImage, pbMask);

	//高斯建模
	if (m_bInitalFrame == TRUE)
	{
		m_bInitalFrame  = FALSE;
		m_dForDriftTime = dDriftTime;

		//当进来为第一帧时进行初始化操作
		InitialPixelGmm();
// 		CopyMemory(m_pnBackGrad, m_pnGrad, m_nSize * 2 * sizeof(*m_pnGrad));
// 		ZeroMemory(m_pbDist, m_nSize);
	}
	else
	{
		if (dDriftTime - m_dForDriftTime >= 0.20 - 1.0e-5)
		{
			float fSlowWeightLearnRate = float(DefaultFrameRate * (dDriftTime - m_dForDriftTime) / SlowWLearnRate);
			float fFastWeightLearnRate = float(DefaultFrameRate * (dDriftTime - m_dForDriftTime) / FastWLearnRate);
			m_fAlphaM = float(DefaultFrameRate * (dDriftTime - m_dForDriftTime) / MeanLearnRate);

			int*         pnTemp1     = m_pnGrad;
			//int*       pnTemp2     = m_pnBackGrad;
			//BYTE*		 pbTemp3	 = m_pbDist;
			BYTE*		 pbTemp5	 = m_pbGmmBw;
			ST_PixelGMM* pstPixelGmm = m_pstPixelGmm;
			for (int j = 0; j < m_nSize; j++)
			{
				//判断是否有匹配的高斯分布
				//前景点判断
				int nMatchFlag = MatchForeJudge(pnTemp1, pstPixelGmm, pbTemp5/*, pnTemp2, pbTemp3*/);

				//更新各高斯模型
				if (nMatchFlag == -1) // 如果没有匹配的则分配新的高斯过程
				{
					m_fAlphaW = fSlowWeightLearnRate;
					m_fAlpha1 = 1 - m_fAlphaW;
					m_fCT	  = 0.05f * m_fAlphaW;

					UpdataNoMatchPixelGmm(pnTemp1, pstPixelGmm);

					//根据权重对模型进行排序
					SortedByKey(pstPixelGmm->pGMM, pstPixelGmm->nGMMUsedNum - 1, pstPixelGmm->pGMM[pstPixelGmm->nGMMUsedNum - 1].fWeight);
				}
				else
				{
					if (*pbTemp5 == FOREGROUND && pstPixelGmm->pGMM[nMatchFlag].fWeight > WeightThL)
					{
						m_fAlphaW = fFastWeightLearnRate;
					}
					else
					{
						m_fAlphaW = fSlowWeightLearnRate;
					}
					m_fAlpha1 = 1 - m_fAlphaW;
					m_fCT	  = 0.05f * m_fAlphaW;

					UpdataMatchPixelGmm(pstPixelGmm, nMatchFlag);	

					//根据权重对模型进行排序
					SortedByKey(pstPixelGmm->pGMM, nMatchFlag, pstPixelGmm->pGMM[nMatchFlag].fWeight);
				}

				pnTemp1 += 2;
				//pnTemp2 += 2;
				//pbTemp3++;
				pbTemp5++;
				pstPixelGmm++;
			}

			m_dForDriftTime = dDriftTime;
		}
		else
		{
			int*         pnTemp1     = m_pnGrad;
			//int*       pnTemp2     = m_pnBackGrad;
			//BYTE*		 pbTemp3	 = m_pbDist;
			BYTE*		 pbTemp5	   = m_pbGmmBw;
			ST_PixelGMM* pstPixelGmm   = m_pstPixelGmm;
			for (int i = 0; i < m_nSize; i++)
			{
				//判断是否有匹配的高斯分布
				//前景点判断
				MatchForeJudge_Skip(pnTemp1, pstPixelGmm, pbTemp5/*, pnTemp2, pbTemp3*/);

				pnTemp1 += 2;
				//pnTemp2 += 2;
				//pbTemp3++;
				pbTemp5++;
				pstPixelGmm++;
			}
		}
	}

	return 1;
}

void xhGradsGMM::InitialPixelGmm()
{
	int*         pnTemp1     = m_pnGrad;
	ST_PixelGMM* pstPixelGmm = m_pstPixelGmm;
	for (int i = 0; i < m_nSize; i++)
	{
		//给新模型赋权重、均值和方差
		pstPixelGmm->nGMMUsedNum	= 1;
		pstPixelGmm->fVar			= InitialVar;

		ST_PixelPerGMM* pGMM		= pstPixelGmm->pGMM;
		pGMM[0].fWeight				= 1.0f / SlowWLearnRate;
		pGMM[0].fGradX				= float(pnTemp1[0]);
		pGMM[0].fGradY				= float(pnTemp1[1]);

		pnTemp1 += 2;
		pstPixelGmm++;
	}
}

int xhGradsGMM::MatchForeJudge(int* pnGrad, ST_PixelGMM* pstPixelGmm, BYTE* pbBw/*, int* pnBackGrad, BYTE* pbDist*/)
{
	//归一化
	int				i			 = 0;
	float			fTotalWeight = 0;	
	ST_PixelPerGMM* pGMM		 = pstPixelGmm->pGMM;
	for (i = 0; i < pstPixelGmm->nGMMUsedNum; i++)
	{
		fTotalWeight += pGMM[i].fWeight;
	}

	if (fTotalWeight > 1)
	{
		for (i = 0; i < pstPixelGmm->nGMMUsedNum; i++)
		{
			pGMM[i].fWeight /= fTotalWeight;	
		}
		fTotalWeight = 1.0f;
	}

	//匹配判断
	float			fDiffX, fDiffY, fDist;
	float           fThres1    = m_fVarRatio1 * pstPixelGmm->fVar;
	float           fThres2    = /*max(*/m_fVarRatio2 * pstPixelGmm->fVar/*, 36)*/;
	int				nMatchFlag1= -1;
	int				nMatchFlag2= -1;
	for (i = 0; i < pstPixelGmm->nGMMUsedNum; i++)
	{
		//根据3倍方差原则判断是否匹配
		fDiffX = pGMM[i].fGradX - pnGrad[0];
		fDiffY = pGMM[i].fGradY - pnGrad[1];
// 		fDist  = fDiffX * fDiffX + fDiffY *fDiffY;
		fDist  = fabs(fDiffX) + fabs(fDiffY);
		fDist  = fDist * fDist;
		if (nMatchFlag2 < 0 && fDist < fThres2)
		{
			nMatchFlag2 = i;
		}
		if (fDist < fThres1)
		{
			nMatchFlag1 = i;

			//更新匹配的高斯模型的均值和方差
// 			double fLearnRate   = m_fAlphaT / pGMM[i].fWeight;
			pGMM[i].fWeight	    = m_fAlpha1 * pGMM[i].fWeight - m_fCT + m_fAlphaW;
			pGMM[i].fGradX	   -= m_fAlphaM * fDiffX;
			pGMM[i].fGradY	   -= m_fAlphaM * fDiffY;
			pstPixelGmm->fVar  -= 0.01f/*fLearnRate*/ * (pstPixelGmm->fVar - fDist);
			if (pstPixelGmm->fVar < MinVar)  //防止方差过小或过大
			{
				pstPixelGmm->fVar = MinVar;
			}
			else if (pstPixelGmm->fVar > MaxVar)
			{
				pstPixelGmm->fVar = MaxVar;
			}

			break;
		}
	}

	//前景点判断
	if (nMatchFlag2 == 0)
	{
		*pbBw		  = BACKGROUND; //击中第一个高斯模型，肯定是背景点
// 		*pbDist		  = 0;
// 		pnBackGrad[0] = pnGrad[0];
// 		pnBackGrad[1] = pnGrad[1];
	}
	else if (nMatchFlag2 == -1)
	{
		*pbBw		  = FOREGROUND; //没有匹配高斯模型，肯定是前景点
// 		*pbDist		  = (abs((pnBackGrad[0] - pnGrad[0]) + abs(pnBackGrad[1] - pnGrad[1])))/* / 2*/;
	}
	else
	{
		double fSumWeight = 0;
		for (i = 0; i < nMatchFlag2; i++)
		{
			fSumWeight += pGMM[i].fWeight;
		}

		if (fSumWeight < Thres * fTotalWeight)
		{
			*pbBw		  = BACKGROUND; //排在击中高斯模型前的几个模型累计权重和小于Thres，则为背景点
// 			*pbDist		  = 0;
// 			pnBackGrad[0] = pnGrad[0];
// 			pnBackGrad[1] = pnGrad[1];
		}
		else
		{
			*pbBw		  = FOREGROUND; //排在击中高斯模型前的几个模型累计权重和大于Thres，则为前景点
// 			*pbDist		  = (abs((pnBackGrad[0] - pnGrad[0]) + abs(pnBackGrad[1] - pnGrad[1])))/* / 2*/;
		}
	}

	return nMatchFlag1;
}

void xhGradsGMM::MatchForeJudge_Skip(int* pnGrad, ST_PixelGMM* pstPixelGmm, BYTE* pbBw/*, int* pnBackGrad, BYTE* pbDist*/)
{
	//归一化
	int				i			 = 0;
	float			fTotalWeight = 0;	
	ST_PixelPerGMM* pGMM		 = pstPixelGmm->pGMM;
	for (i = 0; i < pstPixelGmm->nGMMUsedNum; i++)
	{
		fTotalWeight += pGMM[i].fWeight;
	}

	if (fTotalWeight > 1)
	{
		for (i = 0; i < pstPixelGmm->nGMMUsedNum; i++)
		{
			pGMM[i].fWeight /= fTotalWeight;	
		}
		fTotalWeight = 1.0f;
	}

	//匹配判断
	float			fDiffX, fDiffY, fDist;
	float           fThres     = /*max(*/m_fVarRatio2 * pstPixelGmm->fVar/*, 36)*/;
	int				nMatchFlag = -1;
	for (i = 0; i < pstPixelGmm->nGMMUsedNum; i++)
	{
		//根据3倍方差原则判断是否匹配
		fDiffX = pGMM[i].fGradX - pnGrad[0];
		fDiffY = pGMM[i].fGradY - pnGrad[1];
// 		fDist  = fDiffX * fDiffX + fDiffY *fDiffY;
		fDist  = fabs(fDiffX) + fabs(fDiffY);
		fDist  = fDist * fDist;
		if (fDist < fThres)
		{
			nMatchFlag = i;
			break;
		}
	}

	//前景点判断
	if (nMatchFlag == 0)
	{
		*pbBw		  = BACKGROUND; //击中第一个高斯模型，肯定是背景点
// 		*pbDist		  = 0;
// 		pnBackGrad[0] = pnGrad[0];
// 		pnBackGrad[1] = pnGrad[1];
	}
	else if (nMatchFlag == -1)
	{
		*pbBw		  = FOREGROUND; //没有匹配高斯模型，肯定是前景点
// 		*pbDist		  = (abs((pnBackGrad[0] - pnGrad[0]) + abs(pnBackGrad[1] - pnGrad[1])))/* / 2*/;
	}
	else
	{
		double fSumWeight = 0;
		for (i = 0; i < nMatchFlag; i++)
		{
			fSumWeight += pGMM[i].fWeight;
		}

		if (fSumWeight < Thres * fTotalWeight)
		{
			*pbBw		  = BACKGROUND; //排在击中高斯模型前的几个模型累计权重和小于Thres，则为背景点
// 			*pbDist		  = 0;
// 			pnBackGrad[0] = pnGrad[0];
// 			pnBackGrad[1] = pnGrad[1];
		}
		else
		{
			*pbBw		  = FOREGROUND; //排在击中高斯模型前的几个模型累计权重和大于Thres，则为前景点
// 			*pbDist		  = (abs((pnBackGrad[0] - pnGrad[0]) + abs(pnBackGrad[1] - pnGrad[1])))/* / 2*/;
		}
	}
}

void xhGradsGMM::UpdataNoMatchPixelGmm(int* pnGrad, ST_PixelGMM* pstPixelGmm)
{
	ST_PixelPerGMM* pGMM = pstPixelGmm->pGMM;

	//更新各个模型的权重值
	int nGMMUsedNum = pstPixelGmm->nGMMUsedNum;
	for (int i = 0; i < nGMMUsedNum; i++)
	{
		pGMM[i].fWeight = m_fAlpha1 * pGMM[i].fWeight - m_fCT;
		if (pGMM[i].fWeight < m_fCT)
		{
			//当前位置权重置零
			pGMM[i].fWeight = 0.0f;

			//模型数减一
			pstPixelGmm->nGMMUsedNum--;
		}
	}

	//如果有模型被删除掉，加入的新模型覆盖掉被删除的模型中位置最靠前的模型
	//如果没有模型被删除掉且模型数没有达到最大，加入个新模型，否则用新模型替代权重最小的模型
	if (pstPixelGmm->nGMMUsedNum < GmmMaxNum)
	{
		pstPixelGmm->nGMMUsedNum++;
	}

	//给新模型赋权重、均值和方差
	int nLastModeIndex = pstPixelGmm->nGMMUsedNum - 1;
	pGMM[nLastModeIndex].fWeight   = m_fAlphaW;
	pGMM[nLastModeIndex].fGradX    = float(pnGrad[0]);
	pGMM[nLastModeIndex].fGradY	   = float(pnGrad[1]);
}

void xhGradsGMM::UpdataMatchPixelGmm(ST_PixelGMM* pstPixelGmm, int nMatchFlag)
{
	ST_PixelPerGMM* pGMM = pstPixelGmm->pGMM;

	//更新高斯模型的权重
	int nGMMUsedNum = pstPixelGmm->nGMMUsedNum;
	for (int i = 0; i < nGMMUsedNum; i++)
	{
		if (i != nMatchFlag)
		{
			//对不匹配的模型权重进行更新
			pGMM[i].fWeight = m_fAlpha1 * pGMM[i].fWeight - m_fCT;

			//当权重小于0时，删除该模型
			if (pGMM[i].fWeight < m_fCT)
			{
				//当前位置权重置零
				pGMM[i].fWeight = 0.0f;

				//模型数减一
				pstPixelGmm->nGMMUsedNum--;
			}
		}
	}
}


void xhGradsGMM::SortedByKey(ST_PixelPerGMM* pGMM, int nMatchIndex, float fMatchWeight)
{
	int i = 0;
	for (i = nMatchIndex; i > 0; i--)
	{
		if (fMatchWeight <= pGMM[i - 1].fWeight)
		{
			break;
		}
		else
		{
			ST_PixelPerGMM temp = pGMM[i - 1];
			pGMM[i - 1] = pGMM[i];
			pGMM[i]     = temp;
		}
	}
}

int xhGradsGMM::GetGmmBw(BYTE** pbBw)
{
	*pbBw = m_pbGmmBw;
	return 1;
}

int xhGradsGMM::GetDist(BYTE** pbDist)
{
	*pbDist = m_pbDist;
	return 1;
}

int xhGradsGMM::GetBackImg(int** pnBackImg)
{
	*pnBackImg = m_pnBackGrad;
	return 1;
}