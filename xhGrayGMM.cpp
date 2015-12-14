#include "StdAfx.h"
#include "xhGrayGMM.h"
#include "dccsbase.h"

#define MinVar			 4.0f
#define MaxVar			 9.0f
#define InitialVar		 6.25f
#define VarRatio1		 /*6.25f*/9.0f
#define VarRatio2		 /*12.25f*/16.0f
#define VarRatio3		 4.0f/*6.25f*/
#define VarRatio4		 9.0f/*12.25f*/
#define Thres			 0.67f
#define DefaultFrameRate 15.0f
#define	WeightThL        0.17    // ??????��ֵ�����û�������
#define SlowWLearnRate   225
#define FastWLearnRate   60
#define MeanLearnRate	 30
#define FOREGROUND       255
#define BACKGROUND		 0

xhGrayGMM::xhGrayGMM(void)
{
	m_pstPixelGmm	= NULL;
	m_pbGmmBw		= NULL;
	m_pbBackImg		= NULL;
	m_pbDist		= NULL;

	m_bInitalFrame	= TRUE;

	m_fVarRatio1	= VarRatio1;
	m_fVarRatio2	= VarRatio2;
}

xhGrayGMM::~xhGrayGMM(void)
{
	MEMO_FREE_AND_NULL_N(m_pstPixelGmm);
	MEMO_FREE_AND_NULL_N(m_pbGmmBw);
	MEMO_FREE_AND_NULL_N(m_pbBackImg);
	MEMO_FREE_AND_NULL_N(m_pbDist);
}

int xhGrayGMM::ReSet()
{
	m_bInitalFrame	= TRUE;

	return 1;
}

int xhGrayGMM::SetCirMode(int nCirMode)
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
int xhGrayGMM::SetVideoFrameInfo(int nWidth, int nHeight)
{
	m_nWidth		= nWidth;
	m_nHeight		= nHeight;
	m_nSize			= m_nWidth * m_nHeight;

	MEMO_FREE_AND_NULL_N(m_pstPixelGmm);
	MEMO_FREE_AND_NULL_N(m_pbGmmBw);
	MEMO_FREE_AND_NULL_N(m_pbBackImg);
	MEMO_FREE_AND_NULL_N(m_pbDist);
	m_pstPixelGmm	= new ST_PixelGMM[m_nSize + 31];	
	m_pbGmmBw       = new BYTE[m_nSize + 31];
	m_pbBackImg		= new BYTE[m_nSize + 31];
	m_pbDist		= new BYTE[m_nSize + 31];
	ZeroMemory(m_pbGmmBw, sizeof(*m_pbGmmBw) * m_nSize);

	return 1;
}

void xhGrayGMM::MatchForeJudge_Skip(const BYTE* pbImage, ST_PixelGMM* pstPixelGmm, BYTE* pbBw/*, BYTE* pbBackImg, BYTE* pbDist*/)
{
	// ��һ��
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

	//
	float			fDiffV, fDist;
	float           fThres     = /*max(*/m_fVarRatio2 * pstPixelGmm->fVar/*, 36)*/;
	int				nMatchFlag = -1;
	for (i = 0; i < pstPixelGmm->nGMMUsedNum; i++)
	{
		//����3������ԭ���ж��Ƿ�ƥ��
		fDiffV = pGMM[i].frgbValue - pbImage[0];
		fDist  = fDiffV * fDiffV;
		if (fDist < fThres)
		{
			nMatchFlag = i;
			break;
		}
	}

	//
	if (nMatchFlag == 0)
	{
		*pbBw		 = BACKGROUND; //���е�һ����˹ģ�ͣ��϶��Ǳ�����
	}
	else if (nMatchFlag == -1)
	{
		*pbBw		 = FOREGROUND; //û��ƥ���˹ģ�ͣ��϶���ǰ����
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
			*pbBw		 = BACKGROUND; //���ڻ��и�˹ģ��ǰ�ļ���ģ���ۼ�Ȩ�غ�С��Thres����Ϊ������
		}
		else
		{
			*pbBw		 = FOREGROUND; //���ڻ��и�˹ģ��ǰ�ļ���ģ���ۼ�Ȩ�غʹ���Thres����Ϊǰ����
		}
	}
}

int xhGrayGMM::SetVideoFrame(const BYTE* pbImage, const BYTE* pbMask, double dDriftTime)
{
	if (m_bInitalFrame == TRUE)
	{
		m_bInitalFrame  = FALSE;
		m_dForDriftTime = dDriftTime;

		//������Ϊ��һ֡ʱ���г�ʼ������
		InitialPixelGmm(pbImage);
	}
	else
	{
		if (dDriftTime - m_dForDriftTime >= 0.20 - 1.0e-5)
		{
			float fSlowWeightLearnRate = float(DefaultFrameRate * (dDriftTime - m_dForDriftTime) / SlowWLearnRate);
			float fFastWeightLearnRate = float(DefaultFrameRate * (dDriftTime - m_dForDriftTime) / FastWLearnRate);
			m_fAlphaM = float(DefaultFrameRate * (dDriftTime - m_dForDriftTime) / MeanLearnRate);

			BYTE*		 pbTemp1	   = (BYTE*)pbImage;
			BYTE*		 pbTemp5	   = m_pbGmmBw;
			ST_PixelGMM* pstPixelGmm   = m_pstPixelGmm;
			if (pbMask == NULL)
			{
				for (int i = 0; i < m_nSize; i++)
				{
					//�ж��Ƿ���ƥ��ĸ�˹�ֲ�
					//ǰ�����ж�
					int nMatchFlag = MatchForeJudge(pbTemp1, pstPixelGmm, pbTemp5/*, pbTemp2, pbTemp3*/);

					//���¸���˹ģ��
					if (nMatchFlag == -1) // ���û��ƥ���������µĸ�˹����
					{
						m_fAlphaW = fSlowWeightLearnRate;
						m_fAlpha1 = 1 - m_fAlphaW;
						m_fCT	  = 0.05f * m_fAlphaW;

						UpdataNoMatchPixelGmm(pbTemp1, pstPixelGmm);

						//����Ȩ�ض�ģ�ͽ�������
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

						//����Ȩ�ض�ģ�ͽ�������
						SortedByKey(pstPixelGmm->pGMM, nMatchFlag, pstPixelGmm->pGMM[nMatchFlag].fWeight);
					}

					pbTemp1++;
					pbTemp5++;
					pstPixelGmm++;
				}
			}
			else
			{
				BYTE* pbTemp2 = (BYTE*)pbMask;
				for (int i = 0; i < m_nSize; i++)
				{
					if (*pbTemp2 != 0)
					{
						//�ж��Ƿ���ƥ��ĸ�˹�ֲ�
						//ǰ�����ж�
						int nMatchFlag = MatchForeJudge(pbTemp1, pstPixelGmm, pbTemp5/*, pbTemp2, pbTemp3*/);

						//���¸���˹ģ��
						if (nMatchFlag == -1) // ���û��ƥ���������µĸ�˹����
						{
							m_fAlphaW = fSlowWeightLearnRate;
							m_fAlpha1 = 1 - m_fAlphaW;
							m_fCT	  = 0.05f * m_fAlphaW;

							UpdataNoMatchPixelGmm(pbTemp1, pstPixelGmm);

							//����Ȩ�ض�ģ�ͽ�������
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

							//����Ȩ�ض�ģ�ͽ�������
							SortedByKey(pstPixelGmm->pGMM, nMatchFlag, pstPixelGmm->pGMM[nMatchFlag].fWeight);
						}
					}
					else
					{
						*pbTemp5 = BACKGROUND;
					}
	
					pbTemp1++;
					pbTemp2++;
					pbTemp5++;
					pstPixelGmm++;
				}
			}

			m_dForDriftTime = dDriftTime;
		}
		else
		{
			BYTE*		 pbTemp1	   = (BYTE*)pbImage;
			BYTE*		 pbTemp5	   = m_pbGmmBw;
			ST_PixelGMM* pstPixelGmm   = m_pstPixelGmm;
			if (pbMask == NULL)
			{
				for (int i = 0; i < m_nSize; i++)
				{
					//�ж��Ƿ���ƥ��ĸ�˹�ֲ�
					//ǰ�����ж�
					MatchForeJudge_Skip(pbTemp1, pstPixelGmm, pbTemp5/*, pbTemp2, pbTemp3*/);

					pbTemp1++;
					pbTemp5++;
					pstPixelGmm++;
				}
			}
			else
			{
				BYTE* pbTemp2 = (BYTE*)pbMask;
				for (int i = 0; i < m_nSize; i++)
				{
					if (*pbTemp2 != 0)
					{
						//�ж��Ƿ���ƥ��ĸ�˹�ֲ�
						//ǰ�����ж�
						MatchForeJudge_Skip(pbTemp1, pstPixelGmm, pbTemp5/*, pbTemp2, pbTemp3*/);
					}
					else
					{
						*pbTemp5 = BACKGROUND;
					}

					pbTemp1++;
					pbTemp2++;
					pbTemp5++;
					pstPixelGmm++;
				}
			}
		}
	}

	return 1;
}

void xhGrayGMM::InitialPixelGmm(const BYTE* pbImage)
{
	BYTE*        pbTemp1     = (BYTE*)pbImage;
	ST_PixelGMM* pstPixelGmm = m_pstPixelGmm;
	for (int i = 0; i < m_nSize; i++)
	{
		//����ģ�͸�Ȩ�ء���ֵ�ͷ���
		pstPixelGmm->nGMMUsedNum	= 1;
		pstPixelGmm->fVar			= InitialVar;

		ST_PixelPerGMM* pGMM		= pstPixelGmm->pGMM;
		pGMM[0].fWeight				= 1.0f / SlowWLearnRate;
		pGMM[0].frgbValue			= pbTemp1[0];

		pbTemp1++;
		pstPixelGmm++;
	}
}

int xhGrayGMM::MatchForeJudge(const BYTE* pbImage, ST_PixelGMM* pstPixelGmm, BYTE* pbBw/*, BYTE* pbBackImg, BYTE* pbDist*/)
{
	//��һ��
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

	//
	float			fDiffV, fDist;
	float           fThres1    = m_fVarRatio1 * pstPixelGmm->fVar;
	float           fThres2    = /*max(*/m_fVarRatio2 * pstPixelGmm->fVar/*, 36)*/;
	int				nMatchFlag1= -1;
	int				nMatchFlag2= -1;
	for (i = 0; i < pstPixelGmm->nGMMUsedNum; i++)
	{
		//����3������ԭ���ж��Ƿ�ƥ��
		fDiffV = pGMM[i].frgbValue - pbImage[0];
		fDist  = fDiffV * fDiffV;
		if (nMatchFlag2 < 0 && fDist < fThres2)
		{
			nMatchFlag2 = i;
		}
		if (fDist < fThres1)
		{
			nMatchFlag1 = i;

			//����ƥ��ĸ�˹ģ�͵ľ�ֵ�ͷ���
			pGMM[i].fWeight	   = m_fAlpha1 * pGMM[i].fWeight - m_fCT + m_fAlphaW;
			pGMM[i].frgbValue -= m_fAlphaM * fDiffV;
			pstPixelGmm->fVar -= 0.01f/*fLearnRate*/ * (pstPixelGmm->fVar - fDist);
			if (pstPixelGmm->fVar < MinVar)  //��ֹ�����С�����
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

	//
	if (nMatchFlag2 == 0)
	{
		*pbBw		 = BACKGROUND; //���е�һ����˹ģ�ͣ��϶��Ǳ�����
	}
	else if (nMatchFlag2 == -1)
	{
		*pbBw		 = FOREGROUND; //û��ƥ���˹ģ�ͣ��϶���ǰ����
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
			*pbBw		 = BACKGROUND; //���ڻ��и�˹ģ��ǰ�ļ���ģ���ۼ�Ȩ�غ�С��Thres����Ϊ������
		}
		else
		{
			*pbBw		 = FOREGROUND; //���ڻ��и�˹ģ��ǰ�ļ���ģ���ۼ�Ȩ�غʹ���Thres����Ϊǰ����
		}
	}

	return nMatchFlag1;
}

void xhGrayGMM::UpdataNoMatchPixelGmm(const BYTE* pbImage, ST_PixelGMM* pstPixelGmm)
{
	ST_PixelPerGMM* pGMM = pstPixelGmm->pGMM;

	//���¸���ģ�͵�Ȩ��ֵ
	int nGMMUsedNum = pstPixelGmm->nGMMUsedNum;
	for (int i = 0; i < nGMMUsedNum; i++)
	{
		pGMM[i].fWeight = m_fAlpha1 * pGMM[i].fWeight - m_fCT;
		if (pGMM[i].fWeight < m_fCT)
		{
			//��ǰλ��Ȩ������
			pGMM[i].fWeight = 0.0f;

			//ģ������һ
			pstPixelGmm->nGMMUsedNum--;
		}
	}

	//�����ģ�ͱ�ɾ�������������ģ�͸��ǵ���ɾ����ģ����λ���ǰ��ģ��
	//���û��ģ�ͱ�ɾ������ģ����û�дﵽ��󣬼������ģ�ͣ���������ģ�����Ȩ����С��ģ��
	if (pstPixelGmm->nGMMUsedNum < GmmMaxNum)
	{
		pstPixelGmm->nGMMUsedNum++;
	}

	//����ģ�͸�Ȩ�ء���ֵ�ͷ���
	int nLastModeIndex = pstPixelGmm->nGMMUsedNum - 1;
	pGMM[nLastModeIndex].fWeight   = m_fAlphaW;
	pGMM[nLastModeIndex].frgbValue = pbImage[0];
}

void xhGrayGMM::UpdataMatchPixelGmm(ST_PixelGMM* pstPixelGmm, int nMatchFlag)
{
	ST_PixelPerGMM* pGMM = pstPixelGmm->pGMM;

	//���¸�˹ģ�͵�Ȩ��
	int nGMMUsedNum = pstPixelGmm->nGMMUsedNum;
	for (int i = 0; i < nGMMUsedNum; i++)
	{
		if (i != nMatchFlag)
		{
			//�Բ�ƥ���ģ��Ȩ�ؽ��и���
			pGMM[i].fWeight = m_fAlpha1 * pGMM[i].fWeight - m_fCT;

			//��Ȩ��С��0ʱ��ɾ����ģ��
			if (pGMM[i].fWeight < m_fCT)
			{
				//��ǰλ��Ȩ������
				pGMM[i].fWeight = 0.0f;

				//ģ������һ
				pstPixelGmm->nGMMUsedNum--;
			}
		}
	}
}


void xhGrayGMM::SortedByKey(ST_PixelPerGMM* pGMM, int nMatchIndex, float fMatchWeight)
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

int xhGrayGMM::GetGmmBw(BYTE** pbBw)
{
	*pbBw = m_pbGmmBw;
	return 1;
}

int xhGrayGMM::GetDist(BYTE** pbDist)
{
	*pbDist = m_pbDist;
	return 1;
}

int xhGrayGMM::GetBackImg(BYTE** pbBackImg)
{
	*pbBackImg = m_pbBackImg;
	return 1;
}