#pragma once

#define MEMO_FREE_AND_NULL_N(p)     if (p){delete [] p; p = NULL;}
#define MEMO_FREE_AND_NULL(p)       if (p){delete p; p = NULL;}

class DccsCanny
{
public:
    DccsCanny(void);
    ~DccsCanny(void);

    static void GetCannyEdge(const BYTE* pbImage, const SIZE& szImg, 
        BYTE* pbEdge, double dbSigma, 
        double dbRatioLow, double dbRatioHigh);

private:
    // ��ͼ���byte��ת��Ϊdouble��
    static void BYTE2double(const BYTE* pbImg, const SIZE& szImg, double* pdbImg);

    // ��˹�˲�
    static void GaussianSmooth(const double* pdbImg, const SIZE& szImg, 
        double* pdbSmthdImg);

    // X�����ϵĸ�˹һά�˲�
    static void GaussFilter1D_X(const double* pdbImage, const SIZE& szImg, 
        const double* pdbFiter, const int& nHalfWinLen,
        double* pdbFilterImg);

    // Y�����ϵĸ�˹һά�˲�
    static void GaussFilter1D_Y(const double* pdbImage, const SIZE& szImg, 
        const double* pdbFiter, const int& nHalfWinLen,
        double* pdbFilterImg);

    //��˹��ά��Ȩ΢��
    static void GaussFilter2DReplicate(const double* pdbImage, const SIZE& szImg, 
        double* pdbFiter, const SIZE& szFilter,
        double* pdbFilterImg);

    // �����ݶȵķ���
    static void GradMagnitude(const double* pdbGradX, const double* pdbGradY, 
        const SIZE& szGrad, double* pdbMag);

    // ����TraceEdge��Ҫ�ĵ���ֵ���Լ�Hysteresis����ʹ�õĸ���ֵ 
    static void EstimateThreshold(const double* pdbMag, const SIZE& szMag,
        double dbRatioHigh, double dbRationLow, 
        double& dbHighThd, double& dbLowThd);

    // Ӧ��non-maximum ����
    static void NonmaxSuppress(const double* pdbMag, const double* pdbGradX,
        const double* pdbGradY, const SIZE& szGrad,
        BYTE* pbEdge);

    // Ӧ��Hysteresis���ҵ����еı߽�
    static void Hysteresis(const double* pdbMag, const SIZE& szMag,
        double dbThdHigh, double dbThdLow,
        BYTE* pbEdge);

    // �����ע���������ر�ע�õ�������ĸ�����
    static void ImageRegionLabelCanny(const double* pbImg, BYTE* pbOut, BYTE** ppbTmp,
        int nY, int nX, double nLowThd, 
        int nWidth, int nHeight);

    static void SubImgBoundary(const double* pbImage, const SIZE& szImage, 
        const SIZE& szMask,  const SIZE& szOut, 
        double* pbOut);

    static void ImageErodeOperate(const double* pbImage, const SIZE& szImage, 
        const double* pbMask, const SIZE& szMask, 
        double* pbOut);

    static double ErodeOperateImp(const double* pbStart, const SIZE& szImage, 
        const double* pbMask, const SIZE& szMask);

    static int Thinner(const BYTE* pbImg, const SIZE& szImg, BYTE* pbDesImg, int nNum);

    static void iptMyapplylutc(const BYTE*  pbSrcImg, const SIZE& szSrcImg, 
        BYTE* pbDesImg, BYTE* pbLut);

    static int iptNhood3Offset(const BYTE*  pbSrcImg, const SIZE& szSrcImg,
        int r, int c);

    static void AddImgBoundary(const double* pbImage, const SIZE& szImage, 
        const SIZE& szMask,  const SIZE& szOut, 
        double* pbOut);
};

