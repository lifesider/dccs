/************************************************************************/
/*                    canny边缘检测                                     */
/************************************************************************/
/*************************************************************************
*
* \函数名称：
*   Canny()
*
* \输入参数:
*   const BYTE *pbImage      - 灰度图象数据
*   const SIZE& szImg        - 图像的大小
*   double dbSigma           - 高斯滤波的标准方差
*	 double dbRatioLow         - 低阈值和高阈值之间的比例
*	 double dbRatioHigh        - 高阈值占图象象素总数的比例
*   BYTE* pbEdge              - canny算子计算后的边缘图
*
* \返回值:
*   无
*
* \说明:
*   canny分割算子，计算的结果保存在pbEdge中，逻辑1(255)表示该点为
*   边界点，逻辑0(0)表示该点为非边界点。该函数的参数dbSigma，dbRatioLow
*   dbRatioHigh，是需要指定的。这些参数会影响分割后边界点数目的多少
*************************************************************************
*/ 
#include "StdAfx.h"
#include <assert.h>
#include "DccsCanny.h"
#include "dccsbase.h"

static double s_dbFactorRegon64[128] = 
{ -0.00793650793651, 0.00793650793651,
0.00793650793651, 0.02380952380953,
0.02380952380953, 0.03968253968255,
0.03968253968255, 0.05555555555557,
0.05555555555557, 0.07142857142859,
0.07142857142859, 0.08730158730161,
0.08730158730161, 0.10317460317463,
0.10317460317463, 0.11904761904765,
0.11904761904765, 0.13492063492067,
0.13492063492067, 0.15079365079369,
0.15079365079369, 0.16666666666671,
0.16666666666671, 0.18253968253973,
0.18253968253973, 0.19841269841275,
0.19841269841275, 0.21428571428577,
0.21428571428577, 0.23015873015879,
0.23015873015879,   0.24603174603181,
0.24603174603181,   0.26190476190483,
0.26190476190483,   0.27777777777785,
0.27777777777785,   0.29365079365087,
0.29365079365087,   0.30952380952389,
0.30952380952389,   0.32539682539691,
0.32539682539691,   0.34126984126993,
0.34126984126993,   0.35714285714295,
0.35714285714295,   0.37301587301597,
0.37301587301597,   0.38888888888899,
0.38888888888899,   0.40476190476201,
0.40476190476201,   0.42063492063503,
0.42063492063503,   0.43650793650805,
0.43650793650805,   0.45238095238107,
0.45238095238107,   0.46825396825409,
0.46825396825409,   0.48412698412711,
0.48412698412711,   0.50000000000013,
0.50000000000013,   0.51587301587315,
0.51587301587315,   0.53174603174617,
0.53174603174617,   0.54761904761919,
0.54761904761919,   0.56349206349221,
0.56349206349221,   0.57936507936523,
0.57936507936523,   0.59523809523825,
0.59523809523825,   0.61111111111127,
0.61111111111127,   0.62698412698429,
0.62698412698429,   0.64285714285731,
0.64285714285731,   0.65873015873033,
0.65873015873033,   0.67460317460335,
0.67460317460335,   0.69047619047637,
0.69047619047637,   0.70634920634939,
0.70634920634939,   0.72222222222241,
0.72222222222241,   0.73809523809543,
0.73809523809543,   0.75396825396845,
0.75396825396845,   0.76984126984147,
0.76984126984147,  0.78571428571449,
0.78571428571449,   0.80158730158751,
0.80158730158751,   0.81746031746053,
0.81746031746053,   0.83333333333355,
0.83333333333355,   0.84920634920657,
0.84920634920657,   0.86507936507959,
0.86507936507959,   0.88095238095261,
0.88095238095261,   0.89682539682563,
0.89682539682563,   0.91269841269865,
0.91269841269865,   0.92857142857167,
0.92857142857167,   0.94444444444469,
0.94444444444469,   0.96031746031771,
0.96031746031771,   0.97619047619073,
0.97619047619073,   0.99206349206375,
0.99206349206375,   1.00793650793677
};

static double    g_pdbKernel2D_X[81] = { 
    1.43284234626241e-007,     3.55869164115247e-006,     2.89024929508973e-005,     6.47659933817797e-005, 0 ,   -6.47659933817797e-005 ,   -2.89024929508973e-005,    -3.55869164115247e-006, -1.43284234626241e-007,
    4.7449221882033e-006,      0.000117847682078385,      0.000957119116801882,       0.00214475514239131,  0 ,     -0.00214475514239131,     -0.000957119116801882,     -0.000117847682078385, -4.7449221882033e-006,
    5.78049859017946e-005,       0.00143567867520282,        0.0116600978601128,        0.0261284665693698, 0 ,      -0.0261284665693698,       -0.0116600978601128,      -0.00143567867520282, -5.78049859017946e-005,
    0.000259063973527119,       0.00643426542717393,        0.0522569331387397,         0.117099663048638, 0 ,       -0.117099663048638 ,      -0.0522569331387397,      -0.00643426542717393,  -0.000259063973527119,
    0.000427124283626255,       0.0106083102711121,        0.0861571172073945,         0.193064705260108, 0  ,      -0.193064705260108 ,      -0.0861571172073945,       -0.0106083102711121, -0.000427124283626255,
    0.000259063973527119,      0.00643426542717393,        0.0522569331387397,         0.117099663048638, 0  ,      -0.117099663048638,       -0.0522569331387397,      -0.00643426542717393, -0.000259063973527119,
    5.78049859017946e-005,       0.00143567867520282,        0.0116600978601128,        0.0261284665693698, 0 ,      -0.0261284665693698 ,      -0.0116600978601128,      -0.00143567867520282, -5.78049859017946e-005,
    4.7449221882033e-006,      0.000117847682078385,      0.000957119116801882,       0.00214475514239131, 0  ,    -0.00214475514239131,     -0.000957119116801882,     -0.000117847682078385, -4.7449221882033e-006,
    1.43284234626241e-007,     3.55869164115247e-006,     2.89024929508973e-005,     6.47659933817797e-005, 0    -6.47659933817797e-005,    -2.89024929508973e-005,    -3.55869164115247e-006, -1.43284234626241e-007
};

static SIZE      g_szKernel2D = {9, 9};

static double   g_pdbKernel2D_Y[81] = {
    1.43284234626241e-007,      4.7449221882033e-006,     5.78049859017946e-005,      0.000259063973527119,0.000427124283626255,      0.000259063973527119,     5.78049859017946e-005,      4.7449221882033e-006,1.43284234626241e-007,
    3.55869164115247e-006,      0.000117847682078385,       0.00143567867520282,       0.00643426542717393,0.0106083102711121,       0.00643426542717393,       0.00143567867520282 ,     0.000117847682078385,3.55869164115247e-006,
    2.89024929508973e-005,      0.000957119116801882,        0.0116600978601128,        0.0522569331387397,0.0861571172073945,        0.0522569331387397,        0.0116600978601128,      0.000957119116801882,2.89024929508973e-005,
    6.47659933817797e-005,       0.00214475514239131,        0.0261284665693698,         0.117099663048638,0.193064705260108,         0.117099663048638 ,       0.0261284665693698,       0.00214475514239131,6.47659933817797e-005,
    0,                          0,                           0 ,                        0,                 0,                         0     ,                    0             ,            0,                    0,
    -6.47659933817797e-005,      -0.00214475514239131,       -0.0261284665693698,        -0.117099663048638,-0.193064705260108,        -0.117099663048638,       -0.0261284665693698,      -0.00214475514239131,-6.47659933817797e-005,
    -2.89024929508973e-005,     -0.000957119116801882,       -0.0116600978601128,       -0.0522569331387397, -0.0861571172073945,       -0.0522569331387397,       -0.0116600978601128,     -0.000957119116801882,-2.89024929508973e-005,
    -3.55869164115247e-006,     -0.000117847682078385,      -0.00143567867520282,      -0.00643426542717393,-0.0106083102711121,      -0.00643426542717393,      -0.00143567867520282,     -0.000117847682078385,-3.55869164115247e-006,
    -1.43284234626241e-007,     -4.7449221882033e-006,    -5.78049859017946e-005,     -0.000259063973527119,-0.000427124283626255,     -0.000259063973527119,    -5.78049859017946e-005 ,    -4.7449221882033e-006,-1.43284234626241e-007
};

static double   g_pdbKernel1D[9] = {5.33905354532819e-005, 0.00176805171185202, 
    0.0215392793018486, 0.0965323526300539, 0.159154943091895, 
    0.0965323526300539, 0.0215392793018486, 0.00176805171185202, 
    5.33905354532819e-005};

#define MATRIX_REF(PR, NUMROWS, R, C)  (*((PR) + (NUMROWS)*(C) + (R)))
static int nWeights3[3][3] = { {1, 8, 64}, {2, 16, 128}, {4, 32, 256} };

static BYTE LUTArray1[512] = 
{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,		
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1		
,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,		
1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1		
,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,		
1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1		
,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,		
1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1		
,1,1,1,1,1,1,1,1,1,1,1,1};

//基于索引表的细化表2
static BYTE LUTArray2[512] = 
{  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1};

DccsCanny::DccsCanny(void)
{
}

DccsCanny::~DccsCanny(void)
{
}

void DccsCanny::GetCannyEdge(const BYTE* pbImage, const SIZE& szImg, 
                             BYTE* pbEdge, double dbSigma, 
                             double dbRatioLow, double dbRatioHigh)
{
    // 转化为double 型数据值在[0,1]之间	
    int       nLen      = szImg.cx * szImg.cy;
    double*   pdbImg    = new double [nLen]; 

    BYTE2double(pbImage, szImg, pdbImg);

    // 对原图象进行滤波,高斯滤波相当于图像与一维高斯函数的卷积
    double*   pdbFilterImg = new double [nLen];
    GaussianSmooth(pdbImg, szImg, pdbFilterImg);


    // 计算方向导数，用高斯加权零交叉微分算子进行计算
    double*   pdbGradX    = new double [nLen];
    double*   pdbGradY    = new double [nLen];
    GaussFilter2DReplicate(pdbFilterImg, szImg, g_pdbKernel2D_X, g_szKernel2D, pdbGradX);
    GaussFilter2DReplicate(pdbFilterImg, szImg, g_pdbKernel2D_Y, g_szKernel2D, pdbGradY);

    // 计算梯度的幅度
    double*   pdbGradMag  = new double [nLen];
    GradMagnitude(pdbGradX, pdbGradY, szImg, pdbGradMag);
    //cv::Mat matGradMag(szImg.cy, szImg.cx, CV_64FC1, pdbGradMag);
    //cv::Mat showimg;
    //cv::convertScaleAbs(matGradMag, showimg);
    //cv::imshow("our", matGradMag);
    //cv::waitKey(0);

    // 估计TraceEdge需要的低阈值，以及Hysteresis函数使用的高阈值
    double    dbThdHigh;
    double    dbThdLow;
    EstimateThreshold(pdbGradMag, szImg, dbRatioHigh, dbRatioLow, dbThdHigh, dbThdLow);

    //// 应用non-maximum 抑制
    BYTE*     pbEdgeTmp = new BYTE[nLen];
    NonmaxSuppress(pdbGradMag, pdbGradX, pdbGradY, szImg, pbEdgeTmp);

    // 应用Hysteresis，找到所有的边界
    Hysteresis(pdbGradMag, szImg, dbThdHigh, dbThdLow, pbEdgeTmp);


    Thinner(pbEdgeTmp, szImg, pbEdge, 1);

    // 释放内存
    MEMO_FREE_AND_NULL_N(pdbGradX);
    MEMO_FREE_AND_NULL_N(pdbGradY);
    MEMO_FREE_AND_NULL_N(pdbGradMag);
    MEMO_FREE_AND_NULL_N(pdbImg);
    MEMO_FREE_AND_NULL_N(pdbFilterImg);
    MEMO_FREE_AND_NULL_N(pbEdgeTmp);
}

// 将图像从byte型转化为double型
void DccsCanny::BYTE2double(const BYTE* pbImg, const SIZE& szImg, double* pdbImg)
{
    int         i, nLen = szImg.cx * szImg.cy;

#ifdef SSE_OPTIMIZE
	ucharnorm2double_sse2(pdbImg, pbImg, nLen);
	return;
#endif
    for (i = 0; i < nLen; i++)
    {
        *pdbImg++ = 1.0 * (*pbImg++) / 255;
    }
}

// 高斯滤波
void DccsCanny::GaussianSmooth(const double* pdbImg, const SIZE& szImg, 
    double* pdbSmthdImg)
{
    // 一维高斯数据滤波器
    int      nWindowHalf = 4;

    // X方向滤波
    double*   pdbFilterX  = new double [szImg.cx * szImg.cy];

	GaussFilter1D_X(pdbImg, szImg, g_pdbKernel1D, nWindowHalf, pdbFilterX);
	
    //Y方向滤波
    GaussFilter1D_Y(pdbFilterX, szImg, g_pdbKernel1D, nWindowHalf, pdbSmthdImg);

    MEMO_FREE_AND_NULL_N(pdbFilterX);
}

// X方向上的高斯一维滤波
void DccsCanny::GaussFilter1D_X(const double* pdbImage, const SIZE& szImg, 
                                const double* pdbFiter, const int& nHalfWinLen,
                                double* pdbFilterImg)
{
    int      i, j, k;
    double   dbSum = 0;
    int nPos = 0;
    int nPosy = 0;
    int nPosK = 0;

#ifdef SSE_OPTIMIZE
	__m128d xmm0 = _mm_loadu_pd(pdbFiter);
	__m128d xmm1 = _mm_loadu_pd(pdbFiter + 2);
	__m128d xmm2 = _mm_loadu_pd(pdbFiter + 4);
	__m128d xmm3 = _mm_loadu_pd(pdbFiter + 6);
#endif

    for (i = 0; i < szImg.cy; i++)
    {
        for(j = 0; j < szImg.cx; j++)
        {
            dbSum = 0;
            nPosy = i * szImg.cx;
            nPos = nPosy + j;

            if(j - nHalfWinLen < 0) //图像的左边界
            {				
                for (k = -nHalfWinLen; k <= nHalfWinLen; k++)
                {
                    if (j + k < szImg.cx)
                    {
                        nPosK = k + nHalfWinLen;
                        if (j + k < 0)
                        {
                            dbSum += pdbImage[nPosy] * pdbFiter[nPosK];
                        }
                        else
                        {
                            dbSum += pdbImage[nPosy + (j + k)] * pdbFiter[nPosK];
                        }
                    }
                }
                pdbFilterImg[nPos] = dbSum;
            }
            else if (j + nHalfWinLen > szImg.cx - 1)// 图像的右边界
            {
                for (k = -nHalfWinLen; k <= nHalfWinLen; k++ )
                {
                    if (j + k > 0)
                    {
                        nPosK = k + nHalfWinLen;
                        if (j + k > szImg.cx - 1)
                        {
                            dbSum += pdbImage[nPosy + szImg.cx - 1] * pdbFiter[nPosK];
                        }
                        else
                        {
                            dbSum += pdbImage[nPosy + (j + k)] * pdbFiter[nPosK];
                        }
                    }
                }

                pdbFilterImg[nPos] = dbSum;				
            }
            else
            {
#ifdef SSE_OPTIMIZE
				__m128d xmm4 = _mm_setzero_pd();
				xmm4 = _mm_add_pd(xmm4, _mm_mul_pd(_mm_loadu_pd(pdbImage + nPosy + j - 4), xmm0));
				xmm4 = _mm_add_pd(xmm4, _mm_mul_pd(_mm_loadu_pd(pdbImage + nPosy + j - 2), xmm1));
				xmm4 = _mm_add_pd(xmm4, _mm_mul_pd(_mm_loadu_pd(pdbImage + nPosy + j), xmm2));
				xmm4 = _mm_add_pd(xmm4, _mm_mul_pd(_mm_loadu_pd(pdbImage + nPosy + j + 2), xmm3));
				xmm4 = _mm_add_sd(xmm4, _mm_mul_sd(_mm_load_sd(pdbImage + nPosy + j + 4), _mm_load_sd(pdbFiter + 8)));
				_mm_store_sd(pdbFilterImg + nPos, _mm_add_sd(xmm4, _mm_shuffle_pd(xmm4, xmm4, 1)));
#else
                dbSum += pdbImage[nPosy + (j - 4)] * pdbFiter[0];	
                dbSum += pdbImage[nPosy + (j - 3)] * pdbFiter[1];	
                dbSum += pdbImage[nPosy + (j - 2)] * pdbFiter[2];	
                dbSum += pdbImage[nPosy + (j - 1)] * pdbFiter[3];	
                dbSum += pdbImage[nPosy + (j + 0)] * pdbFiter[4];	
                dbSum += pdbImage[nPosy + (j + 1)] * pdbFiter[5];	
                dbSum += pdbImage[nPosy + (j + 2)] * pdbFiter[6];	
                dbSum += pdbImage[nPosy + (j + 3)] * pdbFiter[7];	
                dbSum += pdbImage[nPosy + (j + 4)] * pdbFiter[8];	
                pdbFilterImg[nPos] = dbSum;
#endif
            }
        }
    }
}

// Y方向上的高斯一维滤波
void DccsCanny::GaussFilter1D_Y(const double* pdbImage, const SIZE& szImg, 
                               const double* pdbFiter, const int& nHalfWinLen,
                               double* pdbFilterImg)
{
#ifdef SSE_OPTIMIZE
	nsp_filter(pdbFilterImg, pdbImage, pdbFiter, nHalfWinLen*2+1, 1, szImg.cx, szImg.cy);
	return;
#endif
    int      i, j, k;
    double   dbSum = 0;
    int nPos = 0;
    int nPosy = 0;
    int nPosK = 0;

    for (j = 0; j < szImg.cx; j++)
    {
        for (i = 0; i < szImg.cy; i++)
        {
            dbSum = 0;
            nPos = i * szImg.cx + j;

            if (i - nHalfWinLen < 0) //图像的上边界
            {
                for (k = -nHalfWinLen; k <= nHalfWinLen; k++)
                {
                    if (i + k < szImg.cy)
                    {
                        if ((i + k) < 0)
                        {
                            dbSum += pdbImage[j] * pdbFiter[k + nHalfWinLen];
                        }
                        else
                        {
                            dbSum += pdbImage[nPos + k * szImg.cx ] *
                                pdbFiter[k + nHalfWinLen];
                        }
                    }
                }

                pdbFilterImg[nPos] = dbSum;
            }
            else if (i + nHalfWinLen > szImg.cy - 1) //图像的下边界
            {
                for (k = -nHalfWinLen; k <= nHalfWinLen; k++)
                {
                    if (i + k >= 0)
                    {
                        if ((i + k) > szImg.cy - 1)
                        {
                            dbSum += pdbImage[(szImg.cy - 1) * szImg.cx + j] *
                                pdbFiter[k + nHalfWinLen];
                        }
                        else
                        {
                            dbSum += pdbImage[nPos + k * szImg.cx] * 
                                pdbFiter[k + nHalfWinLen];
                        }
                    }
                }

                pdbFilterImg[nPos] = dbSum;
            }
            else
            {
                dbSum += pdbImage[nPos - 4 * szImg.cx] * pdbFiter[0];	
                dbSum += pdbImage[nPos - 3 * szImg.cx] * pdbFiter[1];	
                dbSum += pdbImage[nPos - 2 * szImg.cx] * pdbFiter[2];	
                dbSum += pdbImage[nPos - szImg.cx] * pdbFiter[3];	
                dbSum += pdbImage[nPos] * pdbFiter[4];	
                dbSum += pdbImage[nPos + szImg.cx] * pdbFiter[5];	
                dbSum += pdbImage[nPos + 2 * szImg.cx] * pdbFiter[6];	
                dbSum += pdbImage[nPos + 3 * szImg.cx] * pdbFiter[7];	
                dbSum += pdbImage[nPos + 4 * szImg.cx] * pdbFiter[8];	

                pdbFilterImg[nPos] = dbSum;
            }
        }
    }
}

//高斯二维加权微分
void DccsCanny::GaussFilter2DReplicate(const double* pdbImage, const SIZE& szImg, 
                                       double* pdbFiter, const SIZE& szFilter,
                                       double* pdbFilterImg)
{  

    // 扩大图象
    SIZE       szBig  = {szImg.cx + szFilter.cx / 2 * 2, szImg.cy + szFilter.cy / 2 * 2};
    double*    pdbBig = new double [szBig.cx * szBig.cy];
    AddImgBoundary(pdbImage, szImg, szFilter, szBig, pdbBig);

    // 填入值_top
    int        i, j;
    double*    pi = pdbBig;
    for (i = 0; i < 4; i++)
    {
#ifdef SSE_OPTIMIZE
		memcpy(pi + 4, pdbImage, szImg.cx * sizeof(double));
		pi += szBig.cx;
#else
       for (j = 0; j < szImg.cx; j++)
        {
            *(pdbBig + i * szBig.cx + j + 4) = pdbImage[j];
        }
#endif
    }

    // 填入值_bot
    for (i = szBig.cy - 4; i < szBig.cy; i++)
    {
#ifdef SSE_OPTIMIZE
		memcpy(pdbBig + i * szBig.cx + 4, pdbImage + szImg.cx * (szImg.cy - 1), szImg.cx * sizeof(double));
#else
       for (j = 0; j < szImg.cx; j++)
        {
            *(pdbBig + i * szBig.cx + j + 4) = pdbImage[szImg.cx * (szImg.cy - 1) + j];
        }
#endif
    }

    // 填入值_left/right
    pi = pdbBig;
    for (j = 4; j < 4 + szImg.cy; j++)
    {
#ifdef SSE_OPTIMIZE
		__m128d xmm0 = _mm_load_sd(pdbImage + (j-4)*szImg.cx);
		xmm0 = _mm_shuffle_pd(xmm0, xmm0, 0);
		_mm_storeu_pd(pi + j*szBig.cx, xmm0);
		_mm_storeu_pd(pi + j*szBig.cx + 2, xmm0);
		xmm0 = _mm_load_sd(pdbImage + (j-4)*szImg.cx + szImg.cx-1);
		xmm0 = _mm_shuffle_pd(xmm0, xmm0, 0);
		_mm_storeu_pd(pi + j*szBig.cx + szBig.cx - 4, xmm0);
		_mm_storeu_pd(pi + j*szBig.cx + szBig.cx - 2, xmm0);
#else
        for (i = 0; i < 4; i++)
        {
            *(pi + j * szBig.cx + i) = *(pdbImage + (j - 4) * szImg.cx);
        }

        for (i = szBig.cx - 4; i < szBig.cx; i++)
        {
            *(pi + j * szBig.cx + i) = *(pdbImage + (j - 4) * szImg.cx + szImg.cx - 1);
        }
#endif
    }

    for (j = 0; j < 4; j++)
    {
#ifdef SSE_OPTIMIZE
		__m128d xmm0 = _mm_load_sd(pdbImage);
		xmm0 = _mm_shuffle_pd(xmm0, xmm0, 0);
		_mm_storeu_pd(pi + j*szBig.cx, xmm0);
		_mm_storeu_pd(pi + j*szBig.cx + 2, xmm0);
		xmm0 = _mm_load_sd(pdbImage + szImg.cx - 1);
		xmm0 = _mm_shuffle_pd(xmm0, xmm0, 0);
		_mm_storeu_pd(pi + j*szBig.cx + szBig.cx - 4, xmm0);
		_mm_storeu_pd(pi + j*szBig.cx + szBig.cx - 2, xmm0);
#else
        for (i = 0; i < 4; i++)
        {
            *(pi + j * szBig.cx + i) = pdbImage[0];
        }

        for (i = szBig.cx - 4; i < szBig.cx; i++)
        {
            *(pi + j * szBig.cx + i) = pdbImage[szImg.cx - 1];
        }
#endif
    }

    for (j = szBig.cy - 4; j < szBig.cy; j++)
    {
#ifdef SSE_OPTIMIZE
		__m128d xmm0 = _mm_load_sd(pdbImage + szImg.cx * (szImg.cy - 1));
		xmm0 = _mm_shuffle_pd(xmm0, xmm0, 0);
		_mm_storeu_pd(pi + j*szBig.cx, xmm0);
		_mm_storeu_pd(pi + j*szBig.cx + 2, xmm0);
		xmm0 = _mm_load_sd(pdbImage + szImg.cx * (szImg.cy - 1));
		xmm0 = _mm_shuffle_pd(xmm0, xmm0, 0);
		_mm_storeu_pd(pi + j*szBig.cx + szBig.cx - 4, xmm0);
		_mm_storeu_pd(pi + j*szBig.cx + szBig.cx - 2, xmm0);
#else
       for (i = 0; i < 4; i++)
        {
            *(pi + j * szBig.cx + i) = pdbImage[szImg.cx * (szImg.cy - 1)];
        }

        for (i = szBig.cx - 4; i < szBig.cx; i++)
        {
            *(pi + j * szBig.cx + i) = pdbImage[szImg.cx * szImg.cy - 1];
        }
#endif
    }

    // 滤波
    double*    pdbOutBig = new double [szBig.cx * szBig.cy];
    ImageErodeOperate(pdbBig, szBig, pdbFiter, szFilter, pdbOutBig);

    // 恢复原始尺寸
    SubImgBoundary(pdbOutBig, szBig, szFilter, szImg, pdbFilterImg);

    MEMO_FREE_AND_NULL_N(pdbBig);
    MEMO_FREE_AND_NULL_N(pdbOutBig);
}

// 计算梯度的幅度
void DccsCanny::GradMagnitude(const double* pdbGradX, const double* pdbGradY, 
                              const SIZE& szGrad, double* pdbMag)
{
#ifdef SSE_OPTIMIZE
	nsp_calc_norm_magnitude_d(pdbMag, pdbGradX, pdbGradY, szGrad.cx * szGrad.cy);
	return;
#endif
    int             i,  nLen = szGrad.cx * szGrad.cy;
    const double*   px = pdbGradX;
    const double*   py = pdbGradY;
    double*         pr = pdbMag;
    double          dbMaxMag = -1;

    for(i = 0; i < nLen; i++, pr++, px++, py++)
    {
        *pr = sqrt((*px) * (*px) + (*py) * (*py));

        if (*pr > dbMaxMag)
        {
            dbMaxMag = *pr;
        }
    }

    // 归一化
    if(dbMaxMag > 0)
    {
        pr = pdbMag;
        for(i = 0; i < nLen; i++)
        {
            *pr++ /= dbMaxMag;
        }
    }
}

// 估计TraceEdge需要的低阈值，以及Hysteresis函数使用的高阈值 
void DccsCanny::EstimateThreshold(const double* pdbMag, const SIZE& szMag,
                                  double dbRatioHigh, double dbRationLow, 
                                  double& dbHighThd, double& dbLowThd)
{ 
    //统计Mag的直方图，共分为64级，统计方式见matlab的imhist函数	
    int           pnHist[64] = {0};
    double        dbFactor   = 1.0 / 64;
    int           i, k;
    const double* p = pdbMag;
    for (i = 0; i < szMag.cy * szMag.cx; i++, p++)
    {
        for (k = 0; k <= 63; k++)
        {
            if (*p >= s_dbFactorRegon64[2 * k] && *p < s_dbFactorRegon64[2 * k + 1])
            {
                pnHist[k]++;
            }
        }
    }

    int       pnHistSum[64] = {pnHist[0], 0};
    for (i = 1; i < 64; i++)
    {
        pnHistSum[i] = pnHist[i] + pnHistSum[i - 1];
    }

    dbHighThd = 0;
    double  TH = dbRatioHigh * szMag.cx * szMag.cy;
    for (i = 0; i < 64; i++)
    {
        if (pnHistSum [i] > TH)
        {
            dbHighThd = (i + 1) * dbFactor;
            break;
        }
    }

    dbLowThd = dbRationLow * dbHighThd;
}

// 应用non-maximum 抑制
void DccsCanny::NonmaxSuppress(const double* pdbMag, const double* pdbGradX,
                               const double* pdbGradY, const SIZE& szGrad,
                               BYTE* pbEdge)
{
    // 循环控制变量
    int    y;
    int    x;
    int    nPos;

    // x方向梯度分量
    double dbGx;
    double dbGy;

    // 临时变量
    double dbG1, dbG2, dbG3, dbG4 ;
    double dbWeight;
    double dbTmp1;
    double dbTmp2;
    double dbTmp;
    int    nWidth = szGrad.cx;
    int    nHeight = szGrad.cy;

    // 设置图象边缘部分为不可能的边界点
    for(x = 0; x < nWidth; x++)		
    {
        pbEdge[x] = 0 ;
        pbEdge[(nHeight - 1) * nWidth + x] = 0;
    }
    for(y = 0; y < nHeight; y++)		
    {
        pbEdge[y * nWidth] = 0 ;
        pbEdge[y * nWidth + nWidth - 1] = 0;
    }

    for(y = 1; y < nHeight - 1; y++)
    {
        for(x = 1; x < nWidth- 1; x++)
        {
            nPos = y * nWidth + x;

            // 如果当前象素的梯度幅度为0，则不是边界点
            if(pdbMag[nPos] == 0 )
            {
                pbEdge[nPos] = 0;
            }
            else
            {
                // 当前象素的梯度幅度
                dbTmp = pdbMag[nPos];

                // x，y方向导数
                dbGx = pdbGradX[nPos];
                dbGy = pdbGradY[nPos];

                // 如果方向导数y分量比x分量大，说明导数的方向更加“趋向”于y分量。
                if (abs(dbGy) > abs(dbGx)) 
                {
                    // 计算插值的比例
                    dbWeight = fabs(dbGx) / fabs(dbGy); 

                    dbG2 = pdbMag[nPos - nWidth]; 
                    dbG4 = pdbMag[nPos + nWidth];

                    // 如果x，y两个方向的方向导数的符号相同
                    // C是当前象素，与g1-g4的位置关系为：
                    //	g1 g2 
                    //		 C         
                    //		 g4 g3 
                    if (dbGx * dbGy > 0) 
                    { 					
                        dbG1 = pdbMag[nPos - nWidth - 1];
                        dbG3 = pdbMag[nPos + nWidth + 1];
                    } 

                    // 如果x，y两个方向的方向导数的符号相反
                    // C是当前象素，与g1-g4的位置关系为：
                    //	   g2 g1
                    //		 C         
                    //	g3 g4  
                    else 
                    { 
                        dbG1 = pdbMag[nPos - nWidth + 1];
                        dbG3 = pdbMag[nPos + nWidth - 1];
                    } 
                }

                // 如果方向导数x分量比y分量大，说明导数的方向更加“趋向”于x分量
                // 这个判断语句包含了x分量和y分量相等的情况
                else
                {
                    // 计算插值的比例
                    dbWeight = fabs(dbGy) / fabs(dbGx); 

                    dbG2 = pdbMag[nPos + 1]; 
                    dbG4 = pdbMag[nPos - 1];

                    // 如果x，y两个方向的方向导数的符号相同
                    // C是当前象素，与g1-g4的位置关系为：
                    //	g3   
                    //	g4 C g2       
                    //       g1
                    if (dbGx * dbGy > 0) 
                    {				
                        dbG1 = pdbMag[nPos + nWidth + 1];
                        dbG3 = pdbMag[nPos - nWidth - 1];
                    } 
                    // 如果x，y两个方向的方向导数的符号相反
                    // C是当前象素，与g1-g4的位置关系为：
                    //	     g1
                    //	g4 C g2       
                    //  g3     
                    else 
                    { 
                        dbG1 = pdbMag[nPos - nWidth + 1];
                        dbG3 = pdbMag[nPos + nWidth - 1];
                    }
                }

                // 下面利用g1-g4对梯度进行插值
                {
                    dbTmp1 = dbWeight * dbG1 + (1 - dbWeight) * dbG2;
                    dbTmp2 = dbWeight * dbG3 + (1 - dbWeight) * dbG4;

                    // 当前象素的梯度是局部的最大值
                    // 该点可能是个边界点
                    if(dbTmp >= dbTmp1 && dbTmp >= dbTmp2)
                    {
                        pbEdge[nPos] = 128;              
                    }
                    else
                    {
                        // 不可能是边界点
                        pbEdge[nPos] = 0 ;
                    }
                }
            } 
        } 
    }
} 


// 应用Hysteresis，找到所有的边界
void DccsCanny::Hysteresis(const double* pdbMag, const SIZE& szMag,
                           double dbThdHigh, double dbThdLow,
                           BYTE* pbEdge)
{
    int            x, y;
    BYTE**         ppbTmp  = new BYTE* [szMag.cx * szMag.cy * 8];
    const double*  pi = pdbMag;
    BYTE*          po = pbEdge;

    // 这个循环用来寻找大于nThdHigh的点，这些点被用来当作边界点，然后用
    // TraceEdge函数来跟踪该点对应的边界
    for(y = 0; y < szMag.cy; y++)
    {
        for(x = 0; x < szMag.cx; x++)
        {
            // 如果该象素是可能的边界点，并且梯度大于高阈值，该象素作为
            // 一个边界的起点
            if((*po == 128) && (*pi >= dbThdHigh))
            {
                // 设置该点为边界点
                *po = 255;
                ImageRegionLabelCanny(pdbMag, pbEdge, ppbTmp,
                    y, x, dbThdLow, szMag.cx, szMag.cy);
            }

            po++; 
            pi++;
        }
    }

    // 那些还没有被设置为边界点的象素已经不可能成为边界点
#ifdef SSE_OPTIMIZE
	int count = szMag.cx * szMag.cy;
	po = (BYTE*)(((size_t)pbEdge + 15) & ~15);
	for(int i=0; i<(int)(po-pbEdge); ++i)
		if(pbEdge[i] != 255)
			pbEdge[i] = 0;
	count -= po - pbEdge;
	__m128i xmm1 = _mm_setzero_si128();
	xmm1 = _mm_cmpeq_epi8(xmm1, xmm1);
	for(int i=count-16; i>=0; i -= 16, po += 16)
	{
		__m128i xmm0 = _mm_load_si128((__m128i*)po);
		_mm_store_si128((__m128i*)po, _mm_cmpeq_epi8(xmm0, xmm1));
	}
	if(count & 15)
	{
		for(int i=(count&15); i>0; --i, po++)
			if(*po != 255)
				*po = 0;
	}
#else
   po = pbEdge;
    for(y = 0; y < szMag.cy * szMag.cx; y++, po++)
    {
        if(*po != 255)
        {
            *po = 0;
        }
    }
#endif

    MEMO_FREE_AND_NULL_N(ppbTmp);
}

// 区域标注函数，返回标注得到的区域的个数，
void DccsCanny::ImageRegionLabelCanny(const double* pbImg, BYTE* pbOut, BYTE** ppbTmp,
                                      int nY, int nX, double nLowThd, 
                                      int nWidth, int nHeight)
{
    BYTE*  pbFlgEnd = pbOut + nHeight * nWidth; 
    BYTE** ppbPoint = ppbTmp; 
    int    nPointN  = 0; 
    int    x, n; 
    BYTE   nLbl = 128;

    ppbPoint[nPointN++] = pbOut + nY * nWidth + nX;

    while (nPointN > 0) 
    {
        // pop a point
        BYTE* pb = ppbPoint[--nPointN]; 

        // label it 
        n         = pb - pbOut; // offset from origin  
        pbOut [n] = 255; 

        // check surroundings
        x = n % nWidth; 

        // the original row
        if (pb - 1 >= pbOut && x != 0)
        {
            if (pbOut[n - 1] == nLbl && pbImg[n - 1] >= nLowThd)
            {
                ppbPoint[nPointN++] = pb - 1;
            }
        }

        if (pb + 1 < pbFlgEnd && x != nWidth - 1)
        {
            if (pbOut[n + 1] ==nLbl && pbImg[n + 1] >= nLowThd)
            {
                ppbPoint[nPointN++] = pb + 1;
            }
        }

        // the below row
        if (pb - nWidth >= pbOut)
        {
            if (pbOut[n - nWidth] == nLbl && pbImg[n - nWidth] >= nLowThd)
            {
                ppbPoint[nPointN++] = pb - nWidth;
            }
        }

        // the above row
        if (pb + nWidth < pbFlgEnd)  
        {
            if (pbOut[n + nWidth] == nLbl && pbImg[n + nWidth] >= nLowThd) 
            {
                ppbPoint[nPointN++] = pb + nWidth; 
            }
        }

        // the original row
        if (pb - 1 - nWidth >= pbOut && x != 0 && x != nWidth - 1)
        {
            if (pbOut[n - 1 - nWidth] == nLbl && pbImg[n-1-nWidth] >= nLowThd)
            {
                ppbPoint[nPointN++] = pb - 1 - nWidth;
            }
        }

        if (pb - 1 + nWidth < pbFlgEnd && pb - 1 + nWidth >= pbOut 
            && x != 0 && x != nWidth - 1)
        {
            if (pbOut[n - 1 + nWidth] == nLbl && pbImg[n-1+nWidth] >= nLowThd)
            {
                ppbPoint[nPointN++] = pb - 1 + nWidth;
            }
        }

        // the original row
        if (pb + 1 - nWidth >= pbOut && pb + 1 - nWidth < pbFlgEnd 
            && x != 0 && x != nWidth - 1)
        {
            if (pbOut[n + 1 - nWidth] == nLbl && pbImg[n+1-nWidth] >= nLowThd)
            {
                ppbPoint[nPointN++] = pb + 1 - nWidth;
            }
        }

        if (pb + 1 + nWidth < pbFlgEnd && x != 0 && x != nWidth - 1)
        {
            if (pbOut[n + 1 + nWidth] == nLbl && pbImg[n+1+nWidth] >= nLowThd)
            {
                ppbPoint[nPointN++] = pb + 1 + nWidth;
            }
        }

    }
}

// subtract boundary. help function for erode and dilate
void DccsCanny::SubImgBoundary(const double* pbImage, const SIZE& szImage, 
                               const SIZE& szMask,  const SIZE& szOut, 
                               double* pbOut)
{
    int              j;
    double*          po = pbOut;
    const double*    pi = pbImage + szMask.cy / 2 * szImage.cx + szMask.cx / 2;

    for (j = 0; j < szOut.cy; j++)
    {
        CopyMemory(po, pi, sizeof(*pi) * szOut.cx);
        po += szOut.cx;
        pi += szImage.cx;
    }
}

// erode (fs)
void DccsCanny::ImageErodeOperate(const double* pbImage, const SIZE& szImage, 
                                  const double* pbMask, const SIZE& szMask, 
                                  double* pbOut)
{
    // erode 
    const double*   pi = pbImage;
    double*         po = pbOut + szMask.cy / 2 * szImage.cx + szMask.cx / 2;
    int             i, j;

    for (j = 0; j < szImage.cy - szMask.cy + 1; j++)
    {
        for (i = 0; i < szImage.cx - szMask.cx + 1; i++)
        {
            po[i] = ErodeOperateImp(pi + i, szImage, pbMask, szMask);
        }

        pi += szImage.cx;
        po += szImage.cx;
    }
}


double DccsCanny::ErodeOperateImp(const double* pbStart, const SIZE& szImage, 
                                  const double* pbMask, const SIZE& szMask)
{
    double     bRlt = 0;
    int        i, j;
    double     bMaskVal;

    for (j = 0; j < szMask.cy; j++)
    {
#ifdef SSE_OPTIMIZE
		__m128d xmm0 = _mm_setzero_pd();
		double const* __tmp = pbStart;
		for(i=szMask.cx-2; i>=0; i-=2)
		{
			xmm0 = _mm_add_pd(xmm0, _mm_mul_pd(_mm_loadu_pd(pbMask), _mm_loadu_pd(__tmp)));
			pbMask += 2;
			__tmp += 2;
		}
		if(i+2 > 0)
			xmm0 = _mm_add_sd(xmm0, _mm_mul_sd(_mm_load_sd(pbMask), _mm_load_sd(__tmp))), pbMask++;
		_mm_store_sd(&bRlt, _mm_add_sd(_mm_load_sd(&bRlt), _mm_add_sd(xmm0, _mm_shuffle_pd(xmm0, xmm0, 1))));
#else
        for (i = 0; i < szMask.cx; i++)
        {
            bMaskVal = *pbMask++;
            bRlt    += *(pbStart + i) * bMaskVal;
        }
#endif
        pbStart += szImage.cx;
    }

    return   bRlt;
}

int DccsCanny::Thinner(const BYTE* pbImg, const SIZE& szImg, BYTE* pbDesImg, int nNum)
{
    int   i, j;
    int   nLen      = szImg.cx * szImg.cy;
    int   nBytesLen = nLen * sizeof(BYTE);
    BYTE* pbCImg    = new BYTE [nLen];

    for (i = 0; i < szImg.cy; i++)
    {
        for (j = 0; j < szImg.cx; j++)
        {
            *(pbCImg + j * szImg.cy + i)  = *(pbImg + i * szImg.cx + j);
        }
    }

    BYTE* pbLastImg = new BYTE [nLen];
    BYTE* pbIterImg = new BYTE [nLen];
    int  iter = 1;
    bool done = 0;
    bool equalC = 1;

    while (!done)
    {
        memcpy(pbLastImg, pbCImg, nBytesLen);

        iptMyapplylutc(pbCImg, szImg, pbIterImg, LUTArray1);
        iptMyapplylutc(pbIterImg, szImg, pbCImg, LUTArray2);

        for (i = 0; i < nBytesLen; i++)	
        {
            if ( *(pbLastImg + i) != *(pbCImg + i))	
            {
                equalC = 0;
                break;
            }
        }

        done = ((iter >= nNum) | equalC);
        iter++;
    }

    //返回结果
    for (i = 0; i < szImg.cy; i++)
    {
        for (j = 0; j < szImg.cx; j++)
        {
            *(pbDesImg + i * szImg.cx + j)  = *(pbCImg + j * szImg.cy + i);
        }
    }

    MEMO_FREE_AND_NULL_N(pbCImg);
    MEMO_FREE_AND_NULL_N(pbIterImg);
    MEMO_FREE_AND_NULL_N(pbLastImg);

    return  1;
}

void DccsCanny::iptMyapplylutc(const BYTE*  pbSrcImg, const SIZE& szSrcImg, 
                               BYTE* pbDesImg, BYTE* pbLut)
{
    int  r, c;
    int  tmp = 0;

    for (c = 0; c < szSrcImg.cx; c++) 
    {
        for (r = 0; r < szSrcImg.cy; r++)
        {
            tmp = iptNhood3Offset(pbSrcImg, szSrcImg, r, c);

            MATRIX_REF(pbDesImg, szSrcImg.cy, r, c) = (BYTE)
                (*(pbLut + tmp) == 0 ? 0: 255);
        }
    }
}

int DccsCanny::iptNhood3Offset(const BYTE*  pbSrcImg, const SIZE& szSrcImg,
                              int r, int c)
{
    int minR, maxR, minC, maxC;
    int rr, cc;
    int result = 0;

    if (r == 0) 
    {
        minR = 1;
    } 
    else 
    {
        minR = 0;
    }

    if (r == (szSrcImg.cy - 1)) 
    {
        maxR = 1;
    } 
    else 
    {
        maxR = 2;
    }

    if (c == 0)
    {
        minC = 1;
    }
    else 
    {
        minC = 0;
    }

    if (c == (szSrcImg.cx - 1)) 
    {
        maxC = 1;
    }
    else 
    {
        maxC = 2;
    }

    for (rr = minR; rr <= maxR; rr++) 
    {
        for (cc = minC; cc <= maxC; cc++)
        {
            result += nWeights3[rr][cc] * 
                (MATRIX_REF(pbSrcImg, szSrcImg.cy, r + rr - 1, c + cc - 1) != 0);
        }
    }

    return result;
}

void DccsCanny::AddImgBoundary(const double* pbImage, const SIZE& szImage, 
                               const SIZE& szMask,  const SIZE& szOut, 
                               double* pbOut)
{
    int              j;
    double*          po = pbOut + szMask.cy / 2 * szOut.cx + szMask.cx / 2;
    const double*    pi = pbImage; 

    for (j = 0; j < szImage.cy; j++)
    {
        CopyMemory(po, pi, sizeof(*pi) * szImage.cx);
        po += szOut.cx;
        pi += szImage.cx;
    }
}