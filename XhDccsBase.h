#pragma once

// 每个函数前的字母A~D，表示优先程度， A最优先，D最不优先
#define  R_COEF 306  // 0.299 
#define  G_COEF 601  // 0.587
#define  B_COEF 117  // 0.114

#define  DIR_HORIZONTAL  1
#define  DIR_VERTICAL    2

#define MEMO_FREE_AND_NULL_N(p)     if (p){delete [] p; p = NULL;}
#define MEMO_FREE_AND_NULL(p)       if (p){delete p; p = NULL;}

enum COLORTYPE{COLOR_INVALID = 0, COLOR_EXIST = 1, COLOR_BLUE = 2, COLOR_BLACK = 3, COLOR_YELLOW = 4, COLOR_WHITE = 5};

class XhDccsBase
{
public:
    XhDccsBase(void);
    ~XhDccsBase(void);

    // A
    // 彩色图转灰度图
    //    pbyImg - 输入的BGR图像
    //    szImg  - 输入图像尺寸
    //    pbyGray  - 输出的灰度图像
    //    nBpp   - 输入图像位深度，必须为24    
    //    nType = 0 时采用移位方式近似，nType = 1时采用浮点计算
    static void ImgRGB2Gray(const BYTE* pbyImg, 
                            const SIZE& szImg, 
                            BYTE* pbyGray, 
                            int nBpp = 24, 
                            int nType = 0);


    // A
    // 分离BGR图像中GR通道
    //    pbyBGR   - 输入的BGR图像
    //    szImg    - 输入图像尺寸
    //    pbyGreen - 分离出的G通道图像，尺寸与源图像相同，但每个像素只有一个像素值，类似于灰度图
    //    pbyRed   - 分离出的R通道图像，尺寸与源图像相同，但每个像素只有一个像素值，类似于灰度图
    static void SeparateGR(const BYTE* pbyBGR, 
                           const SIZE& szImg, 
                           BYTE* pbyGreen, 
                           BYTE* pbyRed);

    // A
    // 灰度图转二值图
    //    pbyGray    - 输入的灰度图像
    //    szGray     - 输入图像尺寸
    //    pbyBinary  - 输出的二值图像
    //    byThresh    - 二值化阈值 
    static void ImgGray2Binary(const BYTE* pbyGray, 
                               const SIZE& szGray, 
                               BYTE byThresh, 
                               BYTE* pbyBinary);

    // B
    // 提取指定区域图像
    //    pbySrc - 输入图像  
    //    szSrc  - 输入图像尺寸
    //    nBpp   - 输入图像位深度
    //    rtDst  - 提取区域的矩形框位置
    //    pbyDst - 输出的提取区域图像
    static void GetSelectedArea(const BYTE* pbySrc, 
                                const SIZE& szSrc, 
                                int nBpp,
                                const RECT& rtDst, 
                                BYTE* pbyDst);

    // C
    // 最近邻法缩放图像
    //    pbySrc - 输入图像  
    //    szSrc  - 输入图像尺寸
    //    nBpp   - 输入图像位深度
    //    szDst  - 输出图像尺寸
    //    pbyDst - 输出的提取区域图像
    static void RspImgNN(const BYTE* pbySrc, 
                         const SIZE& szSrc, 
                         int nBpp, 
                         const SIZE& szDst, 
                         BYTE* pbyDst);

    // C
    // 双线性插值缩放图像
    //    pbySrc - 输入图像  
    //    szSrc  - 输入图像尺寸
    //    nBpp   - 输入图像位深度
    //    szDst  - 输出图像尺寸
    //    pbyDst - 输出的提取区域图像
    static void RspImgLinear(const BYTE* pbySrc, 
                             const SIZE& szSrc, 
                             int nBpp, 
                             const SIZE& szDst, 
                             BYTE* pbyDst);

    // D
    // 连通域标记-区域标注函数，返回标注得到的区域的个数
    //    pbyImg   - 输入的BGR图像
    //    pnOut    - 输出的标记图像
    //    szImg    - 输入图像尺寸
    //    byIgnore - 不需要标注的像素值，若全部像素值都需要被标注则令为-1（也是默认值），二值图用于标记前景的话byIgnore取0
    //    nType    - 区域标注的邻域类型，必须为4邻域或者8邻域，默认为8.
    static int ImgRegionLabel(const BYTE* pbyImg, 
                              int* pnOut, 
                              const SIZE& szImg, 
                              BYTE byIgnore = -1, 
                              int nType = 8);

    // A
    // 平方型sobel算子梯度
    //    pbyGray    - 输入的灰度图像
    //    szGray     - 输入图像尺寸
    //    nBpp       - 输入图像位深度，必须为8
    //    pbyMask    - 掩膜图像，尺寸与灰度图像相同，某个点为0时，表示该点不处理，梯度图对应点保持为0
    //    pnGrad     - 梯度图像，三邻域对应点求积之和再平方会超过255因此使用int型
    //    dbMeanConv — 平均梯度值
    //    nDir       - 水平或垂直方向，采用不同的sobel算子
    static void SobelPowGrad(const BYTE* pbyGray, 
                             const SIZE& szGray, 
                             int nBpp, 
                             const BYTE* pbyMask, 
                             int* pnGrad, 
                             double& dbMeanConv, 
                             int nDir = DIR_VERTICAL);

    // A
    // 绝对值型sobel算子梯度
    //    pbyGray    - 输入的灰度图像
    //    szGray     - 输入图像尺寸
    //    nBpp       - 输入图像位深度，必须为8
    //    pbyMask    - 掩膜图像，尺寸与灰度图像相同，某个点为0时，表示该点不处理，梯度图对应点保持为0
    //    pnGrad     - 梯度图像，三邻域对应点求积之和再平方会超过255因此使用int型
    //    nDir       - 水平或垂直方向，采用不同的sobel算子
    static void SobelGrad(const BYTE* pbyGray, 
                          const SIZE& szGray, 
                          int nBpp, 
                          const BYTE* pbyMask, 
                          int* pnGrad,
                          int nDir = DIR_VERTICAL);
    
    // A
    // hewitt算子梯度
    //    pbyGray    - 输入的灰度图像
    //    szGray     - 输入图像尺寸
    //    nBpp       - 输入图像位深度，必须为8
    //    pbyMask    - 掩膜图像，尺寸与灰度图像相同，某个点为0时，表示该点不处理，梯度图对应点保持为0
    //    pbyEnhance  - 是否加强梯度值的标记图，该图尺寸与pbyGray相同。 若加强，则将对应点的梯度值加倍
    //    pnGrad     - 梯度图像，三邻域对应点求积之和会超过255因此使用int型
    //    dbMeanConv — 平均梯度值
    //    nDir       - 水平或垂直方向，采用不同的hewitt算子
    static void HewittGrad(const BYTE* pbyGray, 
                           const SIZE& szGray, 
                           int nBpp, 
                           const BYTE* pbyMask, 
                           const BYTE* pbyEnhance, 
                           int* pnGrad, 
                           double& dbMeanConv, 
                           int nDir = DIR_VERTICAL);

    // C
    // 任意3*3归一化算子（算子之和等于1）滤波,
    //    pbySrc    - 输入图像  
    //    szSrc     - 输入图像尺寸
    //    nBpp      - 输入图像位深度
    //    pbyDst    - 输出的提取区域图像, 输出图像尺寸与输入图像相同
    //    pdbKernel - 3*3算子（实际数据为一行九个）
    static void ImgFilter(const BYTE* pbySrc, 
                          const SIZE& szSrc, 
                          int nBpp, 
                          BYTE* pbyDst, 
                          const double* pdbKernel);

    // A
    // 由梯度计算边缘图（采用二值化方法）
    //    pnGrad     - 梯度图像
    //    szGrad     - 梯度图像尺寸
    //    pbyMask    - 掩膜图像，尺寸与梯度图像相同，某个点为0时，表示该点不处理，边缘图对应点保持为0
    //    dbMeanConv — 平均梯度值
    //    dbLevel    - 二值化阈值调节系数
    //    pbyEdge    - 输出的边缘图像
    //    nDir       - 水平或垂直方向
    static void Grad2Edge(const int* pnGrad, 
                          const SIZE& szGrad, 
                          const BYTE* pbyMask,
                          double dbMeanGrad, 
                          double dbLevel, 
                          BYTE* pbyEdge, 
                          int nDir = DIR_VERTICAL);

    // A
    // 蓝黄色区域提取（找出符合条件的像素点，并在水平方向4邻域内膨胀）
    //    pbyImg    - 输入的BGR图像
    //    szImg     - 输入图像尺寸
    //    nBpp      - 输入图像位深度，必须为24  
    //    pbyMask   - 掩膜图像，尺寸与原始图像相同，某个点为0时，表示该点不处理，颜色标记图保持为0
    //    pbyBlue   - 蓝色点标记
    //    pbyYellow - 黄色点标记
    //    pbyFlag   - 蓝色或黄色点颜色标记
    static void FlagPlateColor(const BYTE* pbyImg, 
                               const SIZE& szImg, 
                               int nBpp,
                               const BYTE* pbyMask,
                               BYTE* pbyBlue, 
                               BYTE* pbyYellow, 
                               BYTE* pbyFlag);

    // 实现两段数据的点乘，最后返回乘积之和
    static double MultyDot(double* dbSrcA, double* dbSrcB, int nLen);

    // B
    // 2*3算子图像膨胀（2行3列，边界不处理，膨胀后的图像尺寸不变）
    //    pbyBinary - 输入的二值图像
    //    szBinary  - 输入的二值图像尺寸
    //    nVal      - 膨胀的填充值
    //    pbyDst    - 膨胀后的图像，调用此函数前前分配好内存
    static void DilateFix_On2x3(const BYTE* pbyBinary, 
                                const SIZE& szBinary,
                                int nVal, 
                                BYTE* pbyDst);

    // B
    // 3*3算子图像膨胀（3行3列，边界不处理，膨胀后的图像尺寸不变）
    //    pbyBinary - 输入的二值图像
    //    szBinary  - 输入的二值图像尺寸
    //    nVal      - 膨胀的填充值
    //    pbyDst    - 膨胀后的图像，调用此函数前前分配好内存
    static void DilateFix_On3x3(const BYTE* pbBinary, 
                                const SIZE& szBinary,
                                int nVal, 
                                BYTE* pbDst);

    // B
    // 3*n算子图像膨胀（3行n列，边界处理，膨胀后的图像尺寸增加）
    //    pbyBinary - 输入的二值图像
    //    szBinary  - 输入的二值图像尺寸
    //    nKenelWidth - 算子列数n
    //    nVal      - 膨胀的填充值
    //    szDst     - 膨胀后的图像尺寸，宽等于 szDst.cx + n / 2, 高等于 szDst.cy + 2
    //    pbyDst    - 膨胀后的图像，调用此函数前前分配好内存
    static void DilateVary_On3xn(const BYTE* pbyBinary, 
                                 const SIZE& szBinary,
                                 int nKenelWidth, 
                                 int nVal, 
                                 const SIZE& szDst, 
                                 BYTE* pbyDst);

    // B
    // 5*n算子图像膨胀（5行n列，边界处理，膨胀后的图像尺寸增加）
    //    pbyBinary - 输入的二值图像
    //    szBinary  - 输入的二值图像尺寸
    //    nKenelWidth - 算子列数n
    //    nVal      - 膨胀的填充值
    //    szDst     - 膨胀后的图像尺寸，宽等于 szDst.cx + n / 2, 高等于 szDst.cy + 4
    //    pbyDst    - 膨胀后的图像，调用此函数前前分配好内存
    static void DilateVary_On5xn(const BYTE* pbyBinary, 
                                 const SIZE& szBinary,
                                 int nKenelWidth, 
                                 int nVal, 
                                 const SIZE& szDst, 
                                 BYTE* pbyDst);

    // D
    // 二值图像投影计算
    //    pbyBinary - 输入的二值图像
    //    szBinary  - 输入的二值图像尺寸
    //    nProjLen  - 投影值个数，应该与图像的宽或高相同
    //    pnProj    - 投影序列
    //    nMOde     - 水平或垂直方向
    //    byBackPtVal - 背景值，投影统计时，像素值等于byBackPtVal的不纳入统计
    static void BwProjection(const BYTE* pbyBinary, 
                             const SIZE& szBinary, 
                             int nProjLen,
                             int* pnProj, 
                             int nMode = DIR_VERTICAL, 
                             BYTE byBackPtVal = 0);

    // D
    // 二值图像指定区域内投影
    //    pbyBinary - 输入的二值图像
    //    szBinary  - 输入的二值图像尺寸
    //    rtLoc     - 指定区域的矩形框位置
    //    nProjLen  - 投影值个数，应该与指定区域的宽或高相同
    //    pnProj    - 投影序列
    //    nMOde     - 水平或垂直方向
    //    byBackPtVal - 背景值，投影统计时，像素值等于byBackPtVal的不纳入统计
    static void LocProjection(const BYTE* pbyBinary, 
                              const SIZE& szBinary, 
                              int nProjLen,  
                              const RECT& rtLoc, 
                              int* pnProj, 
                              int nMode= DIR_VERTICAL, 
                              BYTE byBackPtVal = 0);

private:
    // D
    // 四连通域标记-区域标注函数，返回标注得到的区域的个数，
    //    pbyImg   - 输入的BGR图像
    //    pnOut    - 输出的标记图像
    //    szImg    - 输入图像尺寸
    //    byIgnore - 不需要标注的像素值，若全部像素值都需要被标注则令为-1（也是默认值），二值图用于标记前景时 byIgnore取0
    static int ImgRegionLabel_4(const BYTE* pbyImg, 
                                int* pnOut, 
                                const SIZE& szImg, 
                                BYTE byIgnore);

    // D
    // 八连通域标记-区域标注函数，返回标注得到的区域的个数，
    //    pbyImg   - 输入的BGR图像
    //    pnOut    - 输出的标记图像
    //    szImg    - 输入图像尺寸
    //    byIgnore - 不需要标注的像素值，若全部像素值都需要被标注则令为-1（也是默认值），二值图用于标记前景时 byIgnore取0
    static int ImgRegionLabel_8(const BYTE* pbyImg, 
                                int* pnOut, 
                                const SIZE& szImg, 
                                BYTE byIgnore);

    // D
    // 四邻域区域生长
    //    pbyImg          - 输入的BGR图像
    //    szImg           - 输入图像尺寸
    //    nLastCol        - 最后一列的列序号
    //    pbyFlag         - 像素点是否已访问并标记过的标示图，该图尺寸与输入源图像相同
    //    pbyFlagRowStart - 第二行的起始点序号
    //    pbyFlagRowEnd   - 倒数第二行的结束点序号
    //    ppbyStack       - 生长在一起的像素点的指针集合
    //    pbyPxl          - 当前点像素值
    //    nLabelInd       - 当前点的标记号数
    //    pnOut           - 更新标记图，标记图尺寸与输入源图像相同
    static void FillNeighbor_4(const BYTE* pbyImg, 
                               const SIZE& szImg, 
                               int nLastCol,
                               BYTE* pbyFlag, 
                               BYTE* pbyFlagRowStart, 
                               BYTE* pbyFlagRowEnd, 
                               BYTE** ppbyStack, 
                               const BYTE* pbyPxl, 
                               int nLabelInd, 
                               int* pnOut);

    // D
    // 八邻域区域生长
    //    pbyImg          - 输入的BGR图像
    //    szImg           - 输入图像尺寸
    //    nLastCol        - 最后一列的列序号
    //    pbyFlag         - 像素点是否已访问并标记过的标示图，该图尺寸与输入源图像相同
    //    pbyFlagRowStart - 第二行的起始点序号
    //    pbyFlagRowEnd   - 倒数第二行的结束点序号
    //    ppbyStack       - 生长在一起的像素点的指针集合
    //    pbyPxl          - 当前点像素值
    //    nLabelInd       - 当前点的标记号数
    //    pnOut           - 更新标记图，标记图尺寸与输入源图像相同
    static void FillNeighbor_8(const BYTE* pbyImg, 
                               const SIZE& szImg, 
                               int nLastCol,
                               BYTE* pbyFlag, 
                               BYTE* pbyFlagRowStart, 
                               BYTE* pbyFlagRowEnd, 
                               BYTE** ppbyStack, 
                               const BYTE* pbyPxl, 
                               int nLabelInd, 
                               int* pnOut);

    // 判断是否为可能的蓝色点
    //    pi - 单个像素点的起始坐标，作为BGR彩色图，pi, pi + 1, pi + 2应该分别为该像素点的B、G、R通道值
    static bool IsBlue(const BYTE* pi);

    // 判断是否为可能的黄色点
    //    pi - 单个像素点的起始坐标，作为BGR彩色图，pi, pi + 1, pi + 2应该分别为该像素点的B、G、R通道值
    static bool IsYellow(const BYTE* pi);
};

