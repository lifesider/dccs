#pragma once

// ÿ������ǰ����ĸA~D����ʾ���ȳ̶ȣ� A�����ȣ�D�����
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
    // ��ɫͼת�Ҷ�ͼ
    //    pbyImg - �����BGRͼ��
    //    szImg  - ����ͼ��ߴ�
    //    pbyGray  - ����ĻҶ�ͼ��
    //    nBpp   - ����ͼ��λ��ȣ�����Ϊ24    
    //    nType = 0 ʱ������λ��ʽ���ƣ�nType = 1ʱ���ø������
    static void ImgRGB2Gray(const BYTE* pbyImg, 
                            const SIZE& szImg, 
                            BYTE* pbyGray, 
                            int nBpp = 24, 
                            int nType = 0);


    // A
    // ����BGRͼ����GRͨ��
    //    pbyBGR   - �����BGRͼ��
    //    szImg    - ����ͼ��ߴ�
    //    pbyGreen - �������Gͨ��ͼ�񣬳ߴ���Դͼ����ͬ����ÿ������ֻ��һ������ֵ�������ڻҶ�ͼ
    //    pbyRed   - �������Rͨ��ͼ�񣬳ߴ���Դͼ����ͬ����ÿ������ֻ��һ������ֵ�������ڻҶ�ͼ
    static void SeparateGR(const BYTE* pbyBGR, 
                           const SIZE& szImg, 
                           BYTE* pbyGreen, 
                           BYTE* pbyRed);

    // A
    // �Ҷ�ͼת��ֵͼ
    //    pbyGray    - ����ĻҶ�ͼ��
    //    szGray     - ����ͼ��ߴ�
    //    pbyBinary  - ����Ķ�ֵͼ��
    //    byThresh    - ��ֵ����ֵ 
    static void ImgGray2Binary(const BYTE* pbyGray, 
                               const SIZE& szGray, 
                               BYTE byThresh, 
                               BYTE* pbyBinary);

    // B
    // ��ȡָ������ͼ��
    //    pbySrc - ����ͼ��  
    //    szSrc  - ����ͼ��ߴ�
    //    nBpp   - ����ͼ��λ���
    //    rtDst  - ��ȡ����ľ��ο�λ��
    //    pbyDst - �������ȡ����ͼ��
    static void GetSelectedArea(const BYTE* pbySrc, 
                                const SIZE& szSrc, 
                                int nBpp,
                                const RECT& rtDst, 
                                BYTE* pbyDst);

    // C
    // ����ڷ�����ͼ��
    //    pbySrc - ����ͼ��  
    //    szSrc  - ����ͼ��ߴ�
    //    nBpp   - ����ͼ��λ���
    //    szDst  - ���ͼ��ߴ�
    //    pbyDst - �������ȡ����ͼ��
    static void RspImgNN(const BYTE* pbySrc, 
                         const SIZE& szSrc, 
                         int nBpp, 
                         const SIZE& szDst, 
                         BYTE* pbyDst);

    // C
    // ˫���Բ�ֵ����ͼ��
    //    pbySrc - ����ͼ��  
    //    szSrc  - ����ͼ��ߴ�
    //    nBpp   - ����ͼ��λ���
    //    szDst  - ���ͼ��ߴ�
    //    pbyDst - �������ȡ����ͼ��
    static void RspImgLinear(const BYTE* pbySrc, 
                             const SIZE& szSrc, 
                             int nBpp, 
                             const SIZE& szDst, 
                             BYTE* pbyDst);

    // D
    // ��ͨ����-�����ע���������ر�ע�õ�������ĸ���
    //    pbyImg   - �����BGRͼ��
    //    pnOut    - ����ı��ͼ��
    //    szImg    - ����ͼ��ߴ�
    //    byIgnore - ����Ҫ��ע������ֵ����ȫ������ֵ����Ҫ����ע����Ϊ-1��Ҳ��Ĭ��ֵ������ֵͼ���ڱ��ǰ���Ļ�byIgnoreȡ0
    //    nType    - �����ע���������ͣ�����Ϊ4�������8����Ĭ��Ϊ8.
    static int ImgRegionLabel(const BYTE* pbyImg, 
                              int* pnOut, 
                              const SIZE& szImg, 
                              BYTE byIgnore = -1, 
                              int nType = 8);

    // A
    // ƽ����sobel�����ݶ�
    //    pbyGray    - ����ĻҶ�ͼ��
    //    szGray     - ����ͼ��ߴ�
    //    nBpp       - ����ͼ��λ��ȣ�����Ϊ8
    //    pbyMask    - ��Ĥͼ�񣬳ߴ���Ҷ�ͼ����ͬ��ĳ����Ϊ0ʱ����ʾ�õ㲻�����ݶ�ͼ��Ӧ�㱣��Ϊ0
    //    pnGrad     - �ݶ�ͼ���������Ӧ�����֮����ƽ���ᳬ��255���ʹ��int��
    //    dbMeanConv �� ƽ���ݶ�ֵ
    //    nDir       - ˮƽ��ֱ���򣬲��ò�ͬ��sobel����
    static void SobelPowGrad(const BYTE* pbyGray, 
                             const SIZE& szGray, 
                             int nBpp, 
                             const BYTE* pbyMask, 
                             int* pnGrad, 
                             double& dbMeanConv, 
                             int nDir = DIR_VERTICAL);

    // A
    // ����ֵ��sobel�����ݶ�
    //    pbyGray    - ����ĻҶ�ͼ��
    //    szGray     - ����ͼ��ߴ�
    //    nBpp       - ����ͼ��λ��ȣ�����Ϊ8
    //    pbyMask    - ��Ĥͼ�񣬳ߴ���Ҷ�ͼ����ͬ��ĳ����Ϊ0ʱ����ʾ�õ㲻�����ݶ�ͼ��Ӧ�㱣��Ϊ0
    //    pnGrad     - �ݶ�ͼ���������Ӧ�����֮����ƽ���ᳬ��255���ʹ��int��
    //    nDir       - ˮƽ��ֱ���򣬲��ò�ͬ��sobel����
    static void SobelGrad(const BYTE* pbyGray, 
                          const SIZE& szGray, 
                          int nBpp, 
                          const BYTE* pbyMask, 
                          int* pnGrad,
                          int nDir = DIR_VERTICAL);
    
    // A
    // hewitt�����ݶ�
    //    pbyGray    - ����ĻҶ�ͼ��
    //    szGray     - ����ͼ��ߴ�
    //    nBpp       - ����ͼ��λ��ȣ�����Ϊ8
    //    pbyMask    - ��Ĥͼ�񣬳ߴ���Ҷ�ͼ����ͬ��ĳ����Ϊ0ʱ����ʾ�õ㲻�����ݶ�ͼ��Ӧ�㱣��Ϊ0
    //    pbyEnhance  - �Ƿ��ǿ�ݶ�ֵ�ı��ͼ����ͼ�ߴ���pbyGray��ͬ�� ����ǿ���򽫶�Ӧ����ݶ�ֵ�ӱ�
    //    pnGrad     - �ݶ�ͼ���������Ӧ�����֮�ͻᳬ��255���ʹ��int��
    //    dbMeanConv �� ƽ���ݶ�ֵ
    //    nDir       - ˮƽ��ֱ���򣬲��ò�ͬ��hewitt����
    static void HewittGrad(const BYTE* pbyGray, 
                           const SIZE& szGray, 
                           int nBpp, 
                           const BYTE* pbyMask, 
                           const BYTE* pbyEnhance, 
                           int* pnGrad, 
                           double& dbMeanConv, 
                           int nDir = DIR_VERTICAL);

    // C
    // ����3*3��һ�����ӣ�����֮�͵���1���˲�,
    //    pbySrc    - ����ͼ��  
    //    szSrc     - ����ͼ��ߴ�
    //    nBpp      - ����ͼ��λ���
    //    pbyDst    - �������ȡ����ͼ��, ���ͼ��ߴ�������ͼ����ͬ
    //    pdbKernel - 3*3���ӣ�ʵ������Ϊһ�оŸ���
    static void ImgFilter(const BYTE* pbySrc, 
                          const SIZE& szSrc, 
                          int nBpp, 
                          BYTE* pbyDst, 
                          const double* pdbKernel);

    // A
    // ���ݶȼ����Եͼ�����ö�ֵ��������
    //    pnGrad     - �ݶ�ͼ��
    //    szGrad     - �ݶ�ͼ��ߴ�
    //    pbyMask    - ��Ĥͼ�񣬳ߴ����ݶ�ͼ����ͬ��ĳ����Ϊ0ʱ����ʾ�õ㲻������Եͼ��Ӧ�㱣��Ϊ0
    //    dbMeanConv �� ƽ���ݶ�ֵ
    //    dbLevel    - ��ֵ����ֵ����ϵ��
    //    pbyEdge    - ����ı�Եͼ��
    //    nDir       - ˮƽ��ֱ����
    static void Grad2Edge(const int* pnGrad, 
                          const SIZE& szGrad, 
                          const BYTE* pbyMask,
                          double dbMeanGrad, 
                          double dbLevel, 
                          BYTE* pbyEdge, 
                          int nDir = DIR_VERTICAL);

    // A
    // ����ɫ������ȡ���ҳ��������������ص㣬����ˮƽ����4���������ͣ�
    //    pbyImg    - �����BGRͼ��
    //    szImg     - ����ͼ��ߴ�
    //    nBpp      - ����ͼ��λ��ȣ�����Ϊ24  
    //    pbyMask   - ��Ĥͼ�񣬳ߴ���ԭʼͼ����ͬ��ĳ����Ϊ0ʱ����ʾ�õ㲻������ɫ���ͼ����Ϊ0
    //    pbyBlue   - ��ɫ����
    //    pbyYellow - ��ɫ����
    //    pbyFlag   - ��ɫ���ɫ����ɫ���
    static void FlagPlateColor(const BYTE* pbyImg, 
                               const SIZE& szImg, 
                               int nBpp,
                               const BYTE* pbyMask,
                               BYTE* pbyBlue, 
                               BYTE* pbyYellow, 
                               BYTE* pbyFlag);

    // ʵ���������ݵĵ�ˣ���󷵻س˻�֮��
    static double MultyDot(double* dbSrcA, double* dbSrcB, int nLen);

    // B
    // 2*3����ͼ�����ͣ�2��3�У��߽粻�������ͺ��ͼ��ߴ粻�䣩
    //    pbyBinary - ����Ķ�ֵͼ��
    //    szBinary  - ����Ķ�ֵͼ��ߴ�
    //    nVal      - ���͵����ֵ
    //    pbyDst    - ���ͺ��ͼ�񣬵��ô˺���ǰǰ������ڴ�
    static void DilateFix_On2x3(const BYTE* pbyBinary, 
                                const SIZE& szBinary,
                                int nVal, 
                                BYTE* pbyDst);

    // B
    // 3*3����ͼ�����ͣ�3��3�У��߽粻�������ͺ��ͼ��ߴ粻�䣩
    //    pbyBinary - ����Ķ�ֵͼ��
    //    szBinary  - ����Ķ�ֵͼ��ߴ�
    //    nVal      - ���͵����ֵ
    //    pbyDst    - ���ͺ��ͼ�񣬵��ô˺���ǰǰ������ڴ�
    static void DilateFix_On3x3(const BYTE* pbBinary, 
                                const SIZE& szBinary,
                                int nVal, 
                                BYTE* pbDst);

    // B
    // 3*n����ͼ�����ͣ�3��n�У��߽紦�����ͺ��ͼ��ߴ����ӣ�
    //    pbyBinary - ����Ķ�ֵͼ��
    //    szBinary  - ����Ķ�ֵͼ��ߴ�
    //    nKenelWidth - ��������n
    //    nVal      - ���͵����ֵ
    //    szDst     - ���ͺ��ͼ��ߴ磬����� szDst.cx + n / 2, �ߵ��� szDst.cy + 2
    //    pbyDst    - ���ͺ��ͼ�񣬵��ô˺���ǰǰ������ڴ�
    static void DilateVary_On3xn(const BYTE* pbyBinary, 
                                 const SIZE& szBinary,
                                 int nKenelWidth, 
                                 int nVal, 
                                 const SIZE& szDst, 
                                 BYTE* pbyDst);

    // B
    // 5*n����ͼ�����ͣ�5��n�У��߽紦�����ͺ��ͼ��ߴ����ӣ�
    //    pbyBinary - ����Ķ�ֵͼ��
    //    szBinary  - ����Ķ�ֵͼ��ߴ�
    //    nKenelWidth - ��������n
    //    nVal      - ���͵����ֵ
    //    szDst     - ���ͺ��ͼ��ߴ磬����� szDst.cx + n / 2, �ߵ��� szDst.cy + 4
    //    pbyDst    - ���ͺ��ͼ�񣬵��ô˺���ǰǰ������ڴ�
    static void DilateVary_On5xn(const BYTE* pbyBinary, 
                                 const SIZE& szBinary,
                                 int nKenelWidth, 
                                 int nVal, 
                                 const SIZE& szDst, 
                                 BYTE* pbyDst);

    // D
    // ��ֵͼ��ͶӰ����
    //    pbyBinary - ����Ķ�ֵͼ��
    //    szBinary  - ����Ķ�ֵͼ��ߴ�
    //    nProjLen  - ͶӰֵ������Ӧ����ͼ��Ŀ�����ͬ
    //    pnProj    - ͶӰ����
    //    nMOde     - ˮƽ��ֱ����
    //    byBackPtVal - ����ֵ��ͶӰͳ��ʱ������ֵ����byBackPtVal�Ĳ�����ͳ��
    static void BwProjection(const BYTE* pbyBinary, 
                             const SIZE& szBinary, 
                             int nProjLen,
                             int* pnProj, 
                             int nMode = DIR_VERTICAL, 
                             BYTE byBackPtVal = 0);

    // D
    // ��ֵͼ��ָ��������ͶӰ
    //    pbyBinary - ����Ķ�ֵͼ��
    //    szBinary  - ����Ķ�ֵͼ��ߴ�
    //    rtLoc     - ָ������ľ��ο�λ��
    //    nProjLen  - ͶӰֵ������Ӧ����ָ������Ŀ�����ͬ
    //    pnProj    - ͶӰ����
    //    nMOde     - ˮƽ��ֱ����
    //    byBackPtVal - ����ֵ��ͶӰͳ��ʱ������ֵ����byBackPtVal�Ĳ�����ͳ��
    static void LocProjection(const BYTE* pbyBinary, 
                              const SIZE& szBinary, 
                              int nProjLen,  
                              const RECT& rtLoc, 
                              int* pnProj, 
                              int nMode= DIR_VERTICAL, 
                              BYTE byBackPtVal = 0);

private:
    // D
    // ����ͨ����-�����ע���������ر�ע�õ�������ĸ�����
    //    pbyImg   - �����BGRͼ��
    //    pnOut    - ����ı��ͼ��
    //    szImg    - ����ͼ��ߴ�
    //    byIgnore - ����Ҫ��ע������ֵ����ȫ������ֵ����Ҫ����ע����Ϊ-1��Ҳ��Ĭ��ֵ������ֵͼ���ڱ��ǰ��ʱ byIgnoreȡ0
    static int ImgRegionLabel_4(const BYTE* pbyImg, 
                                int* pnOut, 
                                const SIZE& szImg, 
                                BYTE byIgnore);

    // D
    // ����ͨ����-�����ע���������ر�ע�õ�������ĸ�����
    //    pbyImg   - �����BGRͼ��
    //    pnOut    - ����ı��ͼ��
    //    szImg    - ����ͼ��ߴ�
    //    byIgnore - ����Ҫ��ע������ֵ����ȫ������ֵ����Ҫ����ע����Ϊ-1��Ҳ��Ĭ��ֵ������ֵͼ���ڱ��ǰ��ʱ byIgnoreȡ0
    static int ImgRegionLabel_8(const BYTE* pbyImg, 
                                int* pnOut, 
                                const SIZE& szImg, 
                                BYTE byIgnore);

    // D
    // ��������������
    //    pbyImg          - �����BGRͼ��
    //    szImg           - ����ͼ��ߴ�
    //    nLastCol        - ���һ�е������
    //    pbyFlag         - ���ص��Ƿ��ѷ��ʲ���ǹ��ı�ʾͼ����ͼ�ߴ�������Դͼ����ͬ
    //    pbyFlagRowStart - �ڶ��е���ʼ�����
    //    pbyFlagRowEnd   - �����ڶ��еĽ��������
    //    ppbyStack       - ������һ������ص��ָ�뼯��
    //    pbyPxl          - ��ǰ������ֵ
    //    nLabelInd       - ��ǰ��ı�Ǻ���
    //    pnOut           - ���±��ͼ�����ͼ�ߴ�������Դͼ����ͬ
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
    // ��������������
    //    pbyImg          - �����BGRͼ��
    //    szImg           - ����ͼ��ߴ�
    //    nLastCol        - ���һ�е������
    //    pbyFlag         - ���ص��Ƿ��ѷ��ʲ���ǹ��ı�ʾͼ����ͼ�ߴ�������Դͼ����ͬ
    //    pbyFlagRowStart - �ڶ��е���ʼ�����
    //    pbyFlagRowEnd   - �����ڶ��еĽ��������
    //    ppbyStack       - ������һ������ص��ָ�뼯��
    //    pbyPxl          - ��ǰ������ֵ
    //    nLabelInd       - ��ǰ��ı�Ǻ���
    //    pnOut           - ���±��ͼ�����ͼ�ߴ�������Դͼ����ͬ
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

    // �ж��Ƿ�Ϊ���ܵ���ɫ��
    //    pi - �������ص����ʼ���꣬��ΪBGR��ɫͼ��pi, pi + 1, pi + 2Ӧ�÷ֱ�Ϊ�����ص��B��G��Rͨ��ֵ
    static bool IsBlue(const BYTE* pi);

    // �ж��Ƿ�Ϊ���ܵĻ�ɫ��
    //    pi - �������ص����ʼ���꣬��ΪBGR��ɫͼ��pi, pi + 1, pi + 2Ӧ�÷ֱ�Ϊ�����ص��B��G��Rͨ��ֵ
    static bool IsYellow(const BYTE* pi);
};

