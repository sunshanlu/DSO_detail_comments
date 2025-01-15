/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#include "FullSystem/PixelSelector2.h"

//

#include "FullSystem/HessianBlocks.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"

namespace dso {

PixelSelector::PixelSelector(int w, int h) {
    randomPattern = new unsigned char[w * h];
    std::srand(3141592); // want to be deterministic.
    for (int i = 0; i < w * h; i++)
        randomPattern[i] = rand() & 0xFF; // 随机数, 取低8位

    currentPotential = 3;

    // 32*32个块进行计算阈值
    gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)];
    ths = new float[(w / 32) * (h / 32) + 100];
    thsSmoothed = new float[(w / 32) * (h / 32) + 100];

    allowFast = false;
    gradHistFrame = 0;
}

PixelSelector::~PixelSelector() {
    delete[] randomPattern;
    delete[] gradHist;
    delete[] ths;
    delete[] thsSmoothed;
}

/**
 * @brief 根据32 * 32像素的直方图分布hist，求分位数below的对应梯度值
 *
 * @param hist  输入的长度为50，但梯度部分只有[0,48]的直方图，hist0部分存储的是一个小块内的所有个数
 * @param below 给定below分位数，求梯度
 * @return int  输出指定分位数below的梯度
 */
int computeHistQuantil(int *hist, float below) {
    int th = hist[0] * below + 0.5f; // 最低的像素个数

    /// 这里第0个是所有的32 * 32的像素个数，而这里敢用90的原因，是因为i最大为48
    for (int i = 0; i < 90; i++) {
        th -= hist[i + 1];
        if (th < 0)
            return i;
    }
    return 90;
}

/**
 * @brief 生成平滑梯度直方图thsSmoothed
 * @details
 *  1. 将第0层的图像，分成w32 * h32个小块，每个小块的像素个数为32 * 32，对于多余的部分，直接剔除
 *  2. 对每个小块，统计梯度的根下平方和，分成49个hist，用于统计[0,48]的部分，针对大于48的部分，统计到48里面
 *  3. 对每个小块，找到0.5个分位数对应的根号下平方和，并加上偏置7，作为这个小块中的未平滑的梯度阈值ths
 *  4. 使用3 * 3均值滤波的方式，对不同小块的ths进行平滑，得到平滑结果thisSmooted
 * @note 针对3 * 3的均值滤波部分，不满足3 * 3的边缘部分，进行自适应的窗口调整
 * @param fh 输入的FrameHessian数据
 */
void PixelSelector::makeHists(const FrameHessian *const fh) {
    gradHistFrame = fh;
    float *mapmax0 = fh->absSquaredGrad[0]; // 第0层梯度平方和

    // weight and height
    int w = wG[0];
    int h = hG[0];

    int w32 = w / 32;
    int h32 = h / 32;
    thsStep = w32;

    for (int y = 0; y < h32; y++)
        for (int x = 0; x < w32; x++) {
            float *map0 = mapmax0 + 32 * x + 32 * y * w; // y行x列的格
            int *hist0 = gradHist;                       // + 50*(x+y*w32);
            memset(hist0, 0, sizeof(int) * 50);          // 分成50格

            for (int j = 0; j < 32; j++)
                for (int i = 0; i < 32; i++) {
                    int it = i + 32 * x; // 该格里第(j,i)像素的整个图像坐标
                    int jt = j + 32 * y;
                    if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1)
                        continue;                   // 内
                    int g = sqrtf(map0[i + j * w]); // 梯度平方和开根号
                    if (g > 48)
                        g = 48;     //? 为啥是48这个数，因为一共分为了50格
                    hist0[g + 1]++; // 1-49 存相应梯度个数
                    hist0[0]++;     // 所有的像素个数
                }
            // setting_minGradHistCut的默认值为0.5，setting_minGradHistAdd为7，即0.5分位数 + 7作为每个块的阈值
            ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd;
        }

    // 使用3*3的窗口的均值滤波进行梯度的ths的平滑
    for (int y = 0; y < h32; y++)
        for (int x = 0; x < w32; x++) {
            float sum = 0;
            int num = 0;
            if (x > 0) {
                if (y > 0) {
                    num++;
                    sum += ths[x - 1 + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x - 1 + (y + 1) * w32];
                }
                num++;
                sum += ths[x - 1 + (y)*w32];
            }

            if (x < w32 - 1) {
                if (y > 0) {
                    num++;
                    sum += ths[x + 1 + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x + 1 + (y + 1) * w32];
                }
                num++;
                sum += ths[x + 1 + (y)*w32];
            }

            if (y > 0) {
                num++;
                sum += ths[x + (y - 1) * w32];
            }
            if (y < h32 - 1) {
                num++;
                sum += ths[x + (y + 1) * w32];
            }
            num++;
            sum += ths[x + y * w32];

            thsSmoothed[x + y * w32] = (sum / num) * (sum / num);
        }
}

/**
 * @brief 使用像素选择器，进行金字塔第0层上的点选择
 * @details
 *  1. 给定一个初始后续会动态变化的idealPotential，初始为3
 *  2. 首先，使用makeHists函数，生成每个32 * 32 区块的平滑阈值 @see PixelSelector::makeHists
 *  3. 其次使用select函数，根据给定的idealPotential进行点的选择 @see PixelSelector::select
 *  4. 计算want / have的比例，判断是否在0.25到1.25区间范围内
 *      4.1 若小于0.25，则代表目前得到的梯度较大点太多了，需要增大idealPotential，若recursionsLeft > 0 递归调用makeMaps
 *      4.2 若大于1.255，则代表目前得到的梯度较大的点太少了，需要减小idealPotential，若recursionsLeft > 0 递归调用makeMaps 
 *      4.3 若比例在0.25到1.25之间，则代表目前得到的梯度较大点数目比较合适，不会进行递归了
 *  5. 判断若比例在[0.95, 1.25]之间，说明点的数量正合适，返回得到的点的数目
 *  6. 若比例在[0.25, 0.95)之间，需要随机删除点，直到比例控制在1左右
 *  7. 最后，将结果绘制出来即可
 * @param fh 				输入的FrameHessian数据
 * @param map_out 			选出的地图点
 * @param density 			金字塔层的密度
 * @param recursionsLeft 	最大递归次数
 * @param plot 				是否画图
 * @param thFactor 			阈值因子
 * @return int				选择的点的数量
 */
int PixelSelector::makeMaps(const FrameHessian *const fh, float *map_out, float density, int recursionsLeft, bool plot, float thFactor) {
    float numHave = 0;
    float numWant = density;
    float quotia;
    int idealPotential = currentPotential;

    /// 当mapMaps以非递归形式调用时，会使用makeHists函数，生成32 * 32区块的平滑阈值
    if (fh != gradHistFrame)
        makeHists(fh);

    /// 在当前帧上选择符合条件的像素，会使用0、1、2三个金字塔层级去选择
    Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

    numHave = n[0] + n[1] + n[2]; // 选择得到的点
    quotia = numWant / numHave;   // 得到的 与 想要的 比例

    /// DSO给出了一个金字塔采样点模型，符合 numselected = K / (pot + 1) ^ 2，其中K为场景相关的常数。
    /// 这里是计算如果满足上述的金字塔采样点模型，理想的pot应该是多少，当计算的理想pot小于1时，置为1
    float K = numHave * (currentPotential + 1) * (currentPotential + 1); // 相当于覆盖的面积, 每一个像素对应一个pot*pot
    idealPotential = sqrtf(K / numWant) - 1;                             // round down.
    if (idealPotential < 1)
        idealPotential = 1;

    /// 满足递归条件 + 比例大于1.25，代表得到的点太少，需要减少pot的大小
    if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
        /// 这里对是否满足采样模型做了验证，如果理想pot比当前pot小，那么使用理想pot，否则，使用当前pot-1常规意义上的缩小
        if (idealPotential >= currentPotential)
            idealPotential = currentPotential - 1;

        currentPotential = idealPotential;

        /// 进行递归调用，并且将最大递归次数减1，recursionsLeft - 1
        return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
    }
    /// 满足递归条件 + 比例小于0.25，代表得到的点太多了，需要增大pot的大小
    else if (recursionsLeft > 0 && quotia < 0.25) {
        /// 这里同样对是否满足采样模型做验证，若满足采样模型，则使用理想pot，否则，使用当前pot + 1的常规放大
        if (idealPotential <= currentPotential)
            idealPotential = currentPotential + 1;

        currentPotential = idealPotential;
        /// 进行递归调用，并且将最大递归次数减1，recursionsLeft - 1
        return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
    }

    /// 当递归条件不满足时，如果比例还是小于0.95，则代表得到的点还是比较多，进行随机删除
    int numHaveSub = numHave;
    if (quotia < 0.95) {
        int wh = wG[0] * hG[0];
        int rn = 0;
        unsigned char charTH = 255 * quotia;
        for (int i = 0; i < wh; i++) {
            if (map_out[i] != 0) {
                if (randomPattern[rn] > charTH) {
                    map_out[i] = 0;
                    numHaveSub--;
                }
                rn++;
            }
        }
    }

    /// 将理想pot赋值给当前pot
    currentPotential = idealPotential;

    // 画出选择结果
    if (plot) {
        int w = wG[0];
        int h = hG[0];

        MinimalImageB3 img(w, h);

        for (int i = 0; i < w * h; i++) {
            float c = fh->dI[i][0] * 0.7; // 像素值
            if (c > 255)
                c = 255;
            img.at(i) = Vec3b(c, c, c);
        }
        IOWrap::displayImage("Selector Image", &img);

        // 安照不同层数的像素, 画上不同颜色
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int i = x + y * w;
                if (map_out[i] == 1)
                    img.setPixelCirc(x, y, Vec3b(0, 255, 0)); ///< 第0层选择的点，设为绿色
                else if (map_out[i] == 2)
                    img.setPixelCirc(x, y, Vec3b(255, 0, 0)); ///< 第1层选择的点，设为红色
                else if (map_out[i] == 4)
                    img.setPixelCirc(x, y, Vec3b(0, 0, 255)); ///< 第2层选择的点，设为蓝色
            }
        IOWrap::displayImage("Selector Pixels", &img);
    }

    return numHaveSub;
}

/**
 * @brief 使用4pot * 4pot --> 2pot * 2pot --> pot * pot --> pixel * pixel的方式去遍历选择梯度大的像素点
 * @details
 *  1. 对某个pot，进行逐像素的遍历梯度平方和，试图找到满足平滑阈值条件且最大梯度的像素点位置，作为第0层的输出
 *  2. 在2pot * 2pot内，如果4个pot都没有找到平滑阈值条件的像素点位置，则进行0.75倍的阈值缩小
 *      2.1 在2pot * 2pot内，逐像素的遍历点并投影到第1层，试图找到第一层满足平滑阈值条件且最大梯度的像素点位置，作为第1层的输出
 *  3. 在4pot * 4pot内，如果4个2pot * 2pot都没有找到第一层符合平滑阈值条件的像素点位置，则再次进行0.75倍的阈值缩小
 *      3.1 在4pot * 4pot内，逐像素的遍历点，并投影到第2层，试图找到第二层满足平滑阈值条件且最大梯度的像素点位置，作为第2层的输出
 *  4. 这样通过两次梯度减小的方式，可以尝试选择一个大范围小梯度的内的相对大梯度特征点，相当于“瘸子里面选将军！”
 *  5. 这样做的目的
 *      5.1 使用pot的方式，使得观测的选取尽量在一张图片上均匀分布
 *      5.2 使用3层选择的方式，从小梯度里面选择出相对大梯度的点，不至于在小图像梯度的区域里面里面选择不出来
 *
 * @note 代码里面，使用0、1、2层逐像素同时计算的方式，来防止多次遍历。感觉针对一张不同图像，效果提升不多
 * @note //! 个人认为，使用回溯的方式实现这个select的算法，相对来说比较合适，而且应该会相当简洁
 * @param fh        输入的FrameHessian帧
 * @param map_out   输出的选择好的像素，内容填充0、1、2、4，代表未选择、0层、1层和2层像素点
 * @param pot       指定的pot大小，单位为px
 * @param thFactor  梯度阈值factor，通过乘法的方式进一步控制平滑阈值的大小 thSmooth * thFactor
 * @return Eigen::Vector3i  输出的[0层选择个数、1层选择个数、2层选择个数]
 */
Eigen::Vector3i PixelSelector::select(const FrameHessian *const fh, float *map_out, int pot, float thFactor) {

    Eigen::Vector3f const *const map0 = fh->dI;

    // 0, 1, 2层的梯度平方和
    float *mapmax0 = fh->absSquaredGrad[0];
    float *mapmax1 = fh->absSquaredGrad[1];
    float *mapmax2 = fh->absSquaredGrad[2];

    // 不同层的图像大小
    int w = wG[0];
    int w1 = wG[1];
    int w2 = wG[2];
    int h = hG[0];

    // 模都是1
    const Vec2f directions[16] = {Vec2f(0, 1.0000),       Vec2f(0.3827, 0.9239),  Vec2f(0.1951, 0.9808),  Vec2f(0.9239, 0.3827),
                                  Vec2f(0.7071, 0.7071),  Vec2f(0.3827, -0.9239), Vec2f(0.8315, 0.5556),  Vec2f(0.8315, -0.5556),
                                  Vec2f(0.5556, -0.8315), Vec2f(0.9808, 0.1951),  Vec2f(0.9239, -0.3827), Vec2f(0.7071, -0.7071),
                                  Vec2f(0.5556, 0.8315),  Vec2f(0.9808, -0.1951), Vec2f(1.0000, 0.0000),  Vec2f(0.1951, -0.9808)};

    /// 将map_out的内容里面的所有默认值都设为0
    memset(map_out, 0, w * h * sizeof(float));

    /// dw1,dw2分别为1层和2层相对0层阈值减小的比例 0.75 和 0.75 * 0.75
    float dw1 = setting_gradDownweightPerLevel;
    float dw2 = dw1 * dw1;

    /// 这里的n2, n3, n4分别代表着0层选择像素、1层选择像素个数、2层选择像素个数
    int n2 = 0, n3 = 0, n4 = 0;

    /// 4pot * 4pot的遍历，x4和y4代表着4pot * 4pot区域左上角像素位置
    for (int y4 = 0; y4 < h; y4 += (4 * pot))
        for (int x4 = 0; x4 < w; x4 += (4 * pot)) {

            /// my3和mx3代表着4pot * 4pot区域的大小，考虑了边界问题
            int my3 = std::min((4 * pot), h - y4);
            int mx3 = std::min((4 * pot), w - x4);

            int bestIdx4 = -1;
            float bestVal4 = 0;

            /// 随机选择一个方向
            Vec2f dir4 = directions[randomPattern[n2] & 0xF];

            /// 2pot * 2pot的遍历
            for (int y3 = 0; y3 < my3; y3 += (2 * pot))
                for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {

                    /// x34和y34代表着2pot * 2pot区域的左上角像素位置
                    int x34 = x3 + x4;
                    int y34 = y3 + y4;

                    /// my2和mx2代表着2pot * 2pot区域的区域大小，考虑了边界问题
                    int my2 = std::min((2 * pot), h - y34);
                    int mx2 = std::min((2 * pot), w - x34);

                    int bestIdx3 = -1;
                    float bestVal3 = 0;

                    /// 随机选择一个方向
                    Vec2f dir3 = directions[randomPattern[n2] & 0xF];

                    /// pot * pot的遍历
                    for (int y2 = 0; y2 < my2; y2 += pot)
                        for (int x2 = 0; x2 < mx2; x2 += pot) {

                            /// x234和y234代表着pot * pot区域左上角像素位置
                            int x234 = x2 + x34;
                            int y234 = y2 + y34;

                            /// my1和mx1代表着是pot * pot区域的大小，考虑了边界问题
                            int my1 = std::min(pot, h - y234);
                            int mx1 = std::min(pot, w - x234);

                            int bestIdx2 = -1;
                            float bestVal2 = 0;

                            /// 随机选择一个方向
                            Vec2f dir2 = directions[randomPattern[n2] & 0xF];

                            /// pixel * pixel逐像素的遍历
                            for (int y1 = 0; y1 < my1; y1 += 1)
                                for (int x1 = 0; x1 < mx1; x1 += 1) {
                                    assert(x1 + x234 < w);
                                    assert(y1 + y234 < h);

                                    /// idx对应像素的idx，即图像按行展成一行时对应的id
                                    int idx = x1 + x234 + w * (y1 + y234);

                                    /// xf和yf代表的是像素坐标
                                    int xf = x1 + x234;
                                    int yf = y1 + y234;

                                    /// 当像素落在图像的宽度为4的边框内时，不进行判断，直接跳过
                                    if (xf < 4 || xf > w - 4 || yf < 4 || yf > h - 4)
                                        continue;

                                    /// 找到xf和yf像素对应的thisSmoothed的平滑像素梯度阈值，然后以逐级递减的方式获取1层和2层对应平滑阈值
                                    float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep];
                                    float pixelTH1 = pixelTH0 * dw1;
                                    float pixelTH2 = pixelTH1 * dw2;

                                    /// ag0为0层对应的像素梯度平方和
                                    float ag0 = mapmax0[idx];

                                    /// 当ag0大于平滑梯度阈值 * thFactor时
                                    if (ag0 > pixelTH0 * thFactor) {
                                        /// 去除对应位置的能量梯度和随机方向进行投影
                                        Vec2f ag0d = map0[idx].tail<2>();
                                        float dirNorm = fabsf((float)(ag0d.dot(dir2)));

                                        /// 如果setting_selectDirectionDistribution为true，则用于比较大小的值使用能量梯度和随机方向的投影
                                        /// 否则，使用原图梯度平方和作为用于比较大小的值dirNorm
                                        if (!setting_selectDirectionDistribution)
                                            dirNorm = ag0;
                                        // 第0层，通过大小比较的方式取最大值，若有一个0层位置的点被选择，则bestIdx3和bestIdx4都被置为-2
                                        if (dirNorm > bestVal2) {
                                            bestVal2 = dirNorm;
                                            bestIdx2 = idx;
                                            bestIdx3 = -2;
                                            bestIdx4 = -2;
                                        }
                                    }

                                    /// 如果第0层选择了点，则不会在第一层选择，直接continue
                                    if (bestIdx3 == -2)
                                        continue;

                                    /// 如果第0层的梯度不满足梯度阈值要求，那么进行第1层的判断
                                    /// ag1为第1层的像素梯度平方和
                                    float ag1 = mapmax1[(int)(xf * 0.5f + 0.25f) + (int)(yf * 0.5f + 0.25f) * w1];

                                    /// 如果第一层满足阈值要求
                                    if (ag1 > pixelTH1 * thFactor) {
                                        /// 获取能量梯度
                                        Vec2f ag0d = map0[idx].tail<2>();

                                        /// 能量梯度和随机方向的dot
                                        float dirNorm = fabsf((float)(ag0d.dot(dir3)));

                                        /// setting_selectDirectionDistribution的判断
                                        if (!setting_selectDirectionDistribution)
                                            dirNorm = ag1;

                                        /// 如果第1层选择，那么第二层
                                        if (dirNorm > bestVal3) {
                                            bestVal3 = dirNorm;
                                            bestIdx3 = idx;
                                            bestIdx4 = -2;
                                        }
                                    }
                                    if (bestIdx4 == -2)
                                        continue;

                                    float ag2 = mapmax2[(int)(xf * 0.25f + 0.125) + (int)(yf * 0.25f + 0.125) * w2]; // 第2层
                                    if (ag2 > pixelTH2 * thFactor) {
                                        Vec2f ag0d = map0[idx].tail<2>();
                                        float dirNorm = fabsf((float)(ag0d.dot(dir4)));
                                        if (!setting_selectDirectionDistribution)
                                            dirNorm = ag2;

                                        if (dirNorm > bestVal4) {
                                            bestVal4 = dirNorm;
                                            bestIdx4 = idx;
                                        }
                                    }
                                }

                            /// 当使用pixel * pixel的方式，遍历完成pot * pot区域后
                            /// 若0层有满足阈值条件且最优的点，则设置pot * pot区域内idx输出到map_out中，设为1
                            /// 并且将bestVal3置一个足够大的数，接下来不会进入2pot * 2pot区域内进行判断选值了
                            if (bestIdx2 > 0) {
                                map_out[bestIdx2] = 1;
                                bestVal3 = 1e10;
                                n2++;
                            }
                        }

                    /// 当使用pixel * pixel的方式遍历完成2pot * 2pot区域后
                    /// 若整个2pot * 2pot内的第0层像素点，都不满足阈值条件
                    /// 且2pot * 2pot内的第1层像素满足了梯度阈值条件
                    /// 设置2pot * 2pot区域内的idx输出到map_out中，设为2
                    /// 将bestVal4置一个足够大的数，接下来不会进入4pot * 4pot区域内判断选值了
                    if (bestIdx3 > 0) {
                        map_out[bestIdx3] = 2;
                        bestVal4 = 1e10;
                        n3++;
                    }
                }

            /// 若整个4pot * 4pot内，0层和1层都没有满足阈值条件的像素
            /// 且4pot * 4pot内的第2层像素满足了梯度阈值条件
            /// 设置4pot * 4pot内的idx输出到map_out中，设为4，代表第2层
            if (bestIdx4 > 0) {
                map_out[bestIdx4] = 4;
                n4++;
            }
        }

    return Eigen::Vector3i(n2, n3, n4); // 第0, 1, 2层选点的个数
}

} // namespace dso
