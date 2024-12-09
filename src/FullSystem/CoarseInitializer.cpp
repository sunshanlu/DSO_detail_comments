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

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/Residuals.h"
#include "util/nanoflann.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

CoarseInitializer::CoarseInitializer(int ww, int hh)
    : thisToNext_aff(0, 0)
    , thisToNext(SE3()) {
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        points[lvl] = 0;
        numPoints[lvl] = 0;
    }

    JbBuffer = new Vec10f[ww * hh];
    JbBuffer_new = new Vec10f[ww * hh];

    frameID = -1;
    fixAffine = true;
    printDebug = false;

    //! 这是
    wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
    wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
    wM.diagonal()[6] = SCALE_A;
    wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer() {
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        if (points[lvl] != 0)
            delete[] points[lvl];
    }

    delete[] JbBuffer;
    delete[] JbBuffer_new;
}

/**
 * @brief 初始化器中的跟踪frame
 * @details
 *  1. 如果上一帧的tji没有满足要求，那么将点的idepth_new，iR和置为1（统一尺度，放弃优化先验），lastHessian置0，如果tji满足要求，则保留先验
 *  2. 遍历高斯金字塔层级，进行逐层的优化
 *      2.1 针对非顶层部分，使用金字塔点的父子关系和金字塔点的邻居关系（中值），更新当前层lvl点的期望
 *      2.2 针对顶层部分，使用邻居值的期望的平均值进行期望更新
 *      2.3 将lvl层的金字塔点的energy置0
 *      2.4 计算当前参数下的H，Hsc，b，bsc，以及对应的能量值，作为oldRes
 *      2.5 遍历当前层的最大迭代次数
 *          2.5.1 使用Schur的方式，优化delta_x的求解过程，求得优化后的参数 x_new
 *          2.5.2 使用优化后的参数进行H，Hsc，b，bsc和能量newRes的求解
 *          2.5.3 如果发生了能量减小的情况，则接受当前lvl层的优化结果，否则不接受当前的优化结果
 *          2.5.4 若被连续拒绝次数超过2 | 达到最大迭代次数 | 或者优化的增量eps小于阈值，则跳出优化循环
 *      2.6 使用由粗至精的优化方案，得到一个相对准确的Tji、aji、bji和第0层点的逆深度后
 *      2.7 从低到高遍历金字塔层级，使用高斯归一化的方式，更新上一层的 逆深度和逆深度期望iR，并考虑邻居点的iR进行期望的平滑处理（中值）
 *      2.8 维护一个连续优化保证tji满足要求的起始帧id --> snappedAt
 *      2.9 如果优化中，连续5帧都能保证tji满足要求，则认为初始化成功
 * @param newFrameHessian   输入的待跟踪的帧j
 * @param wraps             输入的输出装饰器，用来指定输出结果
 * @return true     初始化成功
 * @return false    初始化失败
 */
bool CoarseInitializer::trackFrame(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper *> &wraps) {
    newFrame = newFrameHessian;
    for (IOWrap::Output3DWrapper *ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

    /// 定义每一层优化迭代的最大迭代次数
    int maxIterations[] = {5, 5, 10, 30, 50};

    /// 用来控制整个优化过程的参数，可以进行动态的调整
    alphaK = 2.5 * 2.5; ///< 和alphaW一起控制 优化变量tji的大小
    alphaW = 150 * 150; ///< 和alphaK一起控制 优化变量tji的大小
    regWeight = 0.8;    ///< iR期望中位数的置信度
    couplingWeight = 1; ///< tji满足时，逆深度的正则化项部分系数

    if (!snapped) /// 若 tji 没有达到 alphaW 和 alphaK 的要求，则需要进行一些修正
    {
        //! 初始化，这里的初始化直接全部重置，相当于去掉了之前优化的内容，是否会增加初始化时间的内容呢？
        thisToNext.translation().setZero(); ///< 1. 改变Tji中的平移向量，将之前优化的内容进行置0处理
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            int npts = numPoints[lvl];
            Pnt *ptsl = points[lvl];
            for (int i = 0; i < npts; i++) {
                ptsl[i].iR = 1;          ///< 2. 将i帧上的点对应的期望iR逆深度全部置为1
                ptsl[i].idepth_new = 1;  ///< 3. 将i帧上的新逆深度置为1
                ptsl[i].lastHessian = 0; ///< 4. 将i帧上之前计算的hessian置为0，即点的部分去掉先验
            }
        }
    }

    /// 拷贝优化初始值,当金字塔优化层被接受时，更新 refToNew_current
    SE3 refToNew_current = thisToNext;
    AffLight refToNew_aff_current = thisToNext_aff;

    /// 如果都存在曝光时间，则根据曝光时间计算出一个初始的仿射参数
    if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
        refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure), 0);

    Vec3f latestRes = Vec3f::Zero();

    /// 从顶层开始估计，一直估计到金字塔的第0层
    for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) {

        if (lvl < pyrLevelsUsed - 1) {
            /// 针对非顶层部分，使用上一层parent和同层neighbours点来初始化当前层的点逆深度期望iR @see propagateDown
            propagateDown(lvl + 1);
        }

        /// 定义 E 对 Tji 、 aji 、 bji 对应的海塞矩阵和雅可比矩阵
        Mat88f H, Hsc; ///< 没有进行schur和进行schur之后的H矩阵
        Vec8f b, bsc;  ///< 没有进行schur和进行schur之后的J向量

        resetPoints(lvl); ///< 重置点的 energy 和 idepth_new 作为逆深度优化初值，并进行顶层点的iR更新

        /// 计算当前参数下的海塞矩阵和雅可比矩阵，用来构建H @see calcResAndGS
        Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);

        /// 详细内容 @see applyStep
        applyStep(lvl); ///< 这里的作用是将point->energy_new 、point->lastHessian_new 、point->isGood_new 更新,idepth_new没有变化

        float lambda = 0.1;
        float eps = 1e-4;
        int fails = 0;

        /// 初始信息
        if (printDebug) {
            printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t", lvl, 0, lambda, "INITIA",
                   sqrtf((float)(resOld[0] / resOld[2])), // 卡方(res*res)平均值
                   sqrtf((float)(resOld[1] / resOld[2])), // 逆深度能量平均值
                   sqrtf((float)(resOld[0] / resOld[2])), sqrtf((float)(resOld[1] / resOld[2])), (resOld[0] + resOld[1]) / resOld[2],
                   (resOld[0] + resOld[1]) / resOld[2], 0.0f);
            std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() << "\n";
        }

        /// 上面使用初值计算除了初始的能量值和与优化对应的H，b，Hsc和bsc
        int iteration = 0;
        while (true) {
            Mat88f Hl = H; ///< 保存边缘化后的H矩阵，并考虑LM的影响 Hnew = H[diag] * (1 + lambda)

            /// 对边缘化的U矩阵，LM方法仅作用在了对角线的部分
            for (int i = 0; i < 8; i++)
                Hl(i, i) *= (1 + lambda);

            /// 对需要减去的部分 W V^-1 W^T，针对V ---> V * (1 + lambda)
            Hl -= Hsc * (1 / (1 + lambda));
            Vec8f bl = b - bsc * (1 / (1 + lambda));

            /// 这里使用了wM，将这里的H和b矩阵映射到了 x = A x'，映射到了x'上了
            Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl])); ///< 针对x'的H矩阵
            bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));      ///< 针对x'的b向量

            ///! 使用边缘化后的Hsc和bsc进行增量计算，但是这里使用wM的方式，貌似没有出现应有的效果，因为后续使用wM系数矩阵又将优化增量部分映射回去了
            Vec8f inc;
            if (fixAffine) ///< 考虑了a和b防射参数固定的情况，原理在于固定待优化量，其雅可比为0，索引可以直接将H矩阵中的某几块置0
            {
                /// 这里使用的是将 6 * 6的部分拿出来了，和置零的情况一致
                inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() * (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                inc.tail<2>().setZero();
            } else                                   ///< 考虑了参数非固定的情况
                inc = -(wM * (Hl.ldlt().solve(bl))); ///< delta_x = -H^-1 * b.

            /// 使用左乘的方式更新Tji
            SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;

            /// 更新仿射参数
            AffLight refToNew_aff_new = refToNew_aff_current;
            refToNew_aff_new.a += inc[6];
            refToNew_aff_new.b += inc[7];

            /// 更新点的逆深度
            doStep(lvl, lambda, inc);

            Mat88f H_new, Hsc_new;
            Vec8f b_new, bsc_new;

            /// 使用calcResAndGS的方式计算新的H矩阵，J矩阵和对应的能量值resNew
            Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
            Vec3f regEnergy = calcEC(lvl); ///< 计算点正则化对应的能量部分

            float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]); ///< 新能量
            float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]); ///< 旧能量

            bool accept = eTotalOld > eTotalNew; ///< 当发生能量减少时，则接受更新

            if (printDebug) {
                printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t", lvl, iteration, lambda,
                       (accept ? "ACCEPT" : "REJECT"), sqrtf((float)(resOld[0] / resOld[2])), sqrtf((float)(regEnergy[0] / regEnergy[2])),
                       sqrtf((float)(resOld[1] / resOld[2])), sqrtf((float)(resNew[0] / resNew[2])), sqrtf((float)(regEnergy[1] / regEnergy[2])),
                       sqrtf((float)(resNew[1] / resNew[2])), eTotalOld / resNew[2], eTotalNew / resNew[2], inc.norm());
                std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() << "\n";
            }

            if (accept) {
                // 这是t平移足够大时
                if (resNew[1] == alphaK * numPoints[lvl]) // 当 alphaEnergy > alphaK*npts
                    snapped = true;
                H = H_new;
                b = b_new;
                Hsc = Hsc_new;
                bsc = bsc_new;
                resOld = resNew;
                refToNew_aff_current = refToNew_aff_new;
                refToNew_current = refToNew_new;
                applyStep(lvl);      ///< 交换了点的状态，深度和能量值，以及中间变量JbBuffer
                optReg(lvl);         ///< 更新iR
                lambda *= 0.5;       ///< lambda 递减，增大下面的可行域
                fails = 0;           ///< 连续失败次数置0
                if (lambda < 0.0001) ///< 要求lambda 最小为0.0001
                    lambda = 0.0001;
            } else {
                fails++;            ///< 连续失败次数+1
                lambda *= 4;        ///< lambda 递增，减小下面更新的可行域
                if (lambda > 10000) ///< 要求lambda 最大不能超过10000
                    lambda = 10000;
            }

            bool quitOpt = false;

            /// 1. deltaX的增量小于eps，代表收敛，优化结束
            /// 2. iteration大于最大迭代次数，代表优化加速，这里的迭代次数包括没有被接受的迭代‘
            /// 3. 当被拒绝的次数大于等于2次，代表收敛，优化结束
            if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
                quitOpt = true;

            if (quitOpt)
                break;
            iteration++;
        }
        latestRes = resOld;
    }

    /// 使用金字塔的方式从 粗 到 精 优化后，将优化的Tji，aji和bji进行更新
    thisToNext = refToNew_current;
    thisToNext_aff = refToNew_aff_current;

    for (int i = 0; i < pyrLevelsUsed - 1; i++)
        propagateUp(i); ///< 使用下一层的点信息更新上一层的idepth和iR，并且考虑邻居的平滑更新iR

    /// 处理snapped的状态
    frameID++;
    if (!snapped)
        snappedAt = 0; ///< snappedAt的意义应该为，连续snapped的第一帧id

    if (snapped && snappedAt == 0)
        snappedAt = frameID; ///< 位移足够的帧id

    debugPlot(0, wraps);

    /// 优化的连续5帧，都应该保证tji足够
    return snapped && frameID > snappedAt + 5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper *> &wraps) {
    bool needCall = false;
    for (IOWrap::Output3DWrapper *ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if (!needCall)
        return;

    int wl = w[lvl], hl = h[lvl];
    Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];

    MinimalImageB3 iRImg(wl, hl);

    for (int i = 0; i < wl * hl; i++)
        iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);

    int npts = numPoints[lvl];

    float nid = 0, sid = 0;
    for (int i = 0; i < npts; i++) {
        Pnt *point = points[lvl] + i;
        if (point->isGood) {
            nid++;
            sid += point->iR;
        }
    }
    float fac = nid / sid;

    for (int i = 0; i < npts; i++) {
        Pnt *point = points[lvl] + i;

        if (!point->isGood)
            iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));

        else
            iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));
    }

    // IOWrap::displayImage("idepth-R", &iRImg, false);
    for (IOWrap::Output3DWrapper *ow : wraps)
        ow->pushDepthImage(&iRImg);
}

/**
 * @brief 计算某个层级lvl上的，H，b，Hsc和bsc，即schur之前和之后的内容
 *
 * @param lvl           输入的金字塔层级
 * @param H_out         输出的H矩阵，仅包含Tji、aji和bji的内容（海塞矩阵）
 * @param b_out         输出的b向量，仅包含Tji、aji和bji的内容（雅可比矩阵）
 * @param H_out_sc      输出的Schur分解之后的Hsc矩阵
 * @param b_out_sc      输出的Schur分解之后的bsc向量
 * @param refToNew      输入输出的Tji
 * @param refToNew_aff  输入输出的aji和bji
 * @param plot          是否绘图
 * @return Vec3f        [能量，tji的正则项能量，用到的点的数量(rk)]
 */
Vec3f CoarseInitializer::calcResAndGS(int lvl, Mat88f &H_out, Vec8f &b_out, Mat88f &H_out_sc, Vec8f &b_out_sc, const SE3 &refToNew,
                                      AffLight refToNew_aff, bool plot) {
    /// 获取金字塔层级的宽度和高度
    int wl = w[lvl], hl = h[lvl];

    /// 获取i帧和j帧，I,dx,dy...
    Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
    Eigen::Vector3f *colorNew = newFrame->dIp[lvl];

    /// 获取Rji * K^-1、tji和e^{aji}和bji
    Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();                 ///< Rji * K_inv
    Vec3f t = refToNew.translation().cast<float>();                                   ///< tji
    Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b); ///< aji、bji光度仿射参数

    /// 获取该层的相机参数，fxl,fyl,cxl,cyl
    float fxl = fx[lvl];
    float fyl = fy[lvl];
    float cxl = cx[lvl];
    float cyl = cy[lvl];

    Accumulator11 E;   ///< 定义一个 1*1 的累加器，用于累加能量值部分
    acc9.initialize(); ///< 初始化acc9，一个9*9的累加器，用来累加计算H和b部分
    E.initialize();    ///< 初始化E累加器

    int npts = numPoints[lvl];
    Pnt *ptsl = points[lvl]; ///< i帧上，lvl层的候选点

    /// 遍历参考帧i上的lvl层级上的所有点point
    for (int i = 0; i < npts; i++) {

        Pnt *point = ptsl + i; ///< 获取pi->point
        point->maxstep = 1e10; ///< 初始化点的maxstep

        /// 针对在其他frame优化上，确定为坏点的情况下，设置完energy后，直接跳过这部分的内容计算
        if (!point->isGood) {
            /// 将坏点的能量部分，累加到E的累加器中
            E.updateSingle((float)(point->energy[0]));

            point->energy_new = point->energy; ///< 将energy_new部分拷贝energy的值，作为优化改变的能量值
            point->isGood_new = false;         ///< 在迭代过程中，判断是否为good，设置为false
            continue;
        }

        /// 这部分用于计算H和b矩阵
        VecNRf dp0; ///< drk / dx，索引为pattern num
        VecNRf dp1; ///< drk / dy，索引为pattern num
        VecNRf dp2; ///< drk / dz，索引为pattern num
        VecNRf dp3; ///< drk / dphi_x，索引为pattern num
        VecNRf dp4; ///< drk / dphi_y，索引为pattern num
        VecNRf dp5; ///< drk / dphi_z，索引为pattern num
        VecNRf dp6; ///< drk / daji，索引为pattern num
        VecNRf dp7; ///< drk / dbji，索引为pattern num
        VecNRf dd;  ///< drk / ddpi，索引为pattern num
        VecNRf r;   ///< rk，索引为pattern num

        /// 10*1 向量，用来缓存H矩阵和b矩阵的中间缓冲量的
        JbBuffer_new[i].setZero();

        bool isGood = true; ///< 用来判断point点是否为good
        float energy = 0;   ///< 用来存储点的能量值

        /// 遍历pi对应的一个pattern，将不同的内容放到dp0、dp1...中
        for (int idx = 0; idx < patternNum; idx++) {
            /// 获取pattern的坐标偏移
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];

            /// Pj' = Rji * K_inv * (pi + dx, pi + dy, 1) + tji * dpi --> Pj = Pj' / dpi，使用host点的逆深度
            Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;

            /// 归一化坐标 Pj_norm = (u, v)
            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];

            /// 像素坐标pj = (Ku, kv)
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;

            /// dpi/pz'--> 新一帧上的逆深度，1 / Pjz
            float new_idepth = point->idepth_new / pt[2];

            /// 判断条件一：若像素点落在边框为2的界内，或者逆深度为负，则认为point为坏点，直接跳出pattern的迭代
            if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0)) {
                isGood = false;
                break;
            }

            /// 得到j帧上的像素值、x方向梯度和y方向梯度，通过插值的方式得到,patch上的某个点，pj
            Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);

            /// 得到i帧上的像素值，同样使用插值的方式获得，patch上的某个点pi对应的像素值
            float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

            /// 判断条件二：像素值有穷, 要求i帧和j帧上的像素值是有穷的，否则认为是坏点
            if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
                isGood = false;
                break;
            }

            // 某个pattern中点的残差，rk = Ij[pj] - a * Ii[pi] - b，k ~ [0, pattern_num]
            float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];

            // Huber权重，根据残差大小确定huber权重，这里也可以使用huber核函数
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
            energy += hw * residual * residual * (2 - hw); ///< 某个pattern的energy（能量值）

            float dxdd = (t[0] - t[2] * u) / pt[2]; ///< pjx对逆深度求导中间量(tx - tz * xj' / zj') / zj'
            float dydd = (t[1] - t[2] * v) / pt[2]; ///< pjy对逆深度求导中间量(ty - tz * yj' / zj') / zj'

            if (hw < 1)
                hw = sqrtf(hw);

            float dxInterp = hw * hitColor[1] * fxl; ///< fx * dx * hw (带有核函数部分)
            float dyInterp = hw * hitColor[2] * fyl; ///< fy * dy * hw (带有核函数部分)

            /// 残差对 Tji左扰动 求导,
            dp0[idx] = new_idepth * dxInterp;                       ///< drk / dx, dpi/pz' * dxfx
            dp1[idx] = new_idepth * dyInterp;                       ///< drk / dy, dpi/pz' * dyfy
            dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp); ///< drk / dz, -dpi/pz' * (px'/pz'*dxfx + py'/pz'*dyfy)
            dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;  ///< drk / dphi_x, - px'py'/pz'^2*dxfy - (1+py'^2/pz'^2)*dyfy
            dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;   ///< drk / dphi_y, (1+px'^2/pz'^2)*dxfx + px'py'/pz'^2*dxfy
            dp5[idx] = -v * dxInterp + u * dyInterp;                ///< drk / dphi_z,  -py'/pz'*dxfx + px'/pz'*dyfy

            /// 残差对光度参数求导
            dp6[idx] = -hw * r2new_aff[0] * rlR; ///< rk / aji, exp(aj-ai)*I(pi)
            dp7[idx] = -hw * 1;                  ///< rk / bji,  对 b 导

            /// 残差对dpi求导
            dd[idx] = dxInterp * dxdd + dyInterp * dydd; ///< dxfx * 1/Pz * (tx - u*tz) +　dyfy * 1/Pz * (tx - u*tz)

            /// 残差res本身
            r[idx] = hw * residual; ///< 残差res本身

            /// 以pj对逆深度求导模长的逆作为maxstep，使用maxstep * delta_pj.norm --> delta_dpi，方便使用像素变化来控制逆深度变化
            /// 后续源码里面使用了0.25个pj像素移动来控制逆深度的更新step，即 0.25 * maxstep作为逆深度更新步长
            float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
            if (maxstep < point->maxstep)
                point->maxstep = maxstep;

            /// 中间变量的缓冲区，这部分主要是用于计算Hsc和bsc部分的内容
            JbBuffer_new[i][0] += dp0[idx] * dd[idx]; ///< sum drk / dx * drk / ddpi
            JbBuffer_new[i][1] += dp1[idx] * dd[idx]; ///< sum drk / dy * drk / ddpi
            JbBuffer_new[i][2] += dp2[idx] * dd[idx]; ///< sum drk / dz * drk / ddpi
            JbBuffer_new[i][3] += dp3[idx] * dd[idx]; ///< sum drk / dphi_x * drk / ddpi
            JbBuffer_new[i][4] += dp4[idx] * dd[idx]; ///< sum drk / dphi_y * drk / ddpi
            JbBuffer_new[i][5] += dp5[idx] * dd[idx]; ///< sum drk / dphi_z * drk / ddpi
            JbBuffer_new[i][6] += dp6[idx] * dd[idx]; ///< sum drk / daji * drk / ddpi
            JbBuffer_new[i][7] += dp7[idx] * dd[idx]; ///< sum drk / dbji * drk / ddpi
            JbBuffer_new[i][8] += r[idx] * dd[idx];   ///< sum rk * drk / ddpi --> pattern能量对逆深度的雅可比
            JbBuffer_new[i][9] += dd[idx] * dd[idx];  ///< sum drk / ddpi * drk / ddpi
        }

        /// 如果点的pattern(其中一个像素)超出图像或者像素值无穷, pattern的能量大于阈值
        if (!isGood || energy > point->outlierTH * 20) {
            /// 认为这个pi点对应的pattern在这次优化中不会起到作用，将该点之前的energy加到E内
            E.updateSingle((float)(point->energy[0]));

            point->isGood_new = false;         ///< 将isGood_new这个点设置为外点
            point->energy_new = point->energy; ///< 将energy_new的内容由之前的energy赋值
            continue;
        }

        /// 如果pattern中的点pi和pj没有发生意外，并且pattern的能量在阈值范围内
        E.updateSingle(energy);        ///< 将当前状态的点能量累加到E中
        point->isGood_new = true;      ///< 将点的isGood_new置为true，说明在当前迭代范围内是可用的
        point->energy_new[0] = energy; ///< 将由能量函数计算的能量部分赋值给energy_new[不带正则项部分]

        /// 使用drk / dx、drk / dy...，和intel芯片的SIMD的SSE指令集进行加速，计算pattern的Hc和bc
        for (int i = 0; i + 3 < patternNum; i += 4)
            acc9.updateSSE(_mm_load_ps(((float *)(&dp0)) + i), _mm_load_ps(((float *)(&dp1)) + i), _mm_load_ps(((float *)(&dp2)) + i),
                           _mm_load_ps(((float *)(&dp3)) + i), _mm_load_ps(((float *)(&dp4)) + i), _mm_load_ps(((float *)(&dp5)) + i),
                           _mm_load_ps(((float *)(&dp6)) + i), _mm_load_ps(((float *)(&dp7)) + i), _mm_load_ps(((float *)(&r)) + i));

        /// 当pattern num不是4的倍数时，使用float的形式，单独加进去，用来更新Hc和bc矩阵
        for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
            acc9.updateSingle((float)dp0[i], (float)dp1[i], (float)dp2[i], (float)dp3[i], (float)dp4[i], (float)dp5[i], (float)dp6[i], (float)dp7[i],
                              (float)r[i]);
    }

    E.finish();    ///< 用于汇总，防止大数吃小数
    acc9.finish(); ///< 用于汇总，防止大数吃小数

    for (int i = 0; i < npts; i++) {
        Pnt *point = ptsl + i;
        if (point->isGood_new) {
            /// 点的能量正则化的部分，在平移不足够的情况下，制定一个monocular的尺度，在优化的前提下，向1移动
            point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
        }
    }

    /// 平移部分的正则化项，用于判断后续是否已经达到足够的 tji 平移尺寸
    float alphaEnergy = alphaW * (refToNew.translation().squaredNorm() * npts);

    float alphaOpt; ///< alphaOpt用来判断是否达到了足够的tji平移尺寸
    if (alphaEnergy > alphaK * npts) {
        /// 如果tji达到一定尺寸后，将alphaOpt置为0，并且alphaEnergy置为一个常量
        alphaOpt = 0;
        alphaEnergy = alphaK * npts;
    } else {
        /// 如果tji没有达到要求，则将alphaOpt置为alphaW
        alphaOpt = alphaW;
    }

    /// schur部分的协方差矩阵计算 W(V^{-1})W^T
    acc9SC.initialize();
    for (int i = 0; i < npts; i++) {
        Pnt *point = ptsl + i;
        if (!point->isGood_new)
            continue;

        point->lastHessian_new = JbBuffer_new[i][9]; ///< i点的海瑟矩阵，赋值，这部分的Hdp不带正则化项的部分

        /// 当平移不够大时，针对不同的点对应的雅可比和海塞矩阵，加上dpi的正则化部分，当平移足够大时，alphaOpt置为0
        JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1); // E'(逆深度正则化)对dpi的雅可比部分
        JbBuffer_new[i][9] += alphaOpt;                           // E'(逆深度正则化)对dpi的海塞矩阵部分

        /// 当平移足够大时，变更E'正则化能量函数，并对变更后的雅可比和海塞矩阵加上变更项
        if (alphaOpt == 0) {
            JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
            JbBuffer_new[i][9] += couplingWeight;
        }

        /// 这里为了防止除0操作，加上了1来取能量对dpi的海瑟矩阵的逆，V的逆
        JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]);

        /// 计算Hsc和bsc，使用的是JbBuffer里面的内容
        acc9SC.updateSingleWeighted((float)JbBuffer_new[i][0], (float)JbBuffer_new[i][1], (float)JbBuffer_new[i][2], (float)JbBuffer_new[i][3],
                                    (float)JbBuffer_new[i][4], (float)JbBuffer_new[i][5], (float)JbBuffer_new[i][6], (float)JbBuffer_new[i][7],
                                    (float)JbBuffer_new[i][8], (float)JbBuffer_new[i][9]);
    }
    acc9SC.finish();

    H_out = acc9.H.topLeftCorner<8, 8>();       ///< H，并没有添加点的正则化部分，但是并没有关系，因为点的正则化不会改变H矩阵（非点部分）
    b_out = acc9.H.topRightCorner<8, 1>();      ///< b，并没有添加点的正则化部分，但是并没有关系，因为点的正则化不会改变b向量（非点部分）
    H_out_sc = acc9SC.H.topLeftCorner<8, 8>();  ///< Hsc，具有点的正则化部分内容，必须要有，因为Hsc的计算需要V
    b_out_sc = acc9SC.H.topRightCorner<8, 1>(); ///< bsc，具有点的正则化部分内容，必须要有，因为Hsc的计算需要V

    ///< 当平移不足时，需要将涉及平移部分的正则化E'的H加到H中
    H_out(0, 0) += alphaOpt * npts;
    H_out(1, 1) += alphaOpt * npts;
    H_out(2, 2) += alphaOpt * npts;

    ///< 当平移不足时，需要将平移部分的正则化E'的J加到b中
    Vec3f tlog = refToNew.log().head<3>().cast<float>(); // 李代数, 平移部分 (上一次的位姿值)
    b_out[0] += tlog[0] * alphaOpt * npts;
    b_out[1] += tlog[1] * alphaOpt * npts;
    b_out[2] += tlog[2] * alphaOpt * npts;

    // 能量值, 平移部分的正则化项（当位移不足时，与tji有关，当位移足够时，为一个常数） , 使用的点的个数
    return Vec3f(E.A, alphaEnergy, E.num);
}

float CoarseInitializer::rescale() {
    float factor = 20 * thisToNext.translation().norm();
    //	float factori = 1.0f/factor;
    //	float factori2 = factori*factori;
    //
    //	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
    //	{
    //		int npts = numPoints[lvl];
    //		Pnt* ptsl = points[lvl];
    //		for(int i=0;i<npts;i++)
    //		{
    //			ptsl[i].iR *= factor;
    //			ptsl[i].idepth_new *= factor;
    //			ptsl[i].lastHessian *= factori2;
    //		}
    //	}
    //	thisToNext.translation() *= factori;

    return factor;
}

/**
 * @brief 计算tji满足要求时，点的正则化能量部分
 *
 * @param lvl       输入的计算的金字塔层级
 * @return Vec3f    能量值，[E_old, E_new, n]
 */
Vec3f CoarseInitializer::calcEC(int lvl) {

    /// 如果tji不满足要求，则直接返回[0, 0, n]
    if (!snapped)
        return Vec3f(0, 0, numPoints[lvl]);

    AccumulatorX<2> E;
    E.initialize();
    int npts = numPoints[lvl];
    for (int i = 0; i < npts; i++) {
        Pnt *point = points[lvl] + i;
        if (!point->isGood_new)
            continue;
        float rOld = (point->idepth - point->iR);
        float rNew = (point->idepth_new - point->iR);
        E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew)); // 求和
    }
    E.finish();

    /// 如果tji满足要求，则将点的正则化部分对应的能量进行返回[E_old, E_new, n]
    return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
}

/**
 * @brief 使用邻近点加权平均的方式，更新点的期望状态
 * @details
 *  1. 遍历lvl层上的点pt
 *  2. 拿到pt的邻近点，且在优化过程中被认为是good的点 neiGoodPts
 *  3. 对neiGoodPts，求其期望深度iR的中位数 iRMedian
 *  4. 将pt的逆深度 dp 和中位数进行加权平均 newIR = (1 - weight) * dp + weight * iRMedian
 *  5. 将newIR更新到pt的iR上
 * @param lvl 输入的金字塔层级
 */
void CoarseInitializer::optReg(int lvl) {
    int npts = numPoints[lvl];
    Pnt *ptsl = points[lvl];

    /// 位移不足够则设置iR是1
    if (!snapped) {
        for (int i = 0; i < npts; i++)
            ptsl[i].iR = 1;
        return;
    }

    for (int i = 0; i < npts; i++) {
        Pnt *point = ptsl + i;
        if (!point->isGood)
            continue;

        float idnn[10];
        int nnn = 0;
        // 获得当前点周围最近10个点, 质量好的点的iR
        for (int j = 0; j < 10; j++) {
            if (point->neighbours[j] == -1)
                continue;
            Pnt *other = ptsl + point->neighbours[j];
            if (!other->isGood)
                continue;
            idnn[nnn] = other->iR;
            nnn++;
        }

        // 与最近点中位数进行加权获得新的iR
        if (nnn > 2) {
            std::nth_element(idnn, idnn + nnn / 2, idnn + nnn); // 获得中位数
            point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
        }
    }
}

//* 使用归一化积来更新高层逆深度值
void CoarseInitializer::propagateUp(int srcLvl) {
    assert(srcLvl + 1 < pyrLevelsUsed);

    int nptss = numPoints[srcLvl];     ///< 当前层点数目
    int nptst = numPoints[srcLvl + 1]; ///< 上一层点数目
    Pnt *ptss = points[srcLvl];
    Pnt *ptst = points[srcLvl + 1];

    // 遍历上一层的点，将点的iR和iRSumNum置0
    for (int i = 0; i < nptst; i++) {
        Pnt *parent = ptst + i;
        parent->iR = 0;
        parent->iRSumNum = 0;
    }

    /// 遍历当前层的点
    for (int i = 0; i < nptss; i++) {
        Pnt *point = ptss + i;
        if (!point->isGood)
            continue;

        Pnt *parent = ptst + point->parent;
        parent->iR += point->iR * point->lastHessian; /// 归一化积的 期望iR更新部分
        parent->iRSumNum += point->lastHessian;       /// 归一化积的 信息矩阵更新部分
    }

    /// 遍历上一层的点parent，使用高斯归一化积更新逆深度idepth、iR，并将点置为good
    for (int i = 0; i < nptst; i++) {
        Pnt *parent = ptst + i;
        if (parent->iRSumNum > 0) {
            parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
            parent->isGood = true;
        }
    }

    optReg(srcLvl + 1); ///< 对于期望部分，还需要考虑邻居之间的平滑关系
}

/**
 * @brief 使用上一层srcLvl的金字塔的点期望来初始化当前层srcLvl - 1的点期望
 * @details
 *  1. 遍历当前层的点point，找到point的上层点parent
 *  2. 如果parent是坏点，‘或者’ parent的lastHessian 小于0.1，则放弃point点的期望iR的更新
 *  3. 如果point点是坏点，但parent点满足2中的要求，则认为当前point为之前的优化偏差
 *      3.1 将point的isGood置为true
 *      3.2 将point的iR、idepth和idepth_new都置为parent的期望iR
 *      3.3 将point的对应的lastHessian置为0，放弃之前优化的先验，因为认定之前point点的优化是错误的
 *  4. 如果point点是good点
 *      4.1 假设 point 的 iR ~ N(point.iR, (point.lastHessian * 2)^-1)
 *      4.2 假设 parent的 iR ~ N(parent.iR, (parent.lastHessian)^-1)))
 *      4.3 使用高斯归一化积的形式，进行更新后的iR求解
 *  5. 在考虑到point点的parent点后，还需要考虑neighbors的平滑关系，使用optReg的方式，对iR进行再一次的更新 @see optReg
 * @param srcLvl 待初始化点状态的金字塔层级的上一层
 */
void CoarseInitializer::propagateDown(int srcLvl) {
    assert(srcLvl > 0);
    // set idepth of target

    int nptst = numPoints[srcLvl - 1]; ///< 当前层的点数目
    Pnt *ptss = points[srcLvl];        ///< 上一层的点集
    Pnt *ptst = points[srcLvl - 1];    ///< 当前层点集

    for (int i = 0; i < nptst; i++) {
        Pnt *point = ptst + i;              ///< 遍历当前层的点
        Pnt *parent = ptss + point->parent; ///< 找到当前点的 parent 点（在上一层）

        /// 在最小二乘优化中，海塞矩阵可以代表参数的信息矩阵，也就是说海塞矩阵越大，待优化的参数越稳定
        if (!parent->isGood || parent->lastHessian < 0.1) ///< parent 点非 good ，或者海塞矩阵小于0.1，则认为点的状态不可信
            continue;

        if (!point->isGood) {
            /// 这里的point一直是i帧的pi，因此时来一个j帧就会估计一次pi，当父点可信时，即便之前估计的点为not good ，这里也会继续尝试
            point->iR = point->idepth = point->idepth_new = parent->iR;
            point->isGood = true;
            point->lastHessian = 0;
        } else {
            // 通过hessian给point和parent加权求得新的iR
            // iR可以看做是深度的值, 使用的高斯归一化积, Hessian是信息矩阵
            float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) / (point->lastHessian * 2 + parent->lastHessian);
            point->iR = point->idepth = point->idepth_new = newiR;
        }
    }

    /// 使用归一化积更新iR之后，会考虑邻近点之间的连续性，使用邻近点的iR再次进行iR的更新
    optReg(srcLvl - 1);
}

//* 低层计算高层, 像素值和梯度
void CoarseInitializer::makeGradients(Eigen::Vector3f **data) {
    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
        int lvlm1 = lvl - 1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        Eigen::Vector3f *dINew_l = data[lvl];
        Eigen::Vector3f *dINew_lm = data[lvlm1];
        // 使用上一层得到当前层的值
        for (int y = 0; y < hl; y++)
            for (int x = 0; x < wl; x++)
                dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] + dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                  dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] + dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
        // 根据像素计算梯度
        for (int idx = wl; idx < wl * (hl - 1); idx++) {
            dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
            dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
        }
    }
}

/**
 * @brief 初始化器，设置第一帧函数
 * @details
 *  1. 计算金字塔上，每一层的相机内参fx,fy,cx,cy,w,h @see CoarseInitializer::makeK
 *  2. 使用像素选择器，对金字塔层级上的0层上的点进行选择 @see PixelSelector::makeMaps
 *  3. 使用像素选择器，对金字塔层级上的非0层上的点进行选择 @see PixelSelector::makePixelStatus
 *  4. 考虑parttern带来的边界问题，忽略掉金字塔各层上选择了边界上的点
 *  5. 对金字塔的每一层上选择的每一个点，计算其的10个同层上的neighbour和上层的1个parent @see CoarseInitializer::makeNN
 *  6. 初始化CoarseInitializer的状态
 * @param HCalib            为第一步提供0层上的初始内参
 * @param newFrameHessian   输入的初始化器的第一帧
 */
void CoarseInitializer::setFirst(CalibHessian *HCalib, FrameHessian *newFrameHessian) {
    /// 计算金字塔图像每层的内参
    makeK(HCalib);
    firstFrame = newFrameHessian;

    /// 构建像素选择器
    PixelSelector sel(w[0], h[0]);

    /// 第0层的点选择状态，涉及到的内容有0、1、2、4分别代表没选择、第0层选择、第一层选择和第二层选择
    float *statusMap = new float[w[0] * h[0]];

    /// 第1层到最高层的点的选择状态，涉及到的内容只有true和false，true为选择、false为不选择
    bool *statusMapB = new bool[w[0] * h[0]];

    /// 不同层之间的理想的点选密度，即wantNum = density * w[level] * h[level]
    float densities[] = {0.03, 0.05, 0.15, 0.5, 1};

    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        /// 设置像素选择器的第0层pot为3，后续会在makeMaps里面动态变化
        sel.currentPotential = 3;
        int npts;
        if (lvl == 0) {
            /// 第0层的选择，策略比较复杂，涉及到金字塔中的0,1,2层， @see makeMaps
            npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
        } else {
            /// 除0层以外的其他层的选择，策略相对0层来说，简单， @see makePixelStatus
            npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);
        }

        /// 去除之前调用状态，并开辟地图点空间
        if (points[lvl] != 0)
            delete[] points[lvl];
        points[lvl] = new Pnt[npts];

        /// 将选择的点进行逆深度的初始化，将你深度都初始化为1.0
        int wl = w[lvl], hl = h[lvl]; ///< 每一层的图像大小
        Pnt *pl = points[lvl];        ///< 每一层上的点

        /// 每层上实际被选择的点，因为计算能量函数时，会有不同pattern，代表不同的border
        /// 因此在border内的被选择像素是会被抛弃掉的
        int nl = 0; ///< 抛弃border后，实际每层上被选择的点的个数

        /// 在选出的像素中, 添加点信息
        for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
            for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {

                if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) {
                    pl[nl].u = x + 0.1;         ///< 点的u值，x方向上
                    pl[nl].v = y + 0.1;         ///< 点的v值，y方向上
                    pl[nl].idepth = 1;          ///< 点的逆深度，初始化为1
                    pl[nl].iR = 1;              ///< 点的逆深度期望，会逐步更新
                    pl[nl].isGood = true;       ///< 这个状态是针对不同帧做变化的，投影后合理为good
                    pl[nl].energy.setZero();    ///< 光度误差和正则化
                    pl[nl].lastHessian = 0;     ///< 逆深度的Hessian矩阵
                    pl[nl].lastHessian_new = 0; ///< 新一次迭代的点的Hessian矩阵

                    ///< 点的类型，仅第0层上的选点有区分，1、2、4，其余层皆为1
                    pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

                    /// 设定点被判断为外点的阈值
                    pl[nl].outlierTH = patternNum * setting_outlierTH;

                    nl++;
                    assert(nl <= npts);
                }
            }
        // 点的数目, 去掉了在边界上的点
        numPoints[lvl] = nl;
    }
    delete[] statusMap;
    delete[] statusMapB;

    /// 对金字塔的每一层上选择的每一个点，计算其的10个同层上的neighbour和上层的1个parent @see makeNN
    makeNN();

    thisToNext = SE3(); ///< 初始化器中，参考帧和当前帧之间的相对位姿
    snapped = false;    ///< 是否收敛
    frameID = 0;        ///< 加入到初始化器的帧数
    snappedAt = 0;      ///< 在第几帧进行收敛

    for (int i = 0; i < pyrLevelsUsed; i++)
        dGrads[i].setZero(); //! what is dGrads;
}

/**
 * @brief 对lvl层级上的点，进行energy、idepth_new的重置
 * @details
 *  1. 遍历lvl层级上的点pts
 *  2. 将pts[i]的能量部分置空，并将当前点的idepth设置为待优化的初始值idepth_new
 *  3. 在非顶层上，使用 @see propagateDown 函数来更新 iR，而在顶层部分，使用的是顶层上的邻近点平均值作为期望iR（这里同样可以使用中位数代替）
 * @param lvl 输入的重置的点的金字塔层级
 */
void CoarseInitializer::resetPoints(int lvl) {
    Pnt *pts = points[lvl];
    int npts = numPoints[lvl];
    for (int i = 0; i < npts; i++) {
        pts[i].energy.setZero();           ///< 将点的能量重置为0，分别为残差平方项和正则化项两部分的内容
        pts[i].idepth_new = pts[i].idepth; ///< 以idepth_new作为优化初始值，拷贝优化之后的逆深度

        /// 如果是最顶层, 且点为非good点
        if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood) {
            float snd = 0, sn = 0;

            /// 统计距离pts最近的neighours点，且为good点
            for (int n = 0; n < 10; n++) {
                if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood)
                    continue;
                snd += pts[pts[i].neighbours[n]].iR;
                sn += 1;
            }

            /// 将统计的结果的期望平均值赋值给pts
            if (sn > 0) {
                pts[i].isGood = true;
                pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
            }
        }
    }
}

/**
 * @brief 更新点的逆深度，使用pj像素限制方式
 * @details
 *  1. 由于V * (1 + lambda) 矩阵为一个主对角线矩阵，因此可以遍历每个点进行计算增量
 *  2. 跳过那些在计算中被认定为外点的pattern点
 *  3. 使用inc和b + W[i]^T * inc，来更新b
 *  4. 使用maxPixelStep来计算逆深度更新的最大步长maxstep
 *  5. 将step 更新到 点的逆深度中
 *  6. 并要求点的逆深度在(1e-3, 50)，即深度在(0.02m, 1000m)范围内
 * @param lvl       待更新逆深度的金字塔层级
 * @param lambda    LM方法的lambda参数，用来计算LM方法的V矩阵 V * (1 + lambda)
 * @param inc       Tji，aji和bji的更新量，用来简化计算
 */
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc) {

    const float maxPixelStep = 0.25;
    const float idMaxStep = 1e10;
    Pnt *pts = points[lvl];
    int npts = numPoints[lvl];

    /// 逆深度对应的H矩阵为稀疏矩阵，因此可以一个一个点进行逆深度的求解
    for (int i = 0; i < npts; i++) {

        /// 跳过那些非good的点
        if (!pts[i].isGood)
            continue;

        /// 将逆深度对应的-b和位姿部分的-bp + W[i]^T * delta_xc
        float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
        float step = -b * JbBuffer[i][9] / (1 + lambda); ///< 求出每个点对应的逆深度增量

        /// 使用0.25的像素步长限制逆深度的更新幅度，使得逆深度的更新反映在pj上不会移动0.25个像素
        float maxstep = maxPixelStep * pts[i].maxstep;
        if (maxstep > idMaxStep)
            maxstep = idMaxStep;

        if (step > maxstep)
            step = maxstep;
        if (step < -maxstep)
            step = -maxstep;

        /// 更新得到新的逆深度，要求新的逆深度在(1e-3, 50)的内 --> 深度在(0.02m, 1000m)范围内
        float newIdepth = pts[i].idepth + step;
        if (newIdepth < 1e-3)
            newIdepth = 1e-3;
        if (newIdepth > 50)
            newIdepth = 50;
        pts[i].idepth_new = newIdepth;
    }
}

/**
 * @brief 对某lvl层金字塔上的点进行更新
 * @details
 *  1. 针对之前迭代放弃的点，将其idepth、idepth_new和iR
 * @param lvl 金字塔层级
 */
void CoarseInitializer::applyStep(int lvl) {
    Pnt *pts = points[lvl];
    int npts = numPoints[lvl];
    for (int i = 0; i < npts; i++) {
        if (!pts[i].isGood) {
            pts[i].idepth = pts[i].idepth_new = pts[i].iR;
            continue;
        }
        pts[i].energy = pts[i].energy_new;
        pts[i].isGood = pts[i].isGood_new;
        pts[i].idepth = pts[i].idepth_new;
        pts[i].lastHessian = pts[i].lastHessian_new;
    }
    std::swap<Vec10f *>(JbBuffer, JbBuffer_new);
}

/**
 * @brief 计算每层金字塔图像的内参pinhole
 * @details
 * 	1. 计算金字塔图像上的w和h，使用>>运算符的操作，直接舍去小数位
 *  2. 计算金字塔图像上的fx，fy，使用逐级缩小0.5的方式
 * 	3. 计算金字塔图像上的cx，cy，使用先加0.5再减0.5的方式
 * 		3.1 在缩小之前，像素的坐标是由像素左上角的部分确定的
 * 		3.2 为了保证像素中心坐标代表像素，则需要先+0.5
 * 		3.3 在这个基础上，进行缩小层级倍
 * 		3.4 最后为了保持图像坐标系中像素的左上角代表像素的坐标，需要再减去0.5
 * @note 计算cx和cy部分比较考究！
 * @param HCalib 用于提供第零层的w,h,fx,fy,cx,cy
 */
void CoarseInitializer::makeK(CalibHessian *HCalib) {
    w[0] = wG[0];
    h[0] = hG[0];

    fx[0] = HCalib->fxl();
    fy[0] = HCalib->fyl();
    cx[0] = HCalib->cxl();
    cy[0] = HCalib->cyl();
    /// 求各层的K参数
    for (int level = 1; level < pyrLevelsUsed; ++level) {
        /// 使用>>右移运算符，可以舍去0.5的部分
        w[level] = w[0] >> level;
        h[level] = h[0] >> level;
        fx[level] = fx[level - 1] * 0.5;
        fy[level] = fy[level - 1] * 0.5;

        //! cx部分的设置，比较讲究，先+0.5，然后缩小完之后，再减去0.5
        cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
        cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
    }

    /// 求K_inverse参数
    for (int level = 0; level < pyrLevelsUsed; ++level) {
        K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
        Ki[level] = K[level].inverse();
        fxi[level] = Ki[level](0, 0);
        fyi[level] = Ki[level](1, 1);
        cxi[level] = Ki[level](0, 2);
        cyi[level] = Ki[level](1, 2);
    }
}

/**
 * @brief 对每一层的每一个点，在同层内找到10个neighbor，在其上一层最近邻找到一个parent
 * @details
 *  1. 对不同层金字塔，构建KDTree
 *  2. 对某一层中的某一个点，使用K近邻搜索，找到同层内的10个neighbor
 *  3. 对某一层中的某一个点，使用仿射变换，计算该点在上层的像素坐标，使用最近邻搜索，找到上层的parent
 * @note 由于某层的某个点的parent是在其上一层中找到的，因此最高层没有parent，只有neighbor
 */
void CoarseInitializer::makeNN() {
    const float NNDistFactor = 0.05;

    /// pcs定义的为金字塔层上的点云
    FLANNPointcloud pcs[PYR_LEVELS];

    /// indexes定义的为每层上点云构建的kdtree
    KDTree *indexes[PYR_LEVELS];

    /// 构建金字塔层上的二维点云和对应的KDTree
    for (int i = 0; i < pyrLevelsUsed; i++) {
        pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
        indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
        indexes[i]->buildIndex();
    }

    const int nn = 10;

    /// 对金字塔层级进行遍历
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];

        int ret_index[nn];  ///< 最近邻的点的索引
        float ret_dist[nn]; ///< 最近邻的点的距离

        nanoflann::KNNResultSet<float, int, int> resultSet(nn); ///< 10个最近邻
        nanoflann::KNNResultSet<float, int, int> resultSet1(1); ///< 1个最近邻

        for (int i = 0; i < npts; i++) {

            resultSet.init(ret_index, ret_dist);
            Vec2f pt = Vec2f(pts[i].u, pts[i].v);

            indexes[lvl]->findNeighbors(resultSet, (float *)&pt, nanoflann::SearchParams());
            int myidx = 0;
            float sumDF = 0;

            /// Pnt的neighboursDist，采用的是e ^ (-0.05 * d) / sum * 10，归一化为[0, 10]
            /// Pnt的neighbours，代表的是本层内10个最近点索引
            for (int k = 0; k < nn; k++) {
                pts[i].neighbours[k] = ret_index[k];
                float df = expf(-ret_dist[k] * NNDistFactor);
                sumDF += df;
                pts[i].neighboursDist[k] = df;
            }
            for (int k = 0; k < nn; k++)
                pts[i].neighboursDist[k] *= 10 / sumDF;

            /// 除最高层外选择的点，其余层点会使用仿射变换 + 最近邻搜索的方式，进行上一层的父点选择
            if (lvl < pyrLevelsUsed - 1) {
                resultSet1.init(ret_index, ret_dist);
                pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
                indexes[lvl + 1]->findNeighbors(resultSet1, (float *)&pt, nanoflann::SearchParams());

                pts[i].parent = ret_index[0];
                pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor);

            } else {
                pts[i].parent = -1;
                pts[i].parentDist = -1;
            }
        }
    }

    for (int i = 0; i < pyrLevelsUsed; i++)
        delete indexes[i];
}
} // namespace dso
