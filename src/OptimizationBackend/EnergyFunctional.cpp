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

#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

bool EFAdjointsValid = false; //!< 是否设置状态伴随矩阵
bool EFIndicesValid = false;  //!< 是否设置frame, point, res的ID
bool EFDeltaValid = false;    //!< 是否设置状态增量值

/**
 * @brief 预先计算线性化点处的dlocal / dglobal --> 以供后续计算出的 delta_local 转换为 delta_global
 * @details
 *  1. 位姿部分，使用伴随矩阵的性质，构建 local --> global 部分的雅可比矩阵
 *  2. 光度参数部分，直接根据构建的残差部分，计算 dlocal / dglobal
 *  3. 针对滑窗内的某对 host target 构建的残差 --> 维护线性化点处的 dlocal / dglobal_h 和 dlocal / dglobal_t
 * @param Hcalib 参数貌似没起作用
 */
void EnergyFunctional::setAdjointsF(CalibHessian *Hcalib) {

    if (adHost != 0)
        delete[] adHost;
    if (adTarget != 0)
        delete[] adTarget;
    adHost = new Mat88[nFrames * nFrames];
    adTarget = new Mat88[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)     ///< host frame
        for (int t = 0; t < nFrames; t++) ///< target frame
        {
            FrameHessian *host = frames[h]->data;
            FrameHessian *target = frames[t]->data;

            /// 得到Tth --> 线性化点处的
            SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

            Mat88 AH = Mat88::Identity(); ///< host delta --> global 到 local 的转换
            Mat88 AT = Mat88::Identity(); ///< target delta --> global 到 local 的转换

            /// local --> global 的转换过程（位姿部分） https://www.cnblogs.com/JingeTU/p/9077372.html
            //? 这里为什么会出现转置呢，结果好像没有这个转置吧
            AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose(); ///< delta_th / delta_h
            AT.topLeftCorner<6, 6>() = Mat66::Identity();               ///< delta_th / delta_t

            /// -ath = tt*exp(at) / th*exp(ah),  -bth = bt - tt*exp(at) / th*exp(ah) * bh
            Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
            AT(6, 6) = -affLL[0]; ///< d(-ath) / dat
            AH(6, 6) = affLL[0];  ///< d(-ath) / dah
            AT(7, 7) = -1;        ///< d(-bth) / dbt
            AH(7, 7) = affLL[0];  ///< d(-bth) / dbh

            /// 右乘 缩放的雅可比矩阵
            AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
            AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
            AH.block<1, 8>(6, 0) *= SCALE_A;
            AH.block<1, 8>(7, 0) *= SCALE_B;
            AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
            AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
            AT.block<1, 8>(6, 0) *= SCALE_A;
            AT.block<1, 8>(7, 0) *= SCALE_B;

            adHost[h + t * nFrames] = AH;
            adTarget[h + t * nFrames] = AT;
        }

    /// 构建一个全是setting_initialCalibHessian的浮点数向量，用做内参的海塞矩阵先验
    cPrior = VecC::Constant(setting_initialCalibHessian);

    /// 同时维护一个float型的 线性化点处的伴随矩阵
    if (adHostF != 0)
        delete[] adHostF;
    if (adTargetF != 0)
        delete[] adTargetF;
    adHostF = new Mat88f[nFrames * nFrames];
    adTargetF = new Mat88f[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++) {
            adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
            adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
        }

    /// 同时维护一个float型的相机内参先验
    cPriorF = cPrior.cast<float>();

    EFAdjointsValid = true; ///< 标志目前滑窗内的伴随矩阵构建完成（线性化点处）
}

EnergyFunctional::EnergyFunctional() {
    adHost = 0;
    adTarget = 0;

    red = 0;

    adHostF = 0;
    adTargetF = 0;
    adHTdeltaF = 0;

    nFrames = nResiduals = nPoints = 0;

    HM = MatXX::Zero(CPARS, CPARS); // 初始的, 后面增加frame改变
    bM = VecX::Zero(CPARS);

    accSSE_top_L = new AccumulatedTopHessianSSE();
    accSSE_top_A = new AccumulatedTopHessianSSE();
    accSSE_bot = new AccumulatedSCHessianSSE();

    resInA = resInL = resInM = 0;
    currentLambda = 0;
}
EnergyFunctional::~EnergyFunctional() {
    for (EFFrame *f : frames) {
        for (EFPoint *p : f->points) {
            for (EFResidual *r : p->residualsAll) {
                r->data->efResidual = 0;
                delete r;
            }
            p->data->efPoint = 0;
            delete p;
        }
        f->data->efFrame = 0;
        delete f;
    }

    if (adHost != 0)
        delete[] adHost;
    if (adTarget != 0)
        delete[] adTarget;

    if (adHostF != 0)
        delete[] adHostF;
    if (adTargetF != 0)
        delete[] adTargetF;
    if (adHTdeltaF != 0)
        delete[] adHTdeltaF;

    delete accSSE_top_L;
    delete accSSE_top_A;
    delete accSSE_bot;
}

/**
 * @brief 计算各种的增量 帧相对位姿增量，相机内参增量，帧状态相对于线性化点增量和相对于先验增量，点的逆深度相对线性化点增量
 * @details
 *
 * @note 帧状态的相对增量，需要根据 dlocal / dhost 和 dlocal / dtarget进行计算
 *
 * @param HCalib 计算相机的内参增量
 */
void EnergyFunctional::setDeltaF(CalibHessian *HCalib) {
    if (adHTdeltaF != 0)
        delete[] adHTdeltaF;
    adHTdeltaF = new Mat18f[nFrames * nFrames];

    /// 计算滑窗内，两帧之间的相对增量 delta_local = dlocal / dhost * delta_host + dloacl / dtarget * delta_target
    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++) {
            int idx = h + t * nFrames;
            adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx] +
                              frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
        }

    /// 计算滑窗内，相机的内参增量
    cDeltaF = HCalib->value_minus_value_zero.cast<float>();

    /// 计算帧状态距离线性化点的绝对增量 和 帧状态距离先验的绝对增量
    for (EFFrame *f : frames) {
        f->delta = f->data->get_state_minus_stateZero().head<8>();
        f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

        /// 计算点的逆深度状态 到线性化点处的逆深度增量
        for (EFPoint *p : f->points)
            p->deltaF = p->data->idepth - p->data->idepth_zero;
    }

    EFDeltaValid = true;
}

/**
 * @brief 构建滑窗对应的H矩阵和b矩阵，考虑先验，不考虑HM，bM，逆深度点
 * @details
 *  1. 初始化累加器 @see AccumulatedTopHessianSSE::setZero
 *  2. 向累加器中，添加点信息 --> 构建了相对量之间的H矩阵和b矩阵 @see AccumulatedTopHessianSSE::addPoint
 *  3. 根据残差信息，构建滑窗对应的H矩阵和b矩阵，考虑先验，不考虑HM，bM和点的先验 @see AccumulatedTopHessianSSE::stitchDoubleMT
 *
 * @param H     输出的H矩阵
 * @param b     输出的b矩阵
 * @param MT    输入的是否使用多线程的标志
 */
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) {
    if (MT) {
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>, accSSE_top_A, &allPoints, this, _1, _2, _3, _4), 0, allPoints.size(), 50);
        accSSE_top_A->stitchDoubleMT(red, H, b, this, true, true);
        resInA = accSSE_top_A->nres[0];
    }

    /// 非多线程状态
    else {
        accSSE_top_A->setZero(nFrames);
        for (EFFrame *f : frames)
            for (EFPoint *p : f->points)
                accSSE_top_A->addPoint<0>(p, this);

        accSSE_top_A->stitchDoubleMT(red, H, b, this, true, false); /// 加先验, 得到H, b
        resInA = accSSE_top_A->nres[0];                             /// 所有残差计数
    }
}

/**
 * @brief 对线性化的部分的残差，构建绝对的H矩阵和b矩阵
 *
 * @see AccumulatedTopHessianSSE::addPoint<1> --> 计算的FEJ条件下的rk变换和 '相对'的H矩阵和b矩阵
 * @see AccumulatedTopHessianSSE::stitchDoubleMT --> 将相对的H矩阵和b矩阵转换为绝对量，有可能加先验
 *
 * @param H     输出的线性化残差部分的H矩阵
 * @param b     输出的线性化残差部分的b矩阵
 * @param MT    多线程标识
 */
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT) {
    if (MT) {
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>, accSSE_top_L, &allPoints, this, _1, _2, _3, _4), 0, allPoints.size(), 50);
        accSSE_top_L->stitchDoubleMT(red, H, b, this, true, true);
        resInL = accSSE_top_L->nres[0];
    } else {
        /// 初始化累加器
        accSSE_top_L->setZero(nFrames);
        for (EFFrame *f : frames)
            for (EFPoint *p : f->points)
                accSSE_top_L->addPoint<1>(p, this);                 ///< 针对滑窗中线性化和激活的残差，构建相对的H和b
        accSSE_top_L->stitchDoubleMT(red, H, b, this, true, false); ///< 将相对的H矩阵和b矩阵 转化为 绝对的H矩阵和b矩阵，不包含逆深度部分的转换
        resInL = accSSE_top_L->nres[0];
    }
}

/**
 * @brief 计算舒尔补中被 边缘化部分的Hsc和bsc矩阵 --> 将H矩阵中，逆深度的部分边缘化掉，方便求解
 * 1. 初始化边缘化累加器 @see AccumulatedSCHessianSSE::setZero
 * 2. 向累加器中添加点 --> Hfc部分，使用th方式拆分，Hff部分，使用h方式拆分 @see AccumulatedSCHessianSSE::addPoint
 * 3. 将相对的Hsc和bsc部分进行相对到绝对的转换，得到滑窗系统的Hsc和bsc @see AccumulatedSCHessianSSE::stitchDoubleMT
 *
 * @param H     输出的边缘化后的H矩阵
 * @param b     输出的边缘化后的b矩阵
 * @param MT    输入的是否使用多线程的标志
 */
void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT) {
    if (MT) {
        red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal, accSSE_bot, &allPoints, true, _1, _2, _3, _4), 0, allPoints.size(), 50);
        accSSE_bot->stitchDoubleMT(red, H, b, this, true);
    } else {
        accSSE_bot->setZero(nFrames);
        for (EFFrame *f : frames)
            for (EFPoint *p : f->points)
                accSSE_bot->addPoint(p, true);
        accSSE_bot->stitchDoubleMT(red, H, b, this, false);
    }
}

/**
 * @brief 计算逆深度增量，更新帧状态增量，更新相机内参状态增量
 * @details
 *  1. 计算逆深度增量的准备工作
 *      1.1 获取相机内参的增量 --> cstep
 *      1.2 获取帧状态的相对增量 --> (dlocal / dglobal_host) * delta_host + (dlocal / dglobal_target) * delta_target --> xAd
 *  2. 根据获得的中间变量cstep 和 xAd 计算点的逆深度增量 @see EnergyFunctional::resubstituteFPt
 *  3. 更新帧状态增量，前8个分别为6维位姿和2维光度参数，最后两维设置为0
 *  4. 更新相机内参状态增量
 *
 * @param x         输入的滑窗中，帧状态增量和相机内参增量
 * @param HCalib    输入的相机内参
 * @param MT        输入的多线程标识
 */
void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT) {
    assert(x.size() == CPARS + nFrames * 8);

    VecXf xF = x.cast<float>();
    HCalib->step = -x.head<CPARS>();

    /// 获取相机内参的增量 --> cstep
    Mat18f *xAd = new Mat18f[nFrames * nFrames];
    VecCf cstep = xF.head<CPARS>();

    /// 获取帧状态的相对增量
    for (EFFrame *h : frames) {
        h->data->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx);
        h->data->step.tail<2>().setZero();

        for (EFFrame *t : frames)
            xAd[nFrames * h->idx + t->idx] = xF.segment<8>(CPARS + 8 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx] +
                                             xF.segment<8>(CPARS + 8 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
    }

    /// 根据获得的中间变量cstep 和 xAd 计算点的逆深度增量
    if (MT)
        red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt, this, cstep, xAd, _1, _2, _3, _4), 0, allPoints.size(), 50);
    else
        resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

    delete[] xAd;
}

/**
 * @brief 计算点的逆深度增量
 * @details
 *  1. Hdf * delta_f + Hdc * delta_C + Hdd * delta_d = b_d
 *  2. b_d = b_d - Hdc * delta_C --> 减去相机内参部分
 *  3. b_d = b_d - Hdd * delta_d --> 减去帧参数部分（相对）
 *  4. delta_d = b_d / Hdd
 * @param xc    输入的相机内参的增量
 * @param xAd   根据绝对位姿和绝对仿射参数，获得相对的位置和相对仿射参数 --> 相对增量
 * @param min   多线程相关
 * @param max   多线程相关
 * @param stats 输出的多线程状态
 * @param tid   多线程相关
 */
void EnergyFunctional::resubstituteFPt(const VecCf &xc, Mat18f *xAd, int min, int max, Vec10 *stats, int tid) {
    for (int k = min; k < max; k++) {
        EFPoint *p = allPoints[k];

        /// 判断点中好残差的数量
        int ngoodres = 0;
        for (EFResidual *r : p->residualsAll)
            if (r->isActive())
                ngoodres++;

        /// 如果点中没有好残差，则认定该点无效，不进行更新 --> 但是没有认定其为外点
        if (ngoodres == 0) {
            p->data->step = 0;
            continue;
        }

        float b = p->bdSumF;
        b -= xc.dot(p->Hcd_accAF); ///< 减去逆深度和内参 部分

        for (EFResidual *r : p->residualsAll) {
            if (!r->isActive())
                continue;

            /// 减去 逆深度和光度参数 和 逆深度和位姿 部分
            b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
        }

        /// 获取逆深度增量
        p->data->step = -b * p->HdiF;
        assert(std::isfinite(p->data->step));
    }
}

/**
 * @brief 求解HM和bM对应的能量值
 *
 * @note 在解优化问题的时候，bM会根据线性化的更新量来改变，我认为这里的边缘化对应的能量值是否应该也更新bM
 * @return double   输出HM和bM对应的能量值 --> 泰勒展开 --> delta_state^T * HM * delta_state + 2 * bM * delta_state
 */
double EnergyFunctional::calcMEnergyF() {

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    VecX delta = getStitchedDeltaF();
    return delta.dot(2 * bM + HM * delta);
}

/**
 * @brief 计算当前滑动窗口中，所有已经线性化后的残差的能量值
 * @details
 *  1. 筛选，所有残差中，已经激活且已经线性化的残差
 *  2. 根据线性化的 dpj / dstate * delta_state --> delta_pj，FEJ获取delta_pj
 *  3. 根据delta_pj, delta_aji, delta_bji 计算获得delta_rk
 *  4. 获取，由状态变化而得到的 '线性化的' 残差能量值 --> 不包含线性化点处的
 *
 * @param min   多线程相关
 * @param max   多线程相关
 * @param stats 多线程状态
 * @param tid   多线程相关
 */
void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10 *stats, int tid) {

    Accumulator11 E;
    E.initialize();
    VecCf dc = cDeltaF;

    for (int i = min; i < max; i++) {
        EFPoint *p = allPoints[i];
        float dd = p->deltaF;

        /// 这里计算的残差能量，不包含新加入的残差
        for (EFResidual *r : p->residualsAll) {
            /// 这里正常来讲，应该没有激活且线性化的残差，也就说滑窗中的所有残差应该都是满足这个条件的，后续的内容不会走啊
            if (!r->isLinearized || !r->isActive())
                continue;

            Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
            RawResidualJacobian *rJ = r->J;

            float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>()) + rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd; ///< delta_xj
            float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>()) + rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd; ///< delta_yj

            __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
            __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
            __m128 delta_a = _mm_set1_ps((float)(dp[6]));
            __m128 delta_b = _mm_set1_ps((float)(dp[7]));

            for (int i = 0; i + 3 < patternNum; i += 4) {
                /// 计算delta_rk = drk / dpj * delta_pj + drk / daji * delta_aji + drk / dbji * delta_bji
                __m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx)) + i), Jp_delta_x);
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx + 1)) + i), Jp_delta_y));
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF)) + i), delta_a));
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF + 1)) + i), delta_b));

                /// 获取线性化点处的残差
                __m128 r0 = _mm_load_ps(((float *)&r->res_toZeroF) + i);

                /// 求得残差对应的能量值，去掉了 r0 ^ 2，因为常数量没有意义
                r0 = _mm_add_ps(r0, r0);
                r0 = _mm_add_ps(r0, Jdelta);
                Jdelta = _mm_mul_ps(Jdelta, r0);
                E.updateSSENoShift(Jdelta);
            }
            /// 128位对齐, 多出来部分
            for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) //* %4 的余数
            {
                float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 + rJ->JabF[0][i] * dp[6] + rJ->JabF[1][i] * dp[7];
                E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
            }
        }
        E.updateSingle(p->deltaF * p->deltaF * p->priorF); // 逆深度先验
    }
    E.finish();
    (*stats)[0] += E.A;
}

/**
 * @brief 计算去除线性化点处残差能量的系统能量值 --> 可以叫做系统能量值，应为线性化点并不改变（线性化点处的能量值是常量）
 * @details
 *  1. 计算 帧先验
 *  2. 计算 相机内参先验
 *  3. 计算 以线性化点处的雅可比 近似计算 delta_r，并计算去除线性化点处的残差能量值
 *  4. 计算 点逆深度的先验
 * @note 在计算系统残差过程中，由于使用了FEJ，需要使用线性化点处的雅可比近似，和欧拉积分的方式计算当前残差部分贡献的能量值 --> 会导致线性化误差
 * @return double 输出的系统的能量值
 */
double EnergyFunctional::calcLEnergyF_MT() {
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    double E = 0;

    /// 1. 计算 frame 帧的先验部分对应的能量 0.5 * (x - x_prior)^T * sigma * (x - x_prior)
    for (EFFrame *f : frames)
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

    /// 2. 计算 K 相机内参先验部分对应的能量 0.5 * (x - x_prior)^T * sigma * (x - x_prior)
    E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

    /// 3.1 以线性化点处为基准，计算 (rf + delta_r)^2 - rf^2 ---> 残差能量相对于线性化点处的变化量
    /// 3.2 针对那些具有先验的点，计算逆深度先验对应的能量 0.5 * (x - x_prior)^T * sigma * (x - x_prior)
    red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt, this, _1, _2, _3, _4), 0, allPoints.size(), 50);

    return E + red->stats[0];
}

/**
 * @brief 向能量函数中插入一个残差 r
 * @details
 *  1. 构建EFResidual，id为 ph 中维护的残差id --> 由point构建的
 *  2. 更新 host -> target 之间的链接图 [还在起作用res数目，被边缘化||删除的res数目]
 * @param r 输入输出的 host point target 之间的残差
 * @return EFResidual*  返回构建的efResidual
 */
EFResidual *EnergyFunctional::insertResidual(PointFrameResidual *r) {
    EFResidual *efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame); ///< 残差用来维护 point host target 三者关系
    efr->idxInAll = r->point->efPoint->residualsAll.size();                                       ///< 由这个点构建的残差id（滑窗内的数目 - 1 为 max）
    r->point->efPoint->residualsAll.push_back(efr);                                               ///< efPoint 维护创建的这个efr

    /// connectivityMap的key中，高位32位是host的id，低位32位是target的id
    /// connectivityMap的valu中，0-->代表两帧之间还在其作用的残差，1-->代表两帧之间已经失效的残差（被边缘化掉 || 被删掉）
    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

    nResiduals++;
    r->efResidual = efr;
    return efr;
}

/**
 * @brief 向energy functional 中添加一帧 fh
 * @details
 *  1. 根据fh，构建EFFrame，并插入到由energy functional 维护的滑动窗口中frames
 *  2. 由于上一次滑动窗口优化会边缘化掉 某个关键帧信息，为了保证HM 和 bM 的维度正确，需要将新添加的帧部分的位置进行拓展 (拓展HM和bM)
 *  3. 在新一轮的滑动窗口优化之前，需要根据当前的线性化点，求解 dlocal / dglobal_h 和 dlocal / dglobal_t --> 用于loacl -> global的转换 @see setAdjointsF
 *  4. 更新滑窗内帧的位置和滑窗内的残差信息（host->idx, target->idx）
 *  5. 更新 host -> target 之间的链接图
 *
 * @param fh        输入的待插入的关键帧 fh
 * @param Hcalib    维护的相机内参，貌似在代码中没有用到
 * @return EFFrame* 输出的EFFrame（根据插入的fh生成的与优化和关键帧有关的类型）
 */
EFFrame *EnergyFunctional::insertFrame(FrameHessian *fh, CalibHessian *Hcalib) {
    /// 建立优化用的能量函数帧eff，并使用 fh 进行维护
    EFFrame *eff = new EFFrame(fh);
    eff->idx = frames.size();
    frames.push_back(eff); ///< Energy functional 维护的滑动窗口中的帧

    nFrames++;
    fh->efFrame = eff;

    assert(HM.cols() == 8 * nFrames + CPARS - 8); ///< 这里是否代表着 使用insertFrame 之前 肯定会边缘化掉一个帧

    /// 对 HM 和 bM 拓展fh对应的部分 --> 反映在bM上应该是添加了8维，反映在HM上应该是添加了末尾的8行8列---> 多余部分都置为0
    /// @note conservativeResize被拓展的值不会被初始化！
    bM.conservativeResize(8 * nFrames + CPARS);
    HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);

    /// 由于使用 conservativeResize 拓展的值不会被初始化，因此需要将拓展的部分手动置零
    bM.tail<8>().setZero();
    HM.rightCols<8>().setZero();
    HM.bottomRows<8>().setZero();

    EFIndicesValid = false;  ///<
    EFAdjointsValid = false; ///< setAdjointsF，以当前估计值为线性化点，计算 d_local / d_global --> true
    EFDeltaValid = false;    ///< 当insertFrame运行成功后，只有DeltaValid为false

    /// 设置 d_local / d_global_t 和 d_loacl / d_global_h，用于后续local_delta --> global_delta
    setAdjointsF(Hcalib);

    /// 更新滑窗中关键帧idx 和 残差的维护的关键帧idx @see EnergyFunctional::makeIDX()
    makeIDX();

    /// 更新连接图
    for (EFFrame *fh2 : frames) {
        /// 以 eff 为 host
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);

        /// 以 滑动窗口内的 fh2 为 host
        if (fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0, 0);
    }

    return eff;
}

/**
 * @brief 向 energy founctional 中插入点
 *
 * @param ph   PointHessian，待插入的已经激活的点
 * @return EFPoint* 输出的由ph构建的 EFPoint--> 与优化和激活点PointHessian有关
 */
EFPoint *EnergyFunctional::insertPoint(PointHessian *ph) {
    /// 构建EFPoint--> 将efp 放入 ph的efframe 中进行维护
    EFPoint *efp = new EFPoint(ph, ph->host->efFrame);
    efp->idxInPoints = ph->host->efFrame->points.size();
    ph->host->efFrame->points.push_back(efp);

    nPoints++;
    ph->efPoint = efp; ///< ph的efPoint 只维护最新生成的 efPoint点

    EFIndicesValid = false; ///< 标志着滑窗中的id是否合法，如果有任何 point、residual或frame 插入都会导致id非法

    return efp;
}

/**
 * @brief 从滑动窗口中删除一个残差 EFResidual
 * @details
 *  1. 找到残差所属的 EFPoint ，使用尾替代法删除点中对应的该残差
 *  2. 更新 尾部 替代残差的 id
 *  3. 更新 connectivityMap 的内容，value的0代表起作用的残差，1代表被边缘化或删除的残差数
 *
 * @param r 输入的待删除的残差
 */
void EnergyFunctional::dropResidual(EFResidual *r) {
    EFPoint *p = r->point;
    assert(r == p->residualsAll[r->idxInAll]);

    /// 尾部替代删除法
    p->residualsAll[r->idxInAll] = p->residualsAll.back();
    p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
    p->residualsAll.pop_back();

    if (r->isActive())
        r->host->data->shell->statistics_goodResOnThis++;
    else
        r->host->data->shell->statistics_outlierResOnThis++;

    /// 更新 连接图关系，value的0项为起作用的残差数，1项为不起作用的残差数
    connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
    nResiduals--;
    r->data->efResidual = 0;
    delete r;
}

/**
 * @brief 能量函数中，边缘化掉某个EFFrame
 * @details
 *  1. 把边缘化的帧对应的H和b部分挪到最右边, 最下边
 *  2. 在帧进行边缘化之前，如果被边缘化的帧有先验，那么需要加上先验信息
 *  3. 在矩阵分解时，防止变态矩阵的影响，需要先进行矩阵缩放，缩放的方法与H * deltax = b类似
 *  4. 为了保证被边缘化部分的对称性，需要使用 0.5 * (hp + hp^T) 方式保证对称性，然后获得它的逆矩阵 hpi
 *  5. 使用舒尔补，进行边缘化 HM -= Hfo * Hff^-1 * Hof, bM -= Hfo * Hff^-1 * bf
 *  6. 边缘化后，使用缩放矩阵的逆变换回来
 *  7. 在设置HM和bM之前，还需要对HM部分做对称性处理
 *  8. 将被边缘化的帧从frames中删除（EnergyFunctional维护的滑窗中删除）
 *  9. 重新设置滑窗内所有参与项的idx
 *
 * @param fh 输入的待被边缘化的一帧
 */
void EnergyFunctional::marginalizeFrame(EFFrame *fh) {

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    assert((int)fh->points.size() == 0);

    int ndim = nFrames * 8 + CPARS - 8; ///< 边缘化之后 H 矩阵的维度
    int odim = nFrames * 8 + CPARS;     ///< 边缘化之前 H 矩阵的维度

    /// 1. 把边缘化的帧对应的H和b部分挪到最右边, 最下边
    if ((int)fh->idx != (int)frames.size() - 1) {

        int io = fh->idx * 8 + CPARS;            ///< 被边缘化的帧开始位置
        int ntail = 8 * (nFrames - fh->idx - 1); ///< 被边缘化的块后续还剩的部分
        assert((io + 8 + ntail) == nFrames * 8 + CPARS);

        /// bM对应的部分，应该是将 边缘化帧对应的8个变量和下面整体的bM进行交换即可
        Vec8 bTmp = bM.segment<8>(io);
        VecX tailTMP = bM.tail(ntail);
        bM.segment(io, ntail) = tailTMP;
        bM.tail<8>() = bTmp;

        /// HM矩阵列方向上的交换 待边缘化部分和尾部部分交换
        MatXX HtmpCol = HM.block(0, io, odim, 8);
        MatXX rightColsTmp = HM.rightCols(ntail);
        HM.block(0, io, odim, ntail) = rightColsTmp;
        HM.rightCols(8) = HtmpCol;

        /// HM行矩阵方向上的交换 待边缘化部分和尾部部分交换
        MatXX HtmpRow = HM.block(io, 0, 8, odim);
        MatXX botRowsTmp = HM.bottomRows(ntail);
        HM.block(io, 0, ntail, odim) = botRowsTmp;
        HM.bottomRows(8) = HtmpRow;
    }

    /// 2. 在帧进行边缘化之前，如果被边缘化的帧有先验，那么需要加上先验信息
    HM.bottomRightCorner<8, 8>().diagonal() += fh->prior;
    bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

    /// 3. 在矩阵分解时，防止变态矩阵的影响，需要先进行矩阵缩放，缩放的方法与H * deltax = b类似
    VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
    VecX SVecI = SVec.cwiseInverse();
    MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
    VecX bMScaled = SVecI.asDiagonal() * bM;

    /// 4. 为了保证被边缘化部分的对称性，需要使用 0.5 * (hp + hp^T) 方式保证对称性，然后获得它的逆矩阵 hpi
    Mat88 hp = HMScaled.bottomRightCorner<8, 8>();
    hp = 0.5f * (hp + hp.transpose());
    Mat88 hpi = hpi.inverse();
    hpi = 0.5f * (hpi + hpi.transpose());

    /// 5. 使用舒尔补，进行边缘化 HM -= Hfo * Hff^-1 * Hof, bM -= Hfo * Hff^-1 * bf
    MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
    HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
    bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

    /// 6. 边缘化后，使用缩放矩阵的逆变换回来
    HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
    bMScaled = SVec.asDiagonal() * bMScaled;

    /// 7. 在设置HM和bM之前，还需要对HM部分做对称性处理
    HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
    bM = bMScaled.head(ndim);

    /// 8. 将被边缘化的帧从frames中删除（EnergyFunctional维护的滑窗中删除）
    for (unsigned int i = fh->idx; i + 1 < frames.size(); i++) {
        frames[i] = frames[i + 1];
        frames[i]->idx = i;
    }
    frames.pop_back();
    nFrames--;
    fh->data->efFrame = 0;

    assert((int)frames.size() * 8 + CPARS == (int)HM.rows());
    assert((int)frames.size() * 8 + CPARS == (int)HM.cols());
    assert((int)frames.size() * 8 + CPARS == (int)bM.size());
    assert((int)frames.size() == (int)nFrames);

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    makeIDX(); ///< 重新设置滑窗内所有参与项的idx
    delete fh;
}

/**
 * @brief 边缘化掉被标记为边缘化的点
 *
 */
void EnergyFunctional::marginalizePointsF() {
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    /// 统计滑动窗口中，待marg的点，并更新连接图的残差内容，++value[1]
    allPointsToMarg.clear();
    for (EFFrame *f : frames) {
        for (int i = 0; i < (int)f->points.size(); i++) {
            EFPoint *p = f->points[i];
            if (p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
                p->priorF *= setting_idepthFixPriorMargFac;
                for (EFResidual *r : p->residualsAll)
                    if (r->isActive())
                        connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
                allPointsToMarg.push_back(p);
            }
        }
    }

    /// 计算该点相连的残差构成的 H, b, HSC, bSC
    accSSE_bot->setZero(nFrames);
    accSSE_top_A->setZero(nFrames);
    for (EFPoint *p : allPointsToMarg) {
        accSSE_top_A->addPoint<2>(p, this); // 这个点的残差, 计算 H b
        accSSE_bot->addPoint(p, false);     // 边缘化部分
        removePoint(p);
    }
    MatXX M, Msc;
    VecX Mb, Mbsc;
    accSSE_top_A->stitchDouble(M, Mb, this, false, false); // 不加先验, 在后面加了
    accSSE_bot->stitchDouble(Msc, Mbsc, this);

    resInM += accSSE_top_A->nres[0];

    MatXX H = M - Msc;
    VecX b = Mb - Mbsc;

    //[ ***step 3*** ] 处理零空间
    // 减去零空间部分
    if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
        // have a look if prior is there.
        bool haveFirstFrame = false;
        for (EFFrame *f : frames)
            if (f->frameID == 0)
                haveFirstFrame = true;

        if (!haveFirstFrame)
            orthogonalize(&b, &H);
    }

    //! 给边缘化的量加了个权重，不准确的线性化
    HM += setting_margWeightFac * H; //* 所以边缘化的部分直接加在HM bM了
    bM += setting_margWeightFac * b;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
        orthogonalize(&bM, &HM);

    EFIndicesValid = false;
    makeIDX(); // 梳理ID
}

/**
 * @brief 丢掉那些状态为 PS_DROP EFPoint，并且更新 energy functional 里面维护的滑动窗口的idx
 *
 * @see EnergyFunctional::removePoint
 */
void EnergyFunctional::dropPointsF() {
    for (EFFrame *f : frames) {
        for (int i = 0; i < (int)f->points.size(); i++) {
            EFPoint *p = f->points[i];
            if (p->stateFlag == EFPointStatus::PS_DROP) {
                removePoint(p);
                i--;
            }
        }
    }

    EFIndicesValid = false;
    makeIDX();
}

/**
 * @brief 从 EnergyFunctional 中丢弃一个点EFPoint
 * @details
 *  1. 丢弃掉该点的所有残差 @see EnergyFunctional::dropResidual
 *  2. 从 EFFrame 中删除该点，后移删除，并更新id的方法
 *  3. EFIndicesValid 标记为False，因为EFPoint的id发生了变化
 *
 * @param p 输入的待删除的efpoint
 */
void EnergyFunctional::removePoint(EFPoint *p) {
    /// 删除该点的所有残差
    for (EFResidual *r : p->residualsAll)
        dropResidual(r);

    /// 尾替代删除法，并更新idx
    EFFrame *h = p->host;
    h->points[p->idxInPoints] = h->points.back();
    h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
    h->points.pop_back();

    nPoints--;
    p->data->efPoint = 0;

    /// 标记目前的滑窗id非法，需要重新规划idx
    EFIndicesValid = false;

    delete p;
}

/**
 * @brief 将输入的系统减去零空间上的状态，可以作用在整个优化矩阵上，也可以作用在x的增量上
 * @details
 *  1. 将滑窗中求解的零空间的基组合成N矩阵
 *  2. 使用svd的方式，求解N矩阵的伪逆，得到N^-1 = (N^T * N)^-1 * N^T
 *  3. 将输入的H和b矩阵做零空间消除，得到不在零空间上的部分
 * @param b 输入的优化b矩阵，或者输入的求解完成的增量x
 * @param H 输入的优化H矩阵，或者输入的0（nullptr）
 */
void EnergyFunctional::orthogonalize(VecX *b, MatXX *H) {
    std::vector<VecX> ns;
    ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());   ///< 位姿部分零空间
    ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end()); ///< 尺度部分零空间

    //? 这里为什么去掉了 ab 对应的零空间
    //	if(setting_affineOptModeA <= 0)
    //		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
    //	if(setting_affineOptModeB <= 0)
    //		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());

    /// 7自由度不可观
    MatXX N(ns[0].rows(), ns.size());
    for (unsigned int i = 0; i < ns.size(); i++)
        N.col(i) = ns[i].normalized();

    /// 求零空间基的伪逆 N^-1 = (N^T * N)^-1 * N^T
    Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

    VecX SNN = svdNN.singularValues();
    double minSv = 1e10, maxSv = 0;
    for (int i = 0; i < SNN.size(); i++) {
        if (SNN[i] < minSv)
            minSv = SNN[i];
        if (SNN[i] > maxSv)
            maxSv = SNN[i];
    }

    /// 比最大奇异值小 (1e5)倍, 则认为是0，将奇异值中较小的部分置0
    for (int i = 0; i < SNN.size(); i++) {
        if (SNN[i] > setting_solverModeDelta * maxSv)
            SNN[i] = 1.0 / SNN[i];
        else
            SNN[i] = 0;
    }

    /// 求N（系统状态零空间）的伪逆 N^-1 = (N^T * N)^-1 * N^T --> 利用svd分解求出来
    MatXX Npi = svdNN.matrixV() * SNN.asDiagonal() * svdNN.matrixU().transpose();

    /// 使用 N * (N^T * N)^-1 * N^T 求解零空间投影矩阵，并且考虑到了投影矩阵的对称性 N_wrt = 0.5 * (N_wrt + N_wrt^T)
    MatXX NNpiT = N * Npi;
    MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose());

    /// 这个投影矩阵，应该是作用在求得的delta_x上的，相当于缩放矩阵
    if (b != 0)
        *b -= NNpiTS * *b;

    if (H != 0)
        *H -= NNpiTS * *H * NNpiTS;
}

/**
 * @brief 求解整个滑窗系统
 * @details
 *  1. 针对新加入的残差，构建残差的H矩阵和b矩阵，其中会更新对应点p的 Hdd_accAF, Hcd_accAF, bd_accAF @see EnergyFunctional::accumulateAF_MT
 *  2. 针对之前线性化的残差，利用状态增量更新rk，并更新对应点p的 Hdd_accLF, Hcd_accLF, bd_accLF @see EnergyFunctional::accumulateLF_MT
 *  3. 针对当前滑窗中存在的所有激活残差，进行Hsc 和 bsc求解 @see EnergyFunctional::accumulateSCF_MT
 *      3.1 @note 考虑到最后使用的是HA + HM - Hsc求得当前滑窗的H矩阵，因此这里是否能推测 HM中包含的信息，并没有边缘化掉那些线性化的点？
 *  4. 求解考虑边缘化后的滑窗系统的H和b矩阵
 *      4.1 使用 HM + HA - Hsc 作为滑窗系统最后的 H 矩阵
 *      4.2 由于使用了FEJ，因此HM不会改变，根据状态的更新，对bM进行更新
 *  5. 使用数值稳定的ldlt的方式求解 帧状态增量（位姿+光度） 相机内参增量
 *      5.1 根据最后的H矩阵对角线部分，构建变换矩阵，防止H矩阵为病态矩阵导致不稳定性
 *      5.2 最后，将变换矩阵左乘到求解的增量中，得到最后的状态增量
 *  6. 考虑VO系统的零空间部分，使用状态增量 - 零空间投影的方式，得到不含零空间的状态增量 @see EnergyFunctional::orthogonalize
 *  7. 根据求解的部分状态增量(不含零空间部分)，对点的逆深度进行更新 @see EnergyFunctional::resubstituteF_MT
 *
 * @param iteration 输入的当前优化迭代次数
 * @param lambda    输入的阻尼参数
 * @param HCalib    输入的相机内参
 */
void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian *HCalib) {
    /// 使用的阻尼高斯牛顿方法，根据不同的solver配置不同的lambda值
    if (setting_solverMode & SOLVER_USE_GN)
        lambda = 0;
    if (setting_solverMode & SOLVER_FIX_LAMBDA)
        lambda = 1e-5;

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    MatXX HL_top, HA_top, H_sc;
    VecX bL_top, bA_top, bM_top, b_sc;

    /// 计算滑动窗口内所有的残差项，对应的H矩阵和b矩阵，并更新涉及到的p状态 Hdd_accAF, Hcd_accAF, bd_accAF
    accumulateAF_MT(HA_top, bA_top, multiThreading);

    /// 计算之前线性化过的残差，对应的H矩阵和b矩阵，并更新涉及到的p状态 Hdd_accLF, Hcd_accLF, bd_accLF，值应该是0，正常来讲应该被边缘化掉了
    accumulateLF_MT(HL_top, bL_top, multiThreading);

    /// 计算 H_sc 和 b_sc 矩阵，这个Hsc和bsc是对当前滑窗内的活动残差的Schur，HA_top, bA_top部分的Schur矩阵
    accumulateSCF_MT(H_sc, b_sc, multiThreading);

    /// HM和bM部分，由于固定了线性化点，因此HM部分不能变化，根据delta的变化更新bM
    bM_top = (bM + HM * getStitchedDeltaF());

    MatXX HFinal_top;
    VecX bFinal_top;

    /// 设置的配置文件中，并没有使用投影系统的方式，而是投影求解后的delta_x
    if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
        // have a look if prior is there.
        bool haveFirstFrame = false;
        for (EFFrame *f : frames)
            if (f->frameID == 0)
                haveFirstFrame = true;

        // 计算Schur之后的
        // MatXX HT_act =  HL_top + HA_top - H_sc;
        MatXX HT_act = HA_top - H_sc;
        // VecX bT_act =   bL_top + bA_top - b_sc;
        VecX bT_act = bA_top - b_sc;

        //! 包含第一帧则不减去零空间
        //! 不包含第一帧, 因为要固定第一帧, 和第一帧统一, 减去零空间, 防止在零空间乱飘
        if (!haveFirstFrame)
            orthogonalize(&bT_act, &HT_act);

        HFinal_top = HT_act + HM;
        bFinal_top = bT_act + bM_top;

        lastHS = HFinal_top;
        lastbS = bFinal_top;
        // LM
        //* 这个阻尼也是加在 Schur complement 计算之后的
        for (int i = 0; i < 8 * nFrames + CPARS; i++)
            HFinal_top(i, i) *= (1 + lambda);

    } else {

        /// HFinal_top = HM + HA_top;
        HFinal_top = HM + HA_top;

        /// bFinal_top = bM_top + bA_top - b_sc;
        bFinal_top = bM_top + bA_top - b_sc;

        /// 没有阻尼的系统 H 和 b 矩阵
        lastHS = HFinal_top - H_sc;
        lastbS = bFinal_top;

        /// 对整个系统添加阻尼（包括逆深度点），然后再计算 schur 后的H 和 b矩阵 --> 注意整个系统添加阻尼 和 除了逆深度点部分添加阻尼是不同的
        for (int i = 0; i < 8 * nFrames + CPARS; i++)
            HFinal_top(i, i) *= (1 + lambda);

        HFinal_top -= H_sc * (1.0f / (1 + lambda));
    }

    /// 求解 H * delta_x = b
    VecX x;
    if (setting_solverMode & SOLVER_SVD) {
        //* 为数值稳定进行缩放
        VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
        VecX bFinalScaled = SVecI.asDiagonal() * bFinal_top;
        //! Hx=b --->  U∑V^T*x = b
        Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

        VecX S = svd.singularValues(); // 奇异值
        double minSv = 1e10, maxSv = 0;
        for (int i = 0; i < S.size(); i++) {
            if (S[i] < minSv)
                minSv = S[i];
            if (S[i] > maxSv)
                maxSv = S[i];
        }

        //! Hx=b --->  U∑V^T*x = b  --->  ∑V^T*x = U^T*b
        VecX Ub = svd.matrixU().transpose() * bFinalScaled;
        int setZero = 0;
        for (int i = 0; i < Ub.size(); i++) {
            if (S[i] < setting_solverModeDelta * maxSv) //* 奇异值小的设置为0
            {
                Ub[i] = 0;
                setZero++;
            }

            if ((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size() - 7)) //* 留出7个不可观的, 零空间
            {
                Ub[i] = 0;
                setZero++;
            }
            //! V^T*x = ∑^-1*U^T*b
            else
                Ub[i] /= S[i];
        }
        //! x = V*∑^-1*U^T*b   把scaled的乘回来
        x = SVecI.asDiagonal() * svd.matrixV() * Ub;

    } else {
        /// 针对H矩阵对角部分的元素，进行矩阵缩放，保证ldlt的数值稳定性，sqrt(对角 + 10)^-1 --> 缩放矩阵
        VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();

        /// 使用缩放矩阵，将结果再求出来
        x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);
    }

    /// 如果设置的是直接对解进行处理, 直接去掉解x中的零空间 --> x向零空间投影，然后去掉零空间的部分增量
    if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
        VecX xOld = x;
        orthogonalize(&x, 0); ///< 解x向零空间投影
    }

    lastX = x;

    /// 根据求解的 帧参数增量 和 内参增量，进行逆深度增量的求解，逆深度的更新没有使用阻尼
    currentLambda = lambda;
    resubstituteF_MT(x, HCalib, multiThreading);
    currentLambda = 0;
}

/**
 * @brief 更新 energy functional 里面维护的滑动窗口的idx，并根据滑窗中的关键帧 更新滑窗中的激活点信息和残差信息
 * @details
 *  1. 更新滑窗内的关键的id信息（因为有新的帧加入，并且旧的帧还被边缘化掉了）
 *  2. 根据关键帧的信息构建滑动窗口的残差信息 --> 主要是更新 残差维护的 host_idx 和 target_idx
 */
void EnergyFunctional::makeIDX() {
    /// 更新滑动窗口关键帧在滑窗中对应的idx(在滑窗中的位置)
    for (unsigned int idx = 0; idx < frames.size(); idx++)
        frames[idx]->idx = idx;

    /// 将上次窗口优化的点清除掉，需要重新构建当前窗口中的约束
    allPoints.clear();

    for (EFFrame *f : frames)          ///< 遍历滑动窗口中的所有关键帧 f
        for (EFPoint *p : f->points) { ///< 遍历f中所有已经激活的点 p
            allPoints.push_back(p);    ///< 把点加入到滑动窗口中

            /// 更新残差中维护的 host 和 target 的滑窗位置 idx（因为前面更新了滑动窗口中帧的位置）
            for (EFResidual *r : p->residualsAll) {
                r->hostIDX = r->host->idx;
                r->targetIDX = r->target->idx;
            }
        }

    EFIndicesValid = true;
}

/**
 * @brief 获取所有状态参数的delta --> state - state_zero
 *
 * @return VecX 输出相对于线性化点处的状态变化量
 */
VecX EnergyFunctional::getStitchedDeltaF() const {
    VecX d = VecX(CPARS + nFrames * 8);
    d.head<CPARS>() = cDeltaF.cast<double>();
    for (int h = 0; h < nFrames; h++)
        d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
    return d;
}

} // namespace dso
