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

#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

/**
 * @brief 根据滑窗内的残差状态，构建残差的 相对量的 Hessian 矩阵 --> 对于H[Tth,dpi]和H[dpi,Tth]之间都没有构建，其余的构建都构建了，包括整个b向量
 * @details
 *	1. mode = 0 --> 针对新加入的激活残差，构建滑窗内的H 和 b矩阵
 *	2. mode = 1 --> 针对之前已经线性化后的残差，构建滑窗内的H 和 b矩阵
 *		2.1 由于 dpj / dstate 中存在FEJ的状态，但是由于状态的变化，这里的rk需要产生一些变化
 *		2.2 FEJ 要求雅可比矩阵必须在线性化的那个状态下，因此 sum(J^T * J) delta_x = J^T * rk --> 这里的rk会发生变化
 *		2.3 state变化 --> dpj变化 (dpj / dstate) * delta_state
 *		2.4 dpj 变化 + ab变化 --> rk 变化 (drk / dpj) * delta_pj + (drk / dab) * delta_ab
 *
 * @note 没有构建 H[(Tth,dpi)和(ath,dpi)和(bth,dpi)]
 * @note 目前仅 mode = 0 和 mode = 1部分完成了解读
 * @note 针对target不是最新关键帧的那些滑窗中的残差，也会走 mode = 0的分支！
 * 
 * @tparam mode 0 = active, 1 = linearized, 2=marginalize
 * @param p		输入的带有残差信息的能量点
 * @param ef	输入的维护滑窗的能量函数
 * @param tid	输入的线程id
 */
template <int mode> void AccumulatedTopHessianSSE::addPoint(EFPoint *p, EnergyFunctional const *const ef, int tid) {
    assert(mode == 0 || mode == 1 || mode == 2);

    VecCf dc = ef->cDeltaF; ///< 内参距离内参线性化点的距离
    float dd = p->deltaF;   ///< 逆深度 - 逆深度线性化点

    float bd_acc = 0;
    float Hdd_acc = 0;
    VecCf Hcd_acc = VecCf::Zero();

    for (EFResidual *r : p->residualsAll) {
        /// 使用最新关键帧优化出来的，最新加入的激活点，但是没有被线性化固定的残差
        if (mode == 0) {
            if (r->isLinearized || !r->isActive())
                continue;
        }

        /// 在新加入点之前，滑动窗口中就已经存在的，且经过优化被线性化固定的残差
        if (mode == 1) {
            if (!r->isLinearized || !r->isActive())
                continue;
        }

        if (mode == 2) {
            if (!r->isActive())
                continue;
            assert(r->isLinearized);
        }

        RawResidualJacobian *rJ = r->J;

        int htIDX = r->hostIDX + r->targetIDX * nframes[tid];
        Mat18f dp = ef->adHTdeltaF[htIDX]; ///< delta_Tth 和 delta_ath 和 delta_bth

        VecNRf resApprox;
        if (mode == 0)
            resApprox = rJ->resF;

        if (mode == 2)
            resApprox = r->res_toZeroF;

        if (mode == 1) {
            /// 针对那些窗口中之前已经被线性化和激活的点，需要根据当前状态的变化，来更新rk，残差
            __m128 Jp_delta_x = _mm_set1_ps(rJ->Jpdxi[0].dot(dp.head<6>()) + rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd); ///< delta_pj[x]
            __m128 Jp_delta_y = _mm_set1_ps(rJ->Jpdxi[1].dot(dp.head<6>()) + rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd); ///< dleta_pj[y]
            __m128 delta_a = _mm_set1_ps((float)(dp[6]));                                                             ///< delta_aji
            __m128 delta_b = _mm_set1_ps((float)(dp[7]));                                                             ///< delta_bji

            for (int i = 0; i < patternNum; i += 4) {
                /// 拿到在线性化点处的残差值 --> 4个rk
                __m128 rtz = _mm_load_ps(((float *)&r->res_toZeroF) + i);
                rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx)) + i), Jp_delta_x));     ///< delta_pj[x]贡献的残差
                rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx + 1)) + i), Jp_delta_y)); ///< delta_pj[y]贡献的残差
                rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF)) + i), delta_a));        ///< delta_aji贡献的残差
                rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF + 1)) + i), delta_b));    ///< delta_bji贡献的残差
                _mm_store_ps(((float *)&resApprox) + i, rtz);                                              ///< 将这部分的残差值，更新到resApprox中
            }
        }

        Vec2f JI_r(0, 0);  ///< sum((drk / dpjf)^T * rk)
        Vec2f Jab_r(0, 0); ///< sum((drk / dab)^T * rk)
        float rr = 0;
        for (int i = 0; i < patternNum; i++) {
            JI_r[0] += resApprox[i] * rJ->JIdx[0][i];  ///< sum(rk * drk / dpjf[x])
            JI_r[1] += resApprox[i] * rJ->JIdx[1][i];  ///< sum(rk * drk / dpjf[y])
            Jab_r[0] += resApprox[i] * rJ->JabF[0][i]; ///< sum(rk * drk / dath)
            Jab_r[1] += resApprox[i] * rJ->JabF[1][i]; ///< sum(rk * drk / dbth)
            rr += resApprox[i] * resApprox[i];         ///< sum(rk * rk)
        }

        /// 计算 target --> host 之间的hessian （Tth，C）--> 10 * 10 的矩阵
        acc[tid][htIDX].update(rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(), rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(), rJ->JIdx2(0, 0), rJ->JIdx2(0, 1),
                               rJ->JIdx2(1, 1));

        /// 计算 target --> host 之间的H[a,b] 和 b[a,b]
        acc[tid][htIDX].updateBotRight(rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0], rJ->Jab2(1, 1), Jab_r[1], rr);

        /// 计算 target --> host 之间的H[(Tth,C)和(ath,bth)] 和 b[(Tth,C)和(ath,bth)]
        acc[tid][htIDX].updateTopRight(rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(), rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(), rJ->JabJIdx(0, 0), rJ->JabJIdx(0, 1),
                                       rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1), JI_r[0], JI_r[1]);

        /// 计算了 target --> host 之间的H[(dpi,dpi)和(dpi,C)],b[dpi]
        Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;                            ///< sum((drk / dpj)^T * (drk / ddpi))
        bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];          ///< sum(rk * drk / ddpi)
        Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);                                ///< sum((drk / ddpi) * (drk / ddpi))
        Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1]; ///< sum((drk / dC)^T * (drk / ddpi))

        nres[tid]++;
    }

    if (mode == 0) {
        p->Hdd_accAF = Hdd_acc;
        p->bd_accAF = bd_acc;
        p->Hcd_accAF = Hcd_acc;
    }
    if (mode == 1 || mode == 2) {
        p->Hdd_accLF = Hdd_acc;
        p->bd_accLF = bd_acc;
        p->Hcd_accLF = Hcd_acc;
    }
    if (mode == 2) // 边缘化掉, 设为0
    {
        p->Hcd_accAF.setZero();
        p->Hdd_accAF = 0;
        p->bd_accAF = 0;
    }
}
// 实例化，编译器会生成类中的所有成员，而且整个程序中仅有一份
template void AccumulatedTopHessianSSE::addPoint<0>(EFPoint *p, EnergyFunctional const *const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint *p, EnergyFunctional const *const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<2>(EFPoint *p, EnergyFunctional const *const ef, int tid);

//@ 对某一个线程进行的 H 和 b 计算, 或者是没有使用多线程
void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior, bool useDelta, int tid) {
    H = MatXX::Zero(nframes[tid] * 8 + CPARS, nframes[tid] * 8 + CPARS);
    b = VecX::Zero(nframes[tid] * 8 + CPARS);

    for (int h = 0; h < nframes[tid]; h++)
        for (int t = 0; t < nframes[tid]; t++) {
            int hIdx = CPARS + h * 8;
            int tIdx = CPARS + t * 8;
            int aidx = h + nframes[tid] * t;

            acc[tid][aidx].finish();
            if (acc[tid][aidx].num == 0)
                continue;

            MatPCPC accH = acc[tid][aidx].H.cast<double>();

            H.block<8, 8>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();

            H.block<8, 8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

            H.block<8, 8>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

            H.block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);

            H.block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

            H.topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);

            b.segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

            b.segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

            b.head<CPARS>().noalias() += accH.block<CPARS, 1>(0, 8 + CPARS);
        }

    // ----- new: copy transposed parts.
    for (int h = 0; h < nframes[tid]; h++) {
        int hIdx = CPARS + h * 8;
        H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

        for (int t = h + 1; t < nframes[tid]; t++) {
            int tIdx = CPARS + t * 8;
            H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
            H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
        }
    }

    if (usePrior) {
        assert(useDelta);
        H.diagonal().head<CPARS>() += EF->cPrior;
        b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
        for (int h = 0; h < nframes[tid]; h++) {
            H.diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
            b.segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
        }
    }
}

/**
 * @brief 构建滑动窗口中（不包含margined帧）H矩阵和b矩阵，会考虑先验（但是这个H矩阵并不完整，需要进一步处理）
 * @note 这里构建的 H 矩阵和 b矩阵 不包含 逆深度信息，并且还没有考虑marg掉产生先验的部分 HM和bM，但是考虑了先验部分（逆深度点的先验除外）
 * @note 以帧0为host，帧1为target，可构建 H[f0,f0]、H[f1,f1]、H[f0,f1]、H[f1,f0]、H[f0,C]、H[f1,C]、H[C,f0]、H[C,f1],使用0为host的逆深度点
 * @note 以帧1为host，帧0为target，可构建 H[f1,f1]、H[f0,f0]、H[f1,f0]、H[f0,f1]、H[f1,C]、H[f0,C]、H[C,f1]、H[C,f0],使用1为host的逆深度点
 * @note 但是源码中，仅仅对H[f0,f0]、H[f1,f1]、H[f0,f1]、H[f0,C]、H[f1,C]进行了更新
 * @note 但是源码中，仅仅对H[f1,f1]、H[f0,f0]、H[f1,f0]、H[f1,C]、H[f0,C]进行了更新
 * @note 缺少的内容为 部分H[f0,f1]、部分H[f1,f0]、所有的H[C,f0]、所有的H[C,f1]，并且可以发现，部分H[f0,f1]、部分H[f1,f0]是互补的，转置相加即可恢复H矩阵
 * @note 所有的H[C,f0]、所有的H[C,f1]可以通过所有的H[f0,C]、所有的H[f1,C]转置获得
 * @details
 * 	1. 遍历 AccumulatedTopHessianSSE::addPoint 构建的 不同 host --> target 之间的相对H矩阵和b矩阵
 * 	2. 构建一个滑窗大小相关的 滑窗系统的H矩阵 和 b矩阵 --> (CPARS + nframes * 8)
 * 	3. 根据伴随关系和ab的雅可比关系，构建 将相对量部分的H矩阵变换为 部分绝对量的H矩阵 --> 并加入其中 H 和 b
 * 	4. 最后，考虑了相机内参C 和 滑窗中帧的先验，并使用先验更新 滑窗系统的H矩阵 和 b矩阵
 * @param H			输出的待构建的滑窗系统的H矩阵
 * @param b			输出的待构建的滑窗系统的b矩阵
 * @param EF		输入的 EnergyFunctional
 * @param usePrior	输入的 是否使用先验的标志
 * @param min		多线程相关
 * @param max		多线程相关
 * @param stats		多线程信息统计相关
 * @param tid		多线程线程id相关
 */
void AccumulatedTopHessianSSE::stitchDoubleInternal(MatXX *H, VecX *b, EnergyFunctional const *const EF, bool usePrior, int min, int max, Vec10 *stats,
                                                    int tid) {
    /// 不使用多线程情况下的处理
    int toAggregate = NUM_THREADS;
    if (tid == -1) {
        toAggregate = 1;
        tid = 0;
    }
    if (min == max)
        return;

    /// 遍历某个 host --> target 的投影
    for (int k = min; k < max; k++) {
        int h = k % nframes[0]; ///< 获取host idx
        int t = k / nframes[0]; ///< 获取target idx

        int hIdx = CPARS + h * 8; ///< host 开始的idx
        int tIdx = CPARS + t * 8; ///< target 开始的idx

        int aidx = h + nframes[0] * t;
        assert(aidx == k);

        MatPCPC accH = MatPCPC::Zero();

        /// 获取 host --> target 维护的 H 矩阵 和 b 矩阵  相机内参(4) + 相对位姿(6) + 光度仿射参数(2) + b矩阵(1) --> 13 * 13
        for (int tid2 = 0; tid2 < toAggregate; tid2++) {
            acc[tid2][aidx].finish();
            if (acc[tid2][aidx].num == 0)
                continue;
            accH += acc[tid2][aidx].H.cast<double>();
        }

        /// 相对的量通过adj变成绝对的量, 并累加到 H, b 中，使用noalias()函数代表内存中只有H，没有其他名字指向这块内存，代码更高效
        /// H[hIdx, hIdx] --> host 对应的 H 矩阵
        /// H[tIdx, tIdx] --> target 对应的 H 矩阵
        /// H[hIdx, tIdx] --> host 和 target H，但是没有更新H[tIdx, hIdx]的部分 --> 这样的话会导致H矩阵的帧参数的非对角线部分，需要转置加和才能代表真正的H矩阵
        /// H[hIdx, C] --> host 和 C H矩阵的下三角阵 --> 这里没有没有更新 H[C, hIdx] 部分
        /// H[tIdx, C] --> target 和 C H矩阵的下三角阵 --> 这里没有更新 H[C, tIdx] 部分
        /// H[C, C] --> C 和 C 对应的H矩阵
        H[tid].block<8, 8>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();
        H[tid].block<8, 8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();
        H[tid].block<8, 8>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();
        H[tid].block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);
        H[tid].block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);
        H[tid].topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);

        b[tid].segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);   ///< sum((drk / dhost)^T * rk)
        b[tid].segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, CPARS + 8); ///< sun((drk / dtarget)^T * rk)
        b[tid].head<CPARS>().noalias() += accH.block<CPARS, 1>(0, CPARS + 8);                         ///< sum((drk / dC)^T * rk)
    }

    /// 将先验部分加到当前滑窗构建的 H矩阵 和 b矩阵中 --> H矩阵为先验矩阵（先验二阶导），b矩阵为 H * delta_X（先验一阶导）
    if (min == 0 && usePrior) {
        H[tid].diagonal().head<CPARS>() += EF->cPrior;                               ///< 相机内参部分对应的先验矩阵
        b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>()); ///< 先验部分对应的雅可比矩阵^T --> H_prior * delta_X

        /// 帧先验矩阵部分
        for (int h = 0; h < nframes[tid]; h++) {
            H[tid].diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;                               ///< 帧状态先验矩阵
            b[tid].segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior); ///< 点状态先验矩阵
        }
    }
}

} // namespace dso
