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

#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso {

/**
 * @brief 针对某个逆深度点p，构建逆深度点的 Hsc 和 bsc --> 值得注意的是，这里的Hsc 和 bsc 都是相对量，后续需要转换
 *
 * @note 成员 accD 		中维护的是 H[(Tth,ath,bth)_i, (Tth,ath,bth)_j] 对应的Hsc --> Hff
 * @note 成员 accHcc 	中维护的是 H[C,C] 对应的Hsc	--> Hcc
 * @note 成员 accE 		中维护的是 H[(Tth,ath,bth)_i, C] 对应的Hsc --> Hfc
 * @note 成员 accbc 	中维护的是 bc 对应的 bcsc [相机内参部分]
 * @note 成员 accEb 	中维护的是 bf 对应的 bfsc [相对帧参数部分]
 * @param p					输入的逆深度点
 * @param shiftPriorToZero	输入的是否考虑点的先验标志
 * @param tid				输入的多线程相关id
 */
void AccumulatedSCHessianSSE::addPoint(EFPoint *p, bool shiftPriorToZero, int tid) {
    int ngoodres = 0;
    for (EFResidual *r : p->residualsAll)
        if (r->isActive())
            ngoodres++;

    if (ngoodres == 0) {
        p->HdiF = 0;
        p->bdSumF = 0;
        p->data->idepth_hessian = 0;
        p->data->maxRelBaseline = 0;
        return;
    }

    /// 逆深度部分对应的 能量Hessian = 正常激活残差部分 + 线性化残差部分 + 先验hessian
    float H = p->Hdd_accAF + p->Hdd_accLF + p->priorF;

    if (H < 1e-10)
        H = 1e-10;

    p->data->idepth_hessian = H;

    p->HdiF = 1.0 / H;

    /// 逆深度部分对应的 能量雅可比 --> 线性化残差部分 + 正常激活残差部分 + 点的先验部分
    p->bdSumF = p->bd_accAF + p->bd_accLF;
    if (shiftPriorToZero)
        p->bdSumF += p->priorF * p->deltaF;

    /// 拿到逆深度和内参的交叉项 --> 线性化残差部分 + 正常激活残差部分
    VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;

    /// Hcd * Hdd_inv * Hdc --> 将Hcd给边缘化掉
    accHcc[tid].update(Hcd, Hcd, p->HdiF);

    /// Hcd * Hdd_inv * bd	--> 将bd给边缘化掉 --> bcsc
    accbc[tid].update(Hcd, p->bdSumF * p->HdiF);

    assert(std::isfinite((float)(p->HdiF)));

    int nFrames2 = nframes[tid] * nframes[tid];

    ///! 这里是否可以，对某个host来讲，仅算一半，其余的一半先不算
    for (EFResidual *r1 : p->residualsAll) {
        if (!r1->isActive())
            continue;
        int r1ht = r1->hostIDX + r1->targetIDX * nframes[tid];

        for (EFResidual *r2 : p->residualsAll) {
            if (!r2->isActive())
                continue;
            /// Hfd_1 * Hdd_inv * Hfd_2^T,  f = [xi, a b]位姿 光度 --> 边缘化掉Hfd_k
            /// 这部分获得的Hsc 对应的是 Hff 部分 (针对某个点的所有残差，构建的Hsc)
            accD[tid][r1ht + r2->targetIDX * nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF);
        }

        /// Hfd * Hdd_inv * Hcd^T
        accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);

        /// Hfd * Hdd_inv * bd --> bfsc
        accEB[tid][r1ht].update(r1->JpJdF, p->HdiF * p->bdSumF);
    }
}

/**
 * @brief 构建Hsc和bsc矩阵
 *
 * @see AccumulatedSCHessianSSE::stitchDouble
 *
 * @param H     输出的Hsc矩阵
 * @param b     输出的bsc矩阵
 * @param EF    输入的EnergyFunctional
 * @param min   多线程相关
 * @param max   多线程相关
 * @param stats 多线程输出相关
 * @param tid   多线程相关
 */
void AccumulatedSCHessianSSE::stitchDoubleInternal(MatXX *H, VecX *b, EnergyFunctional const *const EF, int min, int max, Vec10 *stats, int tid) {
    int toAggregate = NUM_THREADS;
    if (tid == -1) {
        toAggregate = 1;
        tid = 0;
    }
    if (min == max)
        return;

    int nf = nframes[0];
    int nframes2 = nf * nf;

    for (int k = min; k < max; k++) {
        int i = k % nf;
        int j = k / nf;

        int iIdx = CPARS + i * 8;
        int jIdx = CPARS + j * 8;
        int ijIdx = i + nf * j;

        Mat8C Hpc = Mat8C::Zero();
        Vec8 bp = Vec8::Zero();

        //* 所有线程求和
        for (int tid2 = 0; tid2 < toAggregate; tid2++) {
            accE[tid2][ijIdx].finish();
            accEB[tid2][ijIdx].finish();
            Hpc += accE[tid2][ijIdx].A1m.cast<double>();
            bp += accEB[tid2][ijIdx].A1m.cast<double>();
        }
        //! Hfc部分Schur
        H[tid].block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * Hpc;
        H[tid].block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * Hpc;
        //! 位姿,光度部分的残差Schur
        b[tid].segment<8>(iIdx) += EF->adHost[ijIdx] * bp;
        b[tid].segment<8>(jIdx) += EF->adTarget[ijIdx] * bp;

        for (int k = 0; k < nf; k++) {
            int kIdx = CPARS + k * 8;
            int ijkIdx = ijIdx + k * nframes2;
            int ikIdx = i + nf * k;

            Mat88 accDM = Mat88::Zero();

            for (int tid2 = 0; tid2 < toAggregate; tid2++) {
                accD[tid2][ijkIdx].finish();
                if (accD[tid2][ijkIdx].num == 0)
                    continue;
                accDM += accD[tid2][ijkIdx].A1m.cast<double>();
            }

            //! Hff部分Schur
            H[tid].block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
            H[tid].block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
            H[tid].block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
            H[tid].block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
        }
    }

    if (min == 0) {
        for (int tid2 = 0; tid2 < toAggregate; tid2++) {
            accHcc[tid2].finish();
            accbc[tid2].finish();
            //! Hcc 部分Schur
            H[tid].topLeftCorner<CPARS, CPARS>() += accHcc[tid2].A1m.cast<double>();
            //! 内参部分的残差Schur
            b[tid].head<CPARS>() += accbc[tid2].A1m.cast<double>();
        }
    }
}

/**
 * @brief 将求解的相对舒尔补部分，使用adHost和adTarget的方式，实现相对到绝对的变换，获取滑窗绝对的舒尔补矩阵
 * @details
 *  1. 对于某个帧相对状态 和 相机内参之间的Schur 矩阵 Hji[ji, C]_sc 和 bji[ji]_sc
 *  2. 根据计算的dlocal_state / dglobal_state 进行相对量到绝对量之间的转换 H[i, C],H[j,C]和b[i],b[j]部分的更新
 *  3. 针对某个host，进行舒尔补部分的存储--> 将这部分分别拿出来，使用相对和绝对之间的转换，扔到H和b中
 * @param H     输出的绝对舒尔补H矩阵
 * @param b     输出的绝对舒尔补b向量
 * @param EF    输入的 Energy Funcional
 * @param tid   多线程id相关
 */
void AccumulatedSCHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, int tid) {

    int nf = nframes[0];
    int nframes2 = nf * nf;

    H = MatXX::Zero(nf * 8 + CPARS, nf * 8 + CPARS);
    b = VecX::Zero(nf * 8 + CPARS);

    for (int i = 0; i < nf; i++)
        for (int j = 0; j < nf; j++) {
            int iIdx = CPARS + i * 8;
            int jIdx = CPARS + j * 8;
            int ijIdx = i + nf * j;

            accE[tid][ijIdx].finish();
            accEB[tid][ijIdx].finish();

            /// Hji[ji, C]_sc = sum((drk / dji)^T * (drk / dp)) * sum((drk / dp)^T * (drk / dp))) * sum((drk / dp)^T * (drk / dC))
            Mat8C accEM = accE[tid][ijIdx].A1m.cast<double>();

            /// bji[ji]_sc = sum(drk / dji)^T * (drk / dp)) * sum((drk / dp)^T * (drk / dp))) * sum((drk / dp)^T * rk)
            Vec8 accEBV = accEB[tid][ijIdx].A1m.cast<double>();

            H.block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * accEM;   ///< Hic_sc
            H.block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * accEM; ///< Hjc_sc

            b.segment<8>(iIdx) += EF->adHost[ijIdx] * accEBV;   ///< bi_sc
            b.segment<8>(jIdx) += EF->adTarget[ijIdx] * accEBV; ///< bj_sc

            for (int k = 0; k < nf; k++) {
                int kIdx = CPARS + k * 8;
                int ijkIdx = ijIdx + k * nframes2;
                int ikIdx = i + nf * k;

                accD[tid][ijkIdx].finish();
                if (accD[tid][ijkIdx].num == 0)
                    continue;
                Mat88 accDM = accD[tid][ijkIdx].A1m.cast<double>();

                /// Hii_sc +=
                /// (ddelta_ji / ddelta_i)^T * sum((drk / ddelta_ji)^T * (drk / dp)) *
                /// sum((drk / ddp)^T * (drk / ddp)) *
                /// sum((drk / dp)^T * (drk / ddelta_ki))* (ddelta_ki / ddelta_i)
                H.block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();     ///< Hii_sc
                H.block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose(); ///< Hjk_sc
                H.block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();   ///< Hji_sc
                H.block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();   ///< Hik_sc
            }
        }

    accHcc[tid].finish();
    accbc[tid].finish();
    H.topLeftCorner<CPARS, CPARS>() = accHcc[tid].A1m.cast<double>();
    b.head<CPARS>() = accbc[tid].A1m.cast<double>();

    /// 由于相机内参部分还没有经过对称处理，帧参数和侦帧参数部分，都实现了舒尔补的处理
    for (int h = 0; h < nf; h++) {
        int hIdx = CPARS + h * 8;
        H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();
    }
}

} // namespace dso
