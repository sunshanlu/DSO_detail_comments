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

#include "FullSystem/FullSystem.h"

#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace dso {

/**
 * @brief 对所有的残差进行线性化，并将pattern中投影不成功的点和pattern中投影成功但是双线性插值失败的点都设置为OOB，并进行移除标记
 * @details
 *  1. 首先计算 dpj / d(delta_Tth,FEJ)、dpj / d(delta_dpi,FEJ)、dpj / d(K,非FEJ) 和 sum(drk / dpj) @see PointFrameResidual::linearize
 *  2. 根据当前残差的中间状态，计算出 drk / dstate 带有部分FEJ状态 和 部分非FEJ状态 @see PointFrameResidual::applyRes
 *  3. 计算当前残差的 估计状态投影像素点和无穷远深度投影像素点之间的像素距离，被称为relBS，从而更新 ph 维护的 maxRelBaseline
 *  4. 针对 线性化过程中被标记为 OOB 的残差，进行移除标记 @see PointFrameResidual::linearize
 * @param fixLinearization  输入的是否固定线性化点 --> 如果固定线性化点，会使用state_zero进行 dpj / dstate 的雅可比计算FEJ
 * @param toRemove          输出的标记的待移除的残差项 --> 投影不成功的 || 双线性插值失败的点
 * @param min               线程负责的最小索引id
 * @param max               线程负责的最大索引id
 * @param stats             vec10的第0项接受所有残差的能量值加和
 * @param tid               线程id
 */
void FullSystem::linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual *> *toRemove, int min, int max, Vec10 *stats, int tid) {
    for (int k = min; k < max; k++) {
        PointFrameResidual *r = activeResiduals[k];

        /// 对残差进行线性化，里面有FEJ的状态，输出的是残差非FEJ对应的能量值
        (*stats)[0] += r->linearize(&Hcalib);

        if (fixLinearization) {
            /// 将线性化的状态拷贝到EFResidual中，构建drk / dstate
            r->applyRes(true);

            if (r->efResidual->isActive()) {
                /// 这部分的isNew的检测，好像没有什么作用，因为每次构建滑窗优化时，都会进行重新的残差构建，所有残差都是new
                if (r->isNew) {
                    /// 真实投影点pj和host depth 无穷大的虚拟投影点pj之间的距离 ---> 计算出一个relBS --> 以此来更新点p维护的最大maxline
                    PointHessian *p = r->point;
                    Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u, p->v, 1); // projected point assuming infinite depth.
                    Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll * p->idepth_scaled; // projected point with real depth.
                    float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2])).norm(); // 0.01 = one pixel.

                    if (relBS > p->maxRelBaseline)
                        p->maxRelBaseline = relBS;

                    p->numGoodResiduals++;
                }
            } else {
                /// 标记需要被移除的残差
                toRemove[tid].push_back(activeResiduals[k]);
            }
        }
    }
}

/**
 * @brief 根据残差的中间量，计算 残差对应的 Hfd 部分
 *
 * @see PointFrameResidual::applyRes
 * @param copyJacobians
 * @param min
 * @param max
 * @param stats
 * @param tid
 */
void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid) {
    for (int k = min; k < max; k++)
        activeResiduals[k]->applyRes(true);
}

/**
 * @brief 计算新帧的能量阈值，用于判断残差链接的残差是否满足要求 (host or target)
 */
void FullSystem::setNewFrameEnergyTH() {

    // collect all residuals and make decision on TH.
    allResVec.clear();
    allResVec.reserve(activeResiduals.size() * 2);
    FrameHessian *newFrame = frameHessians.back();

    /// 将新帧上的残差能量值放到allResVec中
    for (PointFrameResidual *r : activeResiduals)
        if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame) {
            allResVec.push_back(r->state_NewEnergyWithOutlier);
        }

    if (allResVec.size() == 0) {
        newFrame->frameEnergyTH = 12 * 12 * patternNum;
        return;
    }

    /// 设置setting_frameEnergyTHN分位数所在的索引位置
    int nthIdx = setting_frameEnergyTHN * allResVec.size();

    assert(nthIdx < (int)allResVec.size());
    assert(setting_frameEnergyTHN < 1);

    /// 拿到 70% 所在的能量值，newfh上残差的70%都小于它nthElement
    std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end());
    float nthElement = sqrtf(allResVec[nthIdx]);

    /// 阈值设置，通过 setting_frameEnergyTHFacMedian setting_frameEnergyTHConstWeight setting_overallEnergyTHWeight三种阈值来
    newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
    newFrame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
    newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
    newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
}

/**
 * @brief 线性化 新加入的残差
 * @details
 *  1. 使用单线程或者多线程的方式进行激活残差的线性化，主要是计算 drk / dstate --> 残差的雅可比矩阵 @see FullSystem::linearizeAll_Reductor
 *      1.1 由于整个滑动窗口在优化过程中会将某些帧边缘化掉产生先验信息，因此为了保证系统的能观性不变，会使用状态的FEJ进行雅可比计算
 *      1.2 在DSO的源码里面，针对 drk / dK 的雅可比计算，并没有采用FEJ的情况，这可能是由于K具有较大的先验，在优化过程中的改变非常小
 *      1.3 在DSO的源码里面，针对 drk / ddelta_Tji 和 drk / ddelta_ab 的雅可比计算，采用的是部分FEJ的方式计算
 *          1.3.1 考虑到，FEJ可以维持整个系统的能观性不变，但是使用FEJ总是会造成线性误差
 *          1.3.2 由于图片函数是一个非凸性比较强的函数，因此在 (drk / dpj) * (dpj / dstate) 的雅可比求解过程中，(drk / dpj)是非FEJ的
 *      1.4 除此之外，DSO在源码中，将 pi 投影的pj 计算的 dpj / dstate 来近似的替代 整个pattern的 dpj_pattern / dstate
 *  2. 针对某个残差r，当pi投影到pj时，如果投影到图像边缘处，或者使用双线性插值计算图像灰度失败时，认定 --> 当前残差状态为 OOB --> 待丢弃状态
 *  3. 对于某个残差r，构建成功能量函数后，如果能量值大于host 或者 target的阈值时，认定为 OUT状态 --> 外点状态
 *  4. 将target 为newfh的激活残差拿出来，使用70% 和 不同的超参数阈值设置的方法计算新帧的能量阈值
 *  5. 将在残差线性化过程中被认定为 OOB 状态的残差进行移除，重点更新ph中与残差有关的变量状态
 * @param fixLinearization  是否固定线性化状态
 * @return Vec3 参数第一项为 线性化所有残差的能量和，第二、三项没用
 */
Vec3 FullSystem::linearizeAll(bool fixLinearization) {
    double lastEnergyP = 0;
    double lastEnergyR = 0;
    double num = 0;

    std::vector<PointFrameResidual *> toRemove[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
        toRemove[i].clear();

    if (multiThreading) {
        treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
        lastEnergyP = treadReduce.stats[0];
    } else {
        Vec10 stats;
        linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
        lastEnergyP = stats[0];
    }

    setNewFrameEnergyTH(); ///< 设置新的帧能量阈值

    if (fixLinearization) {
        /// 同步 PointHessian 维护的残差状态 ([0]上次 [1]上上次) --> 直接保存上次和上上次的残差指针，然后查不好吗？
        for (PointFrameResidual *r : activeResiduals) {
            PointHessian *ph = r->point;
            if (ph->lastResiduals[0].first == r)
                ph->lastResiduals[0].second = r->state_state;
            else if (ph->lastResiduals[1].first == r)
                ph->lastResiduals[1].second = r->state_state;
        }

        /// 这里，对标志好待删除的残差(toRemove)中的，进行删除 (ph维护的lastResiduals 和 residuals)
        int nResRemoved = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            for (PointFrameResidual *r : toRemove[i]) {
                PointHessian *ph = r->point;
                // 删除不好的lastResiduals
                if (ph->lastResiduals[0].first == r)
                    ph->lastResiduals[0].first = 0;
                else if (ph->lastResiduals[1].first == r)
                    ph->lastResiduals[1].first = 0;

                for (unsigned int k = 0; k < ph->residuals.size(); k++)
                    if (ph->residuals[k] == r) {
                        ef->dropResidual(r->efResidual);                 ///< energy function丢掉残差
                        deleteOut<PointFrameResidual>(ph->residuals, k); ///< residuals删除第k个
                        nResRemoved++;
                        break;
                    }
            }
        }
    }

    /// 仅仅有lastEnergyP这部分有用 --> 代表的是所有残差线性化点处的能量之和（非FEJ）
    return Vec3(lastEnergyP, lastEnergyR, num);
}

/**
 * @brief 更新滑窗中各种状态量，并判断是否可以提前停止优化，并且将滑窗中的预计算内容和各种状态增量进行更新
 * @details
 *  1. 更新滑窗中的各种状态，根据优化得到的 delta_state 和 要求的步长
 *  2. 统计各种状态增量的均方值，用于后续判断是否优化已经充分
 *  3. 将滑动窗口中的预计算内容和各种状态增量进行更新
 * @param stepfacC  相机内参更新步长
 * @param stepfacT  位姿中平移向量更新步长
 * @param stepfacR  位姿中旋转向量更新步长
 * @param stepfacA  仿射参数更新步长
 * @param stepfacD  逆深度更新步长
 * @return true     优化充分，可以提前停止
 * @return false    优化不充分，需要继续优化
 */
bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD) {
    /// 定义状态更新步长
    Vec10 pstepfac;
    pstepfac.segment<3>(0).setConstant(stepfacT);
    pstepfac.segment<3>(3).setConstant(stepfacR);
    pstepfac.segment<4>(6).setConstant(stepfacA);

    float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

    float sumNID = 0;

    if (setting_solverMode & SOLVER_MOMENTUM) {
        Hcalib.setValue(Hcalib.value_backup + Hcalib.step); // 内参的值进行update
        for (FrameHessian *fh : frameHessians) {
            Vec10 step = fh->step;
            step.head<6>() += 0.5f * (fh->step_backup.head<6>()); //? 为什么加一半  答：这种解法很奇怪。。不管了

            fh->setState(fh->state_backup + step); // 位姿 光度 update
            sumA += step[6] * step[6];             // 光度增量平方
            sumB += step[7] * step[7];
            sumT += step.segment<3>(0).squaredNorm(); // 平移增量
            sumR += step.segment<3>(3).squaredNorm(); // 旋转增量

            for (PointHessian *ph : fh->pointHessians) {
                float step = ph->step + 0.5f * (ph->step_backup); //? 为啥加一半
                ph->setIdepth(ph->idepth_backup + step);
                sumID += step * step;               // 逆深度增量平方
                sumNID += fabsf(ph->idepth_backup); // 逆深度求和
                numID++;

                //* 逆深度没有使用FEJ
                ph->setIdepthZero(ph->idepth_backup + step);
            }
        }
    } else {
        /// 相机内参更新状态
        Hcalib.setValue(Hcalib.value_backup + stepfacC * Hcalib.step);

        /// 更新帧状态（帧位姿 和 帧仿射参数）
        for (FrameHessian *fh : frameHessians) {
            fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
            sumA += fh->step[6] * fh->step[6];
            sumB += fh->step[7] * fh->step[7];
            sumT += fh->step.segment<3>(0).squaredNorm();
            sumR += fh->step.segment<3>(3).squaredNorm();

            /// 逆深度更新，值得注意的是，逆深度的线性化状态也会同步更新 --> 代表着逆深度的FEJ部分，并没有起作用
            for (PointHessian *ph : fh->pointHessians) {
                ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
                sumID += ph->step * ph->step;
                sumNID += fabsf(ph->idepth_backup);
                numID++;

                ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
            }
        }
    }

    sumA /= frameHessians.size(); ///< 滑窗中仿射参数a更新的均方值
    sumB /= frameHessians.size(); ///< 滑窗中仿射参数b更新的均方值
    sumR /= frameHessians.size(); ///< 滑窗中旋转向量更新均方值
    sumT /= frameHessians.size(); ///< 滑窗中平移向量更新均方值
    sumID /= numID;               ///< 滑窗中点的更新量的均方值
    sumNID /= numID;              ///< 滑窗中点的上一次逆深度平均值

    EFDeltaValid = false;

    /// 将滑动窗口内点预计算的内容进行更新 @see FullSystem::setPrecalcValues
    setPrecalcValues();

    /// 当均方更新量小于某个阈值，则认为优化已经收敛，返回true
    return sqrtf(sumA) < 0.0005 * setting_thOptIterations && sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
           sqrtf(sumR) < 0.00005 * setting_thOptIterations && sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
}

/**
 * @brief 对待优化的状态初值进行copy
 * @details
 *  1. 相机内参状态
 *  2. 帧状态
 *      2.1 优化位姿
 *      2.2 仿射参数
 *  3. 点的逆深度
 * @param backupLastStep
 */
void FullSystem::backupState(bool backupLastStep) {
    /// 默认配置中，并不会进入这个SOLVER_MOMENTUM
    if (setting_solverMode & SOLVER_MOMENTUM) {
        if (backupLastStep) {
            Hcalib.step_backup = Hcalib.step;
            Hcalib.value_backup = Hcalib.value;
            for (FrameHessian *fh : frameHessians) {
                fh->step_backup = fh->step;
                fh->state_backup = fh->get_state();
                for (PointHessian *ph : fh->pointHessians) {
                    ph->idepth_backup = ph->idepth;
                    ph->step_backup = ph->step;
                }
            }
        } else {
            Hcalib.step_backup.setZero();
            Hcalib.value_backup = Hcalib.value;
            for (FrameHessian *fh : frameHessians) {
                fh->step_backup.setZero();
                fh->state_backup = fh->get_state();
                for (PointHessian *ph : fh->pointHessians) {
                    ph->idepth_backup = ph->idepth;
                    ph->step_backup = 0;
                }
            }
        }
    } else {
        Hcalib.value_backup = Hcalib.value; ///< 相机内参拷贝优化前值

        for (FrameHessian *fh : frameHessians) {
            fh->state_backup = fh->get_state(); ///< 拷贝帧的状态state
            for (PointHessian *ph : fh->pointHessians)
                ph->idepth_backup = ph->idepth; ///< 拷贝点的优化前逆深度
        }
    }
}

/**
 * @brief 将滑动窗口中，所有的状态，进行回滚，并将预计算的值回滚（根据旧状态重新计算）
 *
 * @details
 *  1. 相机内参回滚
 *  2. 帧状态回滚
 *  3. 点的逆深度回滚，且逆深度初始值同样更新，针对逆深度不使用FEJ
 *  4. 根据旧状态，重新计算预计算值，和状态更新量
 */
void FullSystem::loadSateBackup() {
    Hcalib.setValue(Hcalib.value_backup);
    for (FrameHessian *fh : frameHessians) {
        fh->setState(fh->state_backup);
        for (PointHessian *ph : fh->pointHessians) {
            ph->setIdepth(ph->idepth_backup);
            ph->setIdepthZero(ph->idepth_backup);
        }
    }

    EFDeltaValid = false;

    /// 根据旧状态，重新计算预计算值，和状态更新量
    setPrecalcValues();
}

/**
 * @brief 求解 HM 和 bM 对应的残差能量值
 *
 * @return double 返回的HM和bM对应的残差能量值
 */
double FullSystem::calcMEnergy() {
    if (setting_forceAceptStep)
        return 0;
    return ef->calcMEnergyF();
}

void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b) {
    printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n", res[0], sqrtf((float)(res[0] / (patternNum * ef->resInA))), ef->resInA, ef->resInM, a, b);
}

/**
 * @brief 优化求解 滑动窗口的 整个系统
 * @details
 *  1. 将未线性化的残差, 加入activeResiduals，就是新加入的帧构建的新残差（还没有进行线性化的残差）
 *  2. 对系统中，新添加的残差部分，进行线性化处理获取一些雅可比矩阵，获取所有非线性化残差的能量值
 *  3. 根据初始状态，计算当前状态下的，所有线性化残差的能量值 和 HM和bM对应的系统能量值 --> 这里是否有重复，我认为HM和bM部分，貌似包含部分线性化残差的能量值
 *  4. 计算新添加残差的Hfd部分，为后续的Hsc和bsc做准备
 *  5. 使用GN的阻尼方法，进行迭代优化求解
 *      5.1 备份当前的各个状态值(帧，相机和逆深度)
 *      5.2 求解系统的优化问题 H * deltax = b，考虑先验 + HM + HA @see FullSystem::solveSystem
 *      5.3 根据当前状态和步长，计算不同状态的更新量，然后更新状态，并判断是否可以终止优化，并进行窗口状态预计算 --> 因为状态优化更新了
 *      5.4 根据当前状态，重新计算新加入残差能量，线性化残差能量和HM和bM对应的系统能量值
 *      5.5 判断是否接受当前的更新状态，两种控制策略，强制接受和能量是否减小两种，满足一个即可
 *      5.6 如果不能接受上述的优化结果，则进行状态回滚
 *      5.7 优化的提前退出，两种控制策略，满足更新小量阈值，并且要求达到最小的迭代次数
 *  6. 设置最新帧的线性化状态
 *  7. 计算线性化点处，dlocal / dhost 和 dlocal / dtarget的值，用于后续local雅可比到global雅可比计算，并进行预计算和相对状态增量的更新
 *  8. 根据新加入的残差能量值，计算RMSE --> 均方根能量值 --> 精确到pattern上
 *  9. 把优化的结果给到shell，这里的shell中存储的状态，明显区别于frameHessians中的，是最终的优化状态，需加锁，保证Tracking部分不打架
 *
 * @note 这里在优化过程中，在计算能量时，总是会遗漏一部分内容，即除了以newFH为target的新加入残差能量、帧先验、相机内参先验、HM和bM对应的系统能量值以外
 * @note 还有 滑窗中除了activateResiduals中的残差能量、点先验能量
 * @note 除此之外，还做了部分的无用功，主要表现在，想要计算isLinearized=true状态下的残差能量和对应的HL和bL，在整个系统中，只有在标注点边缘化和点删除部分
 * @note 才会将待删除和待边缘化的内点对应的残差进行isLinearized状态的更新，并且后续点和点对应的残差都被删除掉了
 * @param mnumOptIts 输入的要求优化的迭代次数
 * @return float    输出的优化结束的 pattern粒度的均方根能量值
 */
float FullSystem::optimize(int mnumOptIts) {

    /// 要么0帧，要么大于2帧，0帧代表没有内容，不优化
    if (frameHessians.size() < 2)
        return 0;

    /// 当滑动窗口中的帧数小于3，要求优化迭代次数为20
    if (frameHessians.size() < 3)
        mnumOptIts = 20;

    /// 当滑动窗口中的帧数小于4，要求优化迭代次数为15
    if (frameHessians.size() < 4)
        mnumOptIts = 15;

    /// 1. 将未线性化的残差, 加入activeResiduals，就是新加入的帧构建的新残差（还没有进行线性化的残差,这里还有吗，线性化的部分？）
    activeResiduals.clear();
    int numPoints = 0; ///< 滑窗中点的数量
    int numLRes = 0;   ///< 滑窗中已经线性化的残差数量，应该一直为0才对
    for (FrameHessian *fh : frameHessians)
        for (PointHessian *ph : fh->pointHessians) {
            for (PointFrameResidual *r : ph->residuals) {
                if (!r->efResidual->isLinearized) {
                    activeResiduals.push_back(r);
                    r->resetOOB();
                } else
                    numLRes++;
            }
            numPoints++;
        }

    if (!setting_debugout_runquiet)
        printf("OPTIMIZE %d pts, %d active res, %d lin res!\n", ef->nPoints, (int)activeResiduals.size(), numLRes);

    /// 2. 对系统中，新添加的残差部分，进行线性化处理获取一些雅可比矩阵，获取所有非线性化残差的能量值
    Vec3 lastEnergy = linearizeAll(false); ///< 这里面，新加入的res，都是非active状态

    /// 3. 根据初始状态，计算当前状态下的，所有线性化残差的能量值 和 HM和bM对应的系统能量值
    double lastEnergyL = calcLEnergy(); ///< 滑窗系统中，所有线性化残差的能量值 ---> 这里只有帧状态先验和相机内参状态先验
    double lastEnergyM = calcMEnergy(); ///< 滑窗系统中，HM和bM对应的部分的能量值 --> 这里是HM和bM对应的先验（具有边缘化掉点的先验信息）

    /// 4. 计算新添加残差的Hfd部分
    if (multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
    else
        applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0); ///< 使用这个函数后，activeResiduals里面的残差才算activate

    if (!setting_debugout_runquiet) {
        printf("Initial Error       \t");
        printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
    }
    debugPlotTracking();

    /// 5. 使用GN的阻尼方法，进行迭代优化求解
    double lambda = 1e-1;
    float stepsize = 1;
    VecX previousX = VecX::Constant(CPARS + 8 * frameHessians.size(), NAN);
    for (int iteration = 0; iteration < mnumOptIts; iteration++) {
        /// 5.1 备份当前的各个状态值(帧，相机和逆深度)
        backupState(iteration != 0);

        /// 5.2 求解系统的优化问题 H * deltax = b，考虑先验 + HM + HA @see FullSystem::solveSystem
        solveSystem(iteration, lambda);
        double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
        previousX = ef->lastX;

        //? TUM自己的解法???
        if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) {
            float newStepsize = exp(incDirChange * 1.4);
            if (incDirChange < 0 && stepsize > 1)
                stepsize = 1;

            stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
            if (stepsize > 2)
                stepsize = 2;
            if (stepsize < 0.25)
                stepsize = 0.25;
        }

        /// 5.3 根据当前状态和步长，计算不同状态的更新量，然后更新状态，并判断是否可以终止优化，并进行窗口状态预计算 --> 因为状态优化更新了
        bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

        /// 5.4 根据当前状态，重新计算新加入残差能量，线性化残差能量和HM和bM对应的系统能量值
        Vec3 newEnergy = linearizeAll(false); ///< 对activateResiduals里面的残差计算能量值
        double newEnergyL = calcLEnergy();    ///< 滑窗系统中，所有线性化的残差对应的能量值，包含帧先验和内参先验，线性化部分对应的残差应该不存在
        double newEnergyM = calcMEnergy();    ///< 滑窗系统中，HM和bM对应的部分的能量值

        if (!setting_debugout_runquiet) {
            printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
                   (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM < lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
                   iteration, log10(lambda), incDirChange, stepsize);
            printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
        }

        /// 5.5 判断是否接受当前的更新状态，两种控制策略，强制接受和能量是否减小两种，满足一个即可
        if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM < lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {
            if (multiThreading)
                treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
            else
                applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0); ///< 残差中，对应的Hfd的更新

            lastEnergy = newEnergy;
            lastEnergyL = newEnergyL;
            lastEnergyM = newEnergyM;

            lambda *= 0.25;
        }
        /// 5.6 如果不能接受上述的优化结果，则进行状态回滚
        else {
            loadSateBackup();                 ///< 进行状态回滚
            lastEnergy = linearizeAll(false); ///< 使用旧状态，线性化新加入的所有残差，并计算能量值
            lastEnergyL = calcLEnergy();      ///< 根据旧状态，计算线性化部分的残差对应的能量值
            lastEnergyM = calcMEnergy();      ///< 根据旧状态，计算HM和bM部分的残差对应的能量值
            lambda *= 1e2;
        }

        /// 5.7 优化的提前退出，两种控制策略，满足更新小量阈值，并且要求达到最小的迭代次数
        if (canbreak && iteration >= setting_minOptIterations)
            break;
    }

    /// 6. 设置最新帧的线性化状态
    Vec10 newStateZero = Vec10::Zero();
    newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);
    frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam, newStateZero);

    EFDeltaValid = false;
    EFAdjointsValid = false;

    /// 7. 计算线性化点处，dlocal / dhost 和 dlocal / dtarget的值，用于后续local雅可比到global雅可比计算，并进行预计算和相对状态增量的更新
    ef->setAdjointsF(&Hcalib); ///< 对于新加入的帧，构建线性化点处的 dloacl / dglobal_host 和 dloacl / dglobal_target
    setPrecalcValues();        ///< 对于新计入的帧，构建预计算的值，并计算状态的相对增量（针对线性化点处，和先验点处）

    /// 8. 根据新加入的残差能量值，计算RMSE --> 均方根能量值 --> 精确到pattern上
    lastEnergy = linearizeAll(true); ///< 根据优化后的状态，计算新加入的残差对应的能量值
    if (!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2])) {
        /// 非法的能量值，导致跟跟踪丢失
        printf("KF Tracking failed: LOST!\n");
        isLost = true;
    }
    statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

    if (calibLog != 0) {
        (*calibLog) << Hcalib.value_scaled.transpose() << " " << frameHessians.back()->get_state_scaled().transpose() << " "
                    << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))) << " " << ef->resInM << "\n";
        calibLog->flush();
    }

    /// 9. 把优化的结果给到shell，这里的shell中存储的状态，明显区别于frameHessians中的，是最终的优化状态，需加锁，保证Tracking部分不打架
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        for (FrameHessian *fh : frameHessians) {
            fh->shell->camToWorld = fh->PRE_camToWorld;
            fh->shell->aff_g2l = fh->aff_g2l();
        }
    }

    debugPlotTracking();

    /// 返回均方根能量值
    return statistics_lastFineTrackRMSE;
}

/**
 * @brief 求解整个滑窗的优化问题
 * @details
 *  1. 获取由基组成的零空间 --> 零空间日志记录
 *  2. 求解滑窗系统 --> @see FullSystem::solveSystemF
 * @param iteration 输入的迭代数 --> 第几次迭代
 * @param lambda    输入的lambda值，用于标识可行域大小
 */
void FullSystem::solveSystem(int iteration, double lambda) {
    /// 1. 获取由基组成的零空间
    ef->lastNullspaces_forLogging = getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale, ef->lastNullspaces_affA, ef->lastNullspaces_affB);

    /// 2. 求解滑窗系统
    ef->solveSystemF(iteration, lambda, &Hcalib);
}

/**
 * @brief 计算滑窗系统中，所有已经线性化的残差的能量值 （使用最新的状态和FEJ，计算delta_rk --> 能量值）
 *
 * @return double   输出的系统能量值 (包含先验，但不包含HM和bM部分)
 */
double FullSystem::calcLEnergy() {
    if (setting_forceAceptStep)
        return 0;

    double Ef = ef->calcLEnergyF_MT();
    return Ef;
}

/**
 * @brief 删除pointHessian中，没有残差的点
 *
 * @details
 *  1. 遍历滑窗中所有点 ph，筛选出那些没有残差的点
 *  2. 将该点放到host帧的pointHessiansOut，标记为外点
 *  3. 将该点对应的EFPoint的状态设置为 PS_DROP 供后续删除
 *  4. 将 EnergyFunctional 中的设置为 PS_DROP 状态的EFPoint进行删除 @see EnergyFunctional::dropPointsF
 */
void FullSystem::removeOutliers() {
    int numPointsDropped = 0;
    for (FrameHessian *fh : frameHessians) {
        for (unsigned int i = 0; i < fh->pointHessians.size(); i++) {
            PointHessian *ph = fh->pointHessians[i];
            if (ph == 0)
                continue;

            if (ph->residuals.size() == 0) {
                fh->pointHessiansOut.push_back(ph);
                ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                fh->pointHessians[i] = fh->pointHessians.back();
                fh->pointHessians.pop_back();
                i--;
                numPointsDropped++;
            }
        }
    }
    ef->dropPointsF();
}

/**
 * @brief 获取各种状态的零空间，位姿零空间、尺度零空间、仿射a零空间、仿射b零空间
 *
 * @param nullspaces_pose       输出的位姿零空间 --> 由6组基组成
 * @param nullspaces_scale      输出的尺度零空间 --> 由1组基组成
 * @param nullspaces_affA       输出的仿射a零空间 --> 由1组基组成
 * @param nullspaces_affB       输出的仿射b零空间 --> 由1组基组成
 * @return std::vector<VecX>    输出的所有零空间 ---> 由9组基组成
 */
std::vector<VecX> FullSystem::getNullspaces(std::vector<VecX> &nullspaces_pose, std::vector<VecX> &nullspaces_scale, std::vector<VecX> &nullspaces_affA,
                                            std::vector<VecX> &nullspaces_affB) {
    nullspaces_pose.clear();  // size: 6; vec: 4+8*n
    nullspaces_scale.clear(); // size: 1;
    nullspaces_affA.clear();  // size: 1
    nullspaces_affB.clear();  // size: 1

    int n = CPARS + frameHessians.size() * 8; ///< 所有待求的状态维度
    std::vector<VecX> nullspaces_x0_pre;

    /// 1. 计算位姿的零空间 --> (4 + n * 8) * 6
    for (int i = 0; i < 6; i++) {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (FrameHessian *fh : frameHessians) {
            nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
            nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
            nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_pose.push_back(nullspace_x0);
    }

    /// 2. 计算仿射参数对应的零空间 --> (4 + n * 8) * 2
    for (int i = 0; i < 2; i++) {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (FrameHessian *fh : frameHessians) {
            nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) = fh->nullspaces_affine.col(i).head<2>(); //? 这个head<2>是为什么
            nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
            nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        if (i == 0)
            nullspaces_affA.push_back(nullspace_x0);
        if (i == 1)
            nullspaces_affB.push_back(nullspace_x0);
    }

    /// 3. 计算尺度对应的零空间 --> (4 + n * 8) * 1
    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (FrameHessian *fh : frameHessians) {
        nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
        nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
        nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_scale.push_back(nullspace_x0);

    return nullspaces_x0_pre;
}

} // namespace dso
