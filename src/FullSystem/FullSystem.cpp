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

#include "FullSystem/FullSystem.h"

#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/CoarseTracker.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

namespace dso {
// Hessian矩阵计数, 有点像 shared_ptr
int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;

/**
 * @brief FullSystem的构造函数
 * @note 构造完成后，建图县城已经开始了，使用的函数为 @see mappingLoop
 */
FullSystem::FullSystem() {

    int retstat = 0;

    /// 是否配置日志
    if (setting_logStuff) {
        // shell命令删除旧的文件夹, 创建新的
        retstat += system("rm -rf logs");
        retstat += system("mkdir logs");

        retstat += system("rm -rf mats");
        retstat += system("mkdir mats");

        //! calibLog.txt，目前猜测是pinhole的优化结果
        calibLog = new std::ofstream();
        calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
        calibLog->precision(12);

        //! numsLog.txt猜测是有关点的
        numsLog = new std::ofstream();
        numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
        numsLog->precision(10);

        //! coarseTrackingLog
        coarseTrackingLog = new std::ofstream();
        coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
        coarseTrackingLog->precision(10);

        //! eigenAllLog
        eigenAllLog = new std::ofstream();
        eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
        eigenAllLog->precision(10);

        //! eigenPLog
        eigenPLog = new std::ofstream();
        eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
        eigenPLog->precision(10);

        //! eigenALog
        eigenALog = new std::ofstream();
        eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
        eigenALog->precision(10);

        //! diagonal
        DiagonalLog = new std::ofstream();
        DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
        DiagonalLog->precision(10);

        //! variancesLog
        variancesLog = new std::ofstream();
        variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
        variancesLog->precision(10);

        //! nullspacesLog，猜测是零空间相关
        nullspacesLog = new std::ofstream();
        nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
        nullspacesLog->precision(10);
    } else {
        nullspacesLog = 0;
        variancesLog = 0;
        DiagonalLog = 0;
        eigenALog = 0;
        eigenPLog = 0;
        eigenAllLog = 0;
        numsLog = 0;
        calibLog = 0;
    }

    ///! 这里做这种断言的目的是什么呢？
    assert(retstat != 293847);

    //! 下面的这些变量含义还有待解析
    selectionMap = new float[wG[0] * hG[0]];
    coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
    coarseTracker = new CoarseTracker(wG[0], hG[0]);
    coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
    coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
    pixelSelector = new PixelSelector(wG[0], hG[0]);

    statistics_lastNumOptIts = 0;
    statistics_numDroppedPoints = 0;
    statistics_numActivatedPoints = 0;
    statistics_numCreatedPoints = 0;
    statistics_numForceDroppedResBwd = 0;
    statistics_numForceDroppedResFwd = 0;
    statistics_numMargResFwd = 0;
    statistics_numMargResBwd = 0;

    lastCoarseRMSE.setConstant(100); // 5维向量都=100

    currentMinActDist = 2;
    initialized = false;

    ef = new EnergyFunctional();
    ef->red = &this->treadReduce;

    isLost = false;
    initFailed = false;

    needNewKFAfter = -1;

    linearizeOperation = true;
    runMapping = true;
    mappingThread = boost::thread(&FullSystem::mappingLoop, this); /// 在构建FullSystem时候，建图线程已经开始了，函数为mappingLoop
    lastRefStopID = 0;

    minIdJetVisDebug = -1;
    maxIdJetVisDebug = -1;
    minIdJetVisTracker = -1;
    maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem() {
    blockUntilMappingIsFinished();

    // 删除new的ofstream
    if (setting_logStuff) {
        calibLog->close();
        delete calibLog;
        numsLog->close();
        delete numsLog;
        coarseTrackingLog->close();
        delete coarseTrackingLog;
        // errorsLog->close(); delete errorsLog;
        eigenAllLog->close();
        delete eigenAllLog;
        eigenPLog->close();
        delete eigenPLog;
        eigenALog->close();
        delete eigenALog;
        DiagonalLog->close();
        delete DiagonalLog;
        variancesLog->close();
        delete variancesLog;
        nullspacesLog->close();
        delete nullspacesLog;
    }

    delete[] selectionMap;

    for (FrameShell *s : allFrameHistory)
        delete s;
    for (FrameHessian *fh : unmappedTrackedFrames)
        delete fh;

    delete coarseDistanceMap;
    delete coarseTracker;
    delete coarseTracker_forNewKF;
    delete coarseInitializer;
    delete pixelSelector;
    delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH) {}

/**
 * @brief 设置G映射函数，包括G和G_inv，使用G_inv来求解G本身
 *
 * @param BInv 输入的 G_inv
 */
void FullSystem::setGammaFunction(float *BInv) {
    if (BInv == 0)
        return;

    /// 将G_inv拷贝到Binv中
    memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

    /// 根据G_inv，求解G本身
    for (int i = 1; i < 255; i++) {
        for (int s = 1; s < 255; s++) {
            if (BInv[s] <= i && BInv[s + 1] >= i) {
                Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                break;
            }
        }
    }
    Hcalib.B[0] = 0;
    Hcalib.B[255] = 255;
}

void FullSystem::printResult(std::string file) {
    boost::unique_lock<boost::mutex> lock(trackMutex);
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

    std::ofstream myfile;
    myfile.open(file.c_str());
    myfile << std::setprecision(15);

    for (FrameShell *s : allFrameHistory) {
        if (!s->poseValid)
            continue;

        if (setting_onlyLogKFPoses && s->marginalizedAt == s->id)
            continue;

        myfile << s->timestamp << " " << s->camToWorld.translation().transpose() << " " << s->camToWorld.so3().unit_quaternion().x() << " "
               << s->camToWorld.so3().unit_quaternion().y() << " " << s->camToWorld.so3().unit_quaternion().z() << " "
               << s->camToWorld.so3().unit_quaternion().w() << "\n";
    }
    myfile.close();
}

//@ 使用确定的运动模型对新来的一帧进行跟踪, 得到位姿和光度参数
Vec4 FullSystem::trackNewCoarse(FrameHessian *fh) {

    assert(allFrameHistory.size() > 0);
    // set pose initialization.

    for (IOWrap::Output3DWrapper *ow : outputWrapper)
        ow->pushLiveFrame(fh);

    FrameHessian *lastF = coarseTracker->lastRef; // 参考帧

    AffLight aff_last_2_l = AffLight(0, 0);
    //[ ***step 1*** ] 设置不同的运动状态
    std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
    printf("size: %d \n", lastF_2_fh_tries.size());
    if (allFrameHistory.size() == 2)
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
            lastF_2_fh_tries.push_back(SE3()); //? 这个size()不应该是0么
    else {
        FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];    // 上一帧
        FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3]; // 大上一帧
        SE3 slast_2_sprelast;
        SE3 lastF_2_slast;
        { // lock on global pose consistency!
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;  // 上一帧和大上一帧的运动
            lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld; // 参考帧到上一帧运动
            aff_last_2_l = slast->aff_g2l;
        }
        SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast. // 当前帧到上一帧 = 上一帧和大上一帧的

        //! 尝试不同的运动
        // get last delta-movement.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);                        // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast); // assume double motion (frame skipped)
        lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast);  // assume half motion.
        lastF_2_fh_tries.push_back(lastF_2_slast);                                               // assume zero motion.
        lastF_2_fh_tries.push_back(SE3());                                                       // assume zero motion FROM KF.

        //! 尝试不同的旋转变动
        // just try a TON of different initializations (all rotations). In the end,
        // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
        // also, if tracking rails here we loose, so we really, really want to avoid that.
        for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                       SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0))); // assume constant motion.
        }

        if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) // 有不和法的
        {
            lastF_2_fh_tries.clear();
            lastF_2_fh_tries.push_back(SE3());
        }
    }

    Vec3 flowVecs = Vec3(100, 100, 100);
    SE3 lastF_2_fh = SE3();
    AffLight aff_g2l = AffLight(0, 0);

    //! as long as maxResForImmediateAccept is not reached, I'll continue through the options.
    //! I'll keep track of the so-far best achieved residual for each level in achievedRes.
    //! 把到目前为止最好的残差值作为每一层的阈值
    //! If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.
    //! 粗层的能量值大, 也不继续优化了, 来节省时间

    Vec5 achievedRes = Vec5::Constant(NAN);
    bool haveOneGood = false;
    int tryIterations = 0;
    //! 逐个尝试
    for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
        //[ ***step 2*** ] 尝试不同的运动状态, 得到跟踪是否良好
        AffLight aff_g2l_this = aff_last_2_l; // 上一帧的赋值当前帧
        SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

        bool trackingIsGood = coarseTracker->trackNewestCoarse(fh, lastF_2_fh_this, aff_g2l_this, pyrLevelsUsed - 1,
                                                               achievedRes); // in each level has to be at least as good as the last try.
        tryIterations++;

        if (i != 0) {
            printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f "
                   "%f \n",
                   i, i, pyrLevelsUsed - 1, aff_g2l_this.a, aff_g2l_this.b, achievedRes[0], achievedRes[1], achievedRes[2], achievedRes[3], achievedRes[4],
                   coarseTracker->lastResiduals[0], coarseTracker->lastResiduals[1], coarseTracker->lastResiduals[2], coarseTracker->lastResiduals[3],
                   coarseTracker->lastResiduals[4]);
        }

        //[ ***step 3*** ] 如果跟踪正常, 并且0层残差比最好的还好留下位姿, 保存最好的每一层的能量值
        // do we have a new winner?
        if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
            // printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
            flowVecs = coarseTracker->lastFlowIndicators;
            aff_g2l = aff_g2l_this;
            lastF_2_fh = lastF_2_fh_this;
            haveOneGood = true;
        }

        // take over achieved res (always).
        if (haveOneGood) {
            for (int i = 0; i < 5; i++) {
                if (!std::isfinite((float)achievedRes[i]) ||
                    achievedRes[i] > coarseTracker->lastResiduals[i]) // take over if achievedRes is either bigger or NAN.
                    achievedRes[i] = coarseTracker->lastResiduals[i]; // 里面保存的是各层得到的能量值
            }
        }

        //[ ***step 4*** ] 小于阈值则暂停, 并且为下次设置阈值
        if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
            break;
    }

    if (!haveOneGood) {
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
        flowVecs = Vec3(0, 0, 0);
        aff_g2l = aff_last_2_l;
        lastF_2_fh = lastF_2_fh_tries[0];
    }

    //! 把这次得到的最好值给下次用来当阈值
    lastCoarseRMSE = achievedRes;

    //[ ***step 5*** ] 此时shell在跟踪阶段, 没人使用, 设置值
    // no lock required, as fh is not used anywhere yet.
    fh->shell->camToTrackingRef = lastF_2_fh.inverse();
    fh->shell->trackingRef = lastF->shell;
    fh->shell->aff_g2l = aff_g2l;
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

    if (coarseTracker->firstCoarseRMSE < 0)
        coarseTracker->firstCoarseRMSE = achievedRes[0]; // 第一次跟踪的平均能量值

    if (!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);

    if (setting_logStuff) {
        (*coarseTrackingLog) << std::setprecision(16) << fh->shell->id << " " << fh->shell->timestamp << " " << fh->ab_exposure << " "
                             << fh->shell->camToWorld.log().transpose() << " " << aff_g2l.a << " " << aff_g2l.b << " " << achievedRes[0] << " " << tryIterations
                             << "\n";
    }

    return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

/**
 * @brief 使用前端位姿估计成功的帧fh，使用极线搜索 + 不确定性像素误差 --> 更新滑动窗口中ImmaturePoint的逆深度范围 @see ImmaturePoint::traceOn
 *
 * @param fh  前端位姿估计成功的帧
 */
void FullSystem::traceNewCoarse(FrameHessian *fh) {
    boost::unique_lock<boost::mutex> lock(mapMutex);

    /// 用于统计的调试信息
    int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

    Mat33f K = Mat33f::Identity();

    /// 获取的都是 value_sacled里面的内容 --> value_sacled是第0层真实的K
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();

    /// 遍历 frameHessians 滑动窗口中的 关键帧
    for (FrameHessian *host : frameHessians) {
        SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;                ///< Tcr = Tcw * Twr
        Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse(); ///< K * Rcr * K_inv
        Vec3f Kt = K * hostToNew.translation().cast<float>();                     ///< K * tcr

        /// 根据 host，fh 和 曝光时间，计算仿射参数 aji, bji
        Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

        /// 未成熟点部分的处理（host帧中 i帧中）
        for (ImmaturePoint *ph : host->immaturePoints) {
            /// 未成熟的点，使用fh帧进行状态更新 @see ImmaturePoint::traceOn
            ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

            /// 统计量
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
                trace_good++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
                trace_badcondition++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
                trace_oob++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                trace_out++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
                trace_skip++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
                trace_uninitialized++;
            trace_total++;
        }
    }
}

/**
 * @brief 优化那些 满足激活条件 + 能够投影到 newestFH 上的未成熟点 @see FullSystem::optimizeImmaturePoint
 *
 * @param optimized     输出的优化成功的pointHessian
 * @param toOptimize    输入的待优化的pointHessian
 * @param min           多线程区间最小值
 * @param max           多线程区间最大值
 * @param stats         这个状态好像没有用到？
 * @param tid           线程id
 */
void FullSystem::activatePointsMT_Reductor(std::vector<PointHessian *> *optimized, std::vector<ImmaturePoint *> *toOptimize, int min, int max, Vec10 *stats,
                                           int tid) {
    /// 构建一个临时的 未成熟点 残差 （id为滑动窗口内的帧idx）
    ImmaturePointTemporaryResidual *tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
    for (int k = min; k < max; k++) {
        (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
    }
    delete[] tr;
}

/**
 * @brief 激活某些满足条件的 未成熟点
 * @details
 *  1. 根据 滑动窗口内目前存在的点 和 要求的点数目的差异，动态调节激活的距离阈值 currentMinActDist \in [0, 4]
 *  2. 根据滑动窗口内存在的 ph，在最新关键帧上构建距离场，使得滑窗内的host 投影到 newestFH 上的点pj，应该是足够均匀分布的
 *  3. 由于前面，会使用traceOn，在newestFH上进行未成熟点的跟踪，因此如果判断为outlier，则可以直接将 未成熟点删除了
 *  4. 对于那些非 outlier点，判断是否具有可以激活的潜能
 *      4.1 上次极线搜索距离在8个像素以内
 *      4.2 质量 (第二优 / 第一优) 要大于 3
 *      4.3 idepth_min + idepth_max > 0
 *  5. 对于 具有激活潜能的点，向 newestFH 上的金字塔1层投影，获取距离场对应的距离值
 *  6. 对未成熟点来讲，type--> 1, 2, 4分别代表0，1，2层提取的点--> 要求的距离会更加苛刻
 *  7. 对于那些投影过去看不到的点，认为是OOB点，直接删除
 *  8. 对于那些满足待激活条件，并且能够成功投影到 newestFH 的1层金字塔上，标注为待优化状态
 *  9. 使用类LM方法对 未成熟点的逆深度进行优化（构建host --> target 上的残差)
 *  10. 对于优化成功的点，设置为激活状态，并且更新其idepth，lastResiduals和residuals
 *  11. 对于优化不成功的点，进行删除。
 *  12. 最后对 参与优化的未成熟点，进行集体的删除
 *  13. 将优化成功部分的ph，添加到host frame 和 energy functional中
 */
void FullSystem::activatePointsMT() {
    /// 1. 动态调节阈值，通过目前现有点和目前点数目之间的差异，来动态的调整 激活距离阈值 currentMinActDist
    if (ef->nPoints < setting_desiredPointDensity * 0.66)
        currentMinActDist -= 0.8;
    if (ef->nPoints < setting_desiredPointDensity * 0.8)
        currentMinActDist -= 0.5;
    else if (ef->nPoints < setting_desiredPointDensity * 0.9)
        currentMinActDist -= 0.2;
    else if (ef->nPoints < setting_desiredPointDensity)
        currentMinActDist -= 0.1;

    if (ef->nPoints > setting_desiredPointDensity * 1.5)
        currentMinActDist += 0.8;
    if (ef->nPoints > setting_desiredPointDensity * 1.3)
        currentMinActDist += 0.5;
    if (ef->nPoints > setting_desiredPointDensity * 1.15)
        currentMinActDist += 0.2;
    if (ef->nPoints > setting_desiredPointDensity)
        currentMinActDist += 0.1;

    /// 针对动态调整后的距离阈值，要求 0 <= currentMinActDist <= 4
    if (currentMinActDist < 0)
        currentMinActDist = 0;
    if (currentMinActDist > 4)
        currentMinActDist = 4;

    if (!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n", currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

    /// 拿到刚刚添加进来的一帧frame
    FrameHessian *newestHs = frameHessians.back();

    /// 根据 newestFH 上的金字塔第一层，构建距离地图，将滑窗内的ph投影到 newestFH 的第一层上，构建距离地图 @see CoarseDistanceMap::makeDistanceMap
    coarseDistanceMap->makeK(&Hcalib);
    coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

    std::vector<ImmaturePoint *> toOptimize;
    toOptimize.reserve(20000);

    /// 处理未成熟点, 激活/删除/跳过
    for (FrameHessian *host : frameHessians) {
        if (host == newestHs)
            continue;

        SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;

        /// 第0层到1层
        Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
        Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

        for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1) {
            ImmaturePoint *ph = host->immaturePoints[i];
            ph->idxInImmaturePoints = i;

            /// 当最后一次Trace的时，判断 未成熟点 ph 为外点，则进行点的删除
            if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
                delete ph;
                host->immaturePoints[i] = 0;
                continue;
            }

            /// 激活条件是 1. 非外点 2. 非成熟点的上次极线搜索的距离小于8 3. 质量要大于3 4. idepth_min + idepth_max 要大于0
            bool canActivate = (ph->lastTraceStatus == IPS_GOOD || ph->lastTraceStatus == IPS_SKIPPED || ph->lastTraceStatus == IPS_BADCONDITION ||
                                ph->lastTraceStatus == IPS_OOB) &&
                               ph->lastTracePixelInterval < 8 && ph->quality > setting_minTraceQuality && (ph->idepth_max + ph->idepth_min) > 0;

            /// 针对当前无法激活的点，满足下面条件的，则进行删除处理，后续没有用了
            if (!canActivate) {
                /// 1. 如果点的 host 帧被标记为即将边缘化 （以后也激活不了了）
                /// 2. 或者当前点已经是OOB状态（两次外点，尺度变化大）
                if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB) {
                    delete ph;
                    host->immaturePoints[i] = 0;
                }
                continue;
            }

            /// 将 host上 未成熟的点投影到 newestFH 上，这里加上0.5为了实现round
            Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
            int u = ptp[0] / ptp[2] + 0.5f;
            int v = ptp[1] / ptp[2] + 0.5f;

            if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
                /// 这里在得到距离场的距离后，将ptp[0]的小数部分加到这里面的意义是什么？
                float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] + (ptp[0] - floorf((float)(ptp[0])));

                /// 针对 my_type ---> 1, 2, 4 ---> 0层选择，1层选择，2层选择
                /// 选择点的层级越高，距离条件越苛刻
                if (dist >= currentMinActDist * ph->my_type) {
                    coarseDistanceMap->addIntoDistFinal(u, v);
                    toOptimize.push_back(ph);
                }
            } else {
                /// 对于那些投影不到 newestFH，且满足激活条件的Point，则认为其是OOB的点，直接进行ph的删除
                delete ph;
                host->immaturePoints[i] = 0;
            }
        }
    }

    std::vector<PointHessian *> optimized;
    optimized.resize(toOptimize.size());

    /// 进行未成熟点的逆深度优化
    if (multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

    else
        activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

    /// 遍历待优化点和优化完成点，根据状态进行删除操作和向ef中添加的点和残差的操作
    for (unsigned k = 0; k < toOptimize.size(); k++) {
        PointHessian *newpoint = optimized[k];
        ImmaturePoint *ph = toOptimize[k];

        /// 对于优化成功的点，从host中删除immature，并将成功的点和对应的残差添加到ef中
        if (newpoint != 0 && newpoint != (PointHessian *)((long)(-1))) {
            newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0; ///< 将host保存的immature点置为nullptr
            newpoint->host->pointHessians.push_back(newpoint);           ///< 将host中pointhessians中添加newpoint
            ef->insertPoint(newpoint);                                   ///< 向能量函数中插入 newpoint

            for (PointFrameResidual *r : newpoint->residuals) ///< 向能量函数中，插入newpoint中构建的残差
                ef->insertResidual(r);

            assert(newpoint->efPoint != 0);
            delete ph; ///< 析构immaturePoint
        }
        /// 对于没有优化成功的点或者OOB没参加优化的点，从host中删除后，析构掉
        else if (newpoint == (PointHessian *)((long)(-1)) || ph->lastTraceStatus == IPS_OOB) {
            ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
            delete ph;
        } else {
            assert(newpoint == 0 || newpoint == (PointHessian *)((long)(-1)));
        }
    }

    /// 将滑窗中未成熟点的nullptr部分删除，要求维护的未成熟点中不能有nullptr
    for (FrameHessian *host : frameHessians) {
        for (int i = 0; i < (int)host->immaturePoints.size(); i++) {
            if (host->immaturePoints[i] == 0) {
                host->immaturePoints[i] = host->immaturePoints.back(); // 没有顺序要求, 直接最后一个给空的
                host->immaturePoints.pop_back();
                i--;
            }
        }
    }
}

void FullSystem::activatePointsOldFirst() { assert(false); }

/**
 * @brief 遍历整个滑动窗口点，判断点是否需要被丢掉或者是否需要被边缘化掉，仅做标记，不执行实际的HM和bM的转换
 * @details
 *  1. 将滑动窗口中，逆深度小于0，或者点的残差数量为0的点 标记为PS_DROP
 *  2. host 为待边缘化的帧 或者 被判断为需要被丢掉或者被边缘化掉 @see PointHessian::isOOB
 *      2.1 如果是一个内点，需要确定是 被丢掉 还是 被边缘化掉 @see PointHessian::isInlierNew
 *          2.1.1 对内点中的所有残差进行线性化，并求解Hfd
 *          2.1.2 对那些在滑窗中激活的残差，求解其线性化点处的残差值
 *          2.1.3 然后，根据逆深度的协方差矩阵，逆深度的协方差很大则标记为丢掉, 小的标记为边缘化掉
 *      2.2 如不是内点则标记为丢掉
 */
void FullSystem::flagPointsForRemoval() {
    assert(EFIndicesValid);

    std::vector<FrameHessian *> fhsToKeepPoints; ///< 没什么用
    std::vector<FrameHessian *> fhsToMargPoints;

    for (int i = 0; i < (int)frameHessians.size(); i++)
        if (frameHessians[i]->flaggedForMarginalization)
            fhsToMargPoints.push_back(frameHessians[i]);

    int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

    for (FrameHessian *host : frameHessians)
    {
        for (unsigned int i = 0; i < host->pointHessians.size(); i++) {
            PointHessian *ph = host->pointHessians[i];
            if (ph == 0)
                continue;

            /// 将滑动窗口中，逆深度小于0，或者点的残差数量为0的点 标记为PS_DROP
            if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
                host->pointHessiansOut.push_back(ph);
                ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                host->pointHessians[i] = 0;
                flag_nores++;
            }

            /// host 为待边缘化的帧 或者 被判断为需要被丢掉或者被边缘化掉
            else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization) {
                flag_oob++;

                /// 如果是一个内点，需要确定是 被丢掉 还是 被边缘化掉 @see PointHessian::isInlierNew
                if (ph->isInlierNew()) {
                    flag_in++;
                    int ngoodRes = 0;
                    for (PointFrameResidual *r : ph->residuals) {
                        r->resetOOB();         ///< 线性化残差前的准备
                        r->linearize(&Hcalib); ///< 线性化残差，求解中间变量的雅可比矩阵
                        r->efResidual->isLinearized = false;
                        r->applyRes(true); ///< 作用到EFResidual，并且求解残差的Hfd

                        /// 如果残差在线性化过程中，被认定为是内点，求解残差在 线性化点处 的残差值
                        if (r->efResidual->isActive()) {
                            r->efResidual->fixLinearizationF(ef);
                            ngoodRes++;
                        }
                    }

                    /// 如果逆深度的协方差很大则标记为丢掉, 小的标记为边缘化掉
                    if (ph->idepth_hessian > setting_minIdepthH_marg) {
                        flag_inin++;
                        ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
                        host->pointHessiansMarginalized.push_back(ph);
                    } else {
                        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                        host->pointHessiansOut.push_back(ph);
                    }
                }
                /// 不是内点则标记为丢掉
                else {
                    host->pointHessiansOut.push_back(ph);
                    ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                }

                host->pointHessians[i] = 0;
            }
        }

        /// 将被判断为 丢掉或边缘化掉的点，从host帧维护的pointHessians中删除掉（使用的尾部替代删除法）
        for (int i = 0; i < (int)host->pointHessians.size(); i++) {
            if (host->pointHessians[i] == 0) {
                host->pointHessians[i] = host->pointHessians.back();
                host->pointHessians.pop_back();
                i--;
            }
        }
    }
}

/**
 * @brief 前端的函数开始部分
 *
 * @param image 输入的光度矫正后的图像和曝光时间
 * @param id 	在mian的执行函数里面，代表的是图像的id/索引，如果要求速度的话，id可能不连续
 */
void FullSystem::addActiveFrame(ImageAndExposure *image, int id) {
    if (isLost)
        return;

    ///! 这种在整个前端过程中加锁的设计，是否是必要的，会不会造成时间上的浪费呢？
    boost::unique_lock<boost::mutex> lock(trackMutex);

    /// 创建FrameHessian和FrameShell, 并进行相应初始化, 并存储所有帧中的核心部分，FrameShell
    FrameHessian *fh = new FrameHessian();
    FrameShell *shell = new FrameShell();
    shell->camToWorld = SE3();
    shell->aff_g2l = AffLight(0, 0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
    fh->shell = shell;

    /// 在allFrameHistory中保存了frameShell的内容（所有的帧都在这里面）
    allFrameHistory.push_back(shell);

    fh->ab_exposure = image->exposure_time; ///< 保存曝光时间
    fh->makeImages(image->image, &Hcalib);  ///< 构建dpI 和 absSquaredGrad（原图的梯度平方和）

    /// 初始化部分
    if (!initialized) {
        /// 使用初始化器，构造第一帧
        if (coarseInitializer->frameID < 0) {
            coarseInitializer->setFirst(&Hcalib, fh);
            return;
        }

        /// 初始化器跟踪成功, 完成初始化
        if (coarseInitializer->trackFrame(fh, outputWrapper)) {
            /// 完成初始化
            initializeFromInitializer(fh);
            lock.unlock();
            deliverTrackedFrame(fh, true);
        }
        /// 还在初始化过程中，这时认定 frame 位姿是非法的
        else {
            fh->shell->poseValid = false;
            delete fh;
        }
    }
    /// 实现前端部分操作
    else {
        //[ ***step 5*** ] 对新来的帧进行跟踪, 得到位姿光度, 判断跟踪状态
        // =========================== SWAP tracking reference?. =========================
        if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
            // 交换参考帧和当前帧的coarseTracker
            boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
            CoarseTracker *tmp = coarseTracker;
            coarseTracker = coarseTracker_forNewKF;
            coarseTracker_forNewKF = tmp;
        }

        // TODO 使用旋转和位移对像素移动的作用比来判断运动状态
        Vec4 tres = trackNewCoarse(fh);
        if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3])) {
            printf("Initial Tracking failed: LOST!\n");
            isLost = true;
            return;
        }
        //[ ***step 6*** ] 判断是否插入关键帧
        bool needToMakeKF = false;
        if (setting_keyframesPerSecond > 0) // 每隔多久插入关键帧
        {
            needToMakeKF = allFrameHistory.size() == 1 || (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
        } else {
            Vec2 refToFh =
                AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure, coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

            // BRIGHTNESS CHECK
            needToMakeKF =
                allFrameHistory.size() == 1 ||
                setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +          // 平移像素位移
                        setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +  // TODO 旋转像素位移, 设置为0???
                        setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) + // 旋转+平移像素位移
                        setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) >
                    1 ||                                      // 光度变化大
                2 * coarseTracker->firstCoarseRMSE < tres[0]; // 误差能量变化太大(最初的两倍)
        }

        for (IOWrap::Output3DWrapper *ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);

        //[ ***step 7*** ] 把该帧发布出去
        lock.unlock();
        deliverTrackedFrame(fh, needToMakeKF);
        return;
    }
}

//@ 把跟踪的帧, 给到建图线程, 设置成关键帧或非关键帧
void FullSystem::deliverTrackedFrame(FrameHessian *fh, bool needKF) {
    /// 线性执行操作
    if (linearizeOperation) {

        /// 可视化的操作
        if (goStepByStep && lastRefStopID != coarseTracker->refFrameID) {
            MinimalImageF3 img(wG[0], hG[0], fh->dI);
            IOWrap::displayImage("frameToTrack", &img);
            while (true) {
                char k = IOWrap::waitKey(0);
                if (k == ' ')
                    break;
                handleKey(k);
            }
            lastRefStopID = coarseTracker->refFrameID;
        } else
            handleKey(IOWrap::waitKey(1));

        if (needKF)
            makeKeyFrame(fh);
        else
            makeNonKeyFrame(fh);
    } else {
        boost::unique_lock<boost::mutex> lock(trackMapSyncMutex); // 跟踪和建图同步锁
        unmappedTrackedFrames.push_back(fh);
        if (needKF)
            needNewKFAfter = fh->shell->trackingRef->id;
        trackedFrameSignal.notify_all();

        while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1) {
            mappedFrameSignal.wait(lock); // 当没有跟踪的图像, 就一直阻塞trackMapSyncMutex, 直到notify
        }

        lock.unlock();
    }
}

//@ 建图线程
void FullSystem::mappingLoop() {
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

    while (runMapping) {
        while (unmappedTrackedFrames.size() == 0) {
            trackedFrameSignal.wait(lock); // 没有图像等待trackedFrameSignal唤醒
            if (!runMapping)
                return;
        }

        FrameHessian *fh = unmappedTrackedFrames.front();
        unmappedTrackedFrames.pop_front();

        // guaranteed to make a KF for the very first two tracked frames.
        if (allKeyFramesHistory.size() <= 2) {
            lock.unlock(); // 运行makeKeyFrame是不会影响unmappedTrackedFrames的, 所以解锁
            makeKeyFrame(fh);
            lock.lock();
            mappedFrameSignal.notify_all(); // 结束前唤醒
            continue;
        }

        if (unmappedTrackedFrames.size() > 3)
            needToKetchupMapping = true;

        if (unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
        {
            lock.unlock();
            makeNonKeyFrame(fh);
            lock.lock();

            if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) // 太多了给处理掉
            {
                FrameHessian *fh = unmappedTrackedFrames.front();
                unmappedTrackedFrames.pop_front();
                {
                    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                    assert(fh->shell->trackingRef != 0);
                    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
                    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
                }
                delete fh;
            }

        } else {
            if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id) // 后面需要关键帧
            {
                lock.unlock();
                makeKeyFrame(fh);
                needToKetchupMapping = false;
                lock.lock();
            } else {
                lock.unlock();
                makeNonKeyFrame(fh);
                lock.lock();
            }
        }
        mappedFrameSignal.notify_all();
    }
    printf("MAPPING FINISHED!\n");
}

//@ 结束建图线程
void FullSystem::blockUntilMappingIsFinished() {
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    runMapping = false;
    trackedFrameSignal.notify_all();
    lock.unlock();

    mappingThread.join();
}

/**
 * @brief 针对非关键帧的处理，仅使用非关键帧fh，对滑动窗口内的ImmatruePoint进行 逆深度范围更新 @see ImmaturePoint::traceOn
 *
 * @param fh 输入的非关键帧
 */
void FullSystem::makeNonKeyFrame(FrameHessian *fh) {
    // needs to be set by mapping thread. no lock required since we are in mapping thread.
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        assert(fh->shell->trackingRef != 0);

        /// 滑动考虑到建图线程可能还在继续，因此这里frame的参考帧的位姿可能已经完成了更新（通过参考帧进行位姿更新）
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

        /// 在traceNewCoarse中，主要使用了 PRE_worldToCam，其实中间可能并不需要零空间的计算
        fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
    }

    traceNewCoarse(fh);
    delete fh;
}

/**
 * @brief 
 * 
 * @param fh 
 */
void FullSystem::makeKeyFrame(FrameHessian *fh) {

    /// 1. 设置当前估计的fh的位姿, 光度参数，计算FEJ 和 参数的零空间
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        assert(fh->shell->trackingRef != 0);

        /// 通过参考帧为桥梁，计算更新得到fh帧的位姿
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

        /// 设置状态估计的初始值 （Tcw, FEJ, nullspace, a, b）
        fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
    }

    /// 2. 把这一帧来更新滑动窗口内的未成熟点 @see FullSystem::traceNewCoarse
    traceNewCoarse(fh); // 更新未成熟点(深度未收敛的点)

    boost::unique_lock<boost::mutex> lock(mapMutex);

    /// 3. 标记滑动窗口中，需要被边缘化的帧 @see FullSystem::flagFramesForMarginalization
    flagFramesForMarginalization(fh);

    /// 4. 把当前帧添加到滑动窗口 和 能量函数中
    fh->idx = frameHessians.size();           ///< 设置滑窗id --> idx
    frameHessians.push_back(fh);              ///< 将当前帧加到滑窗中 frameHessians
    fh->frameID = allKeyFramesHistory.size(); ///< 设置关键帧id --> frameID
    allKeyFramesHistory.push_back(fh->shell); ///< 将fh添加到关键帧中
    ef->insertFrame(fh, &Hcalib);             ///< 将当前帧插入到能量函数中

    /// 设置滑动窗口内的预计算的值 @see setPrecalcValues
    setPrecalcValues();

    /// 遍历滑动窗口内的帧 fh1
    int numFwdResAdde = 0;
    for (FrameHessian *fh1 : frameHessians) {
        if (fh1 == fh)
            continue;

        /// 构造滑动窗口内存在的 ph 和 当前帧fh的残差 对象，只不过这里还没有计算残差值
        for (PointHessian *ph : fh1->pointHessians) {
            PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh); ///< 这里仅构造了残差对象，没有实际计算
            r->setState(ResState::IN);
            ph->residuals.push_back(r);                                                        ///< 添加到ph的残差列表中
            ef->insertResidual(r);                                                             ///< 添加到能量函数中
            ph->lastResiduals[1] = ph->lastResiduals[0];                                       ///< 设置上上个残差
            ph->lastResiduals[0] = std::pair<PointFrameResidual *, ResState>(r, ResState::IN); ///< 当前的设置为上一个
            numFwdResAdde += 1;
        }
    }

    /// 为了维护整个滑动窗口中，至少有2000个点，需要对滑动窗口内的关键帧中的部分未成熟点进行激活 @see FullSystem::activatePointsMT
    activatePointsMT(); ///< 考虑了Pi点投影到 newestFH 上pj 时的分布性，构建 newestFH上的 距离地图，根据动态参数进行待激活点的调整
    ef->makeIDX();      ///< 设置 ef 中的 idx 和 残差对应的idx<host id, target id>

    /// 优化滑窗内的关键帧 @see FullSystem::optimize
    fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
    float rmse = optimize(setting_maxOptIterations);

    // =========================== Figure Out if INITIALIZATION FAILED =========================
    //* 所有的关键帧数小于4，认为还是初始化，此时残差太大认为初始化失败
    if (allKeyFramesHistory.size() <= 4) {
        if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor) {
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed = true;
        }
        if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor) {
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed = true;
        }
        if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor) {
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed = true;
        }
    }

    if (isLost)
        return; // 优化后的能量函数太大, 认为是跟丢了

    /// 删除那些没有残差的点
    removeOutliers();

    /// 设置 coarseTracker_forNewKF 的内参和残差帧
    {
        boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
        coarseTracker_forNewKF->makeK(&Hcalib);
        coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
    }

    debugPlot("post Optimize");

    flagPointsForRemoval(); ///< 标记需要删除和边缘化的点
    ef->dropPointsF();      ///< 需要删除点，直接丢掉

    /// 获取当前系统的零空间
    getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale, ef->lastNullspaces_affA, ef->lastNullspaces_affB);

    /// 边缘化掉点, 加在HM, bM上
    ef->marginalizePointsF();

    makeNewTraces(fh, 0); ///< 提取像素点，用来构建新关键帧上的未成熟点

    for (IOWrap::Output3DWrapper *ow : outputWrapper) {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }

    /// 边缘化掉关键帧
    for (unsigned int i = 0; i < frameHessians.size(); i++)
        if (frameHessians[i]->flaggedForMarginalization) {
            marginalizeFrame(frameHessians[i]);
            i = 0;
        }

    printLogLine();
}

/**
 * @brief 从初始化器中获取一些参数内容，并完成初始化操作
 * @details
 *  1. 把第一帧设置为关键帧，并放入滑动窗口 frameHessians 和 能量函数 ef 中
 *  2. 求解firstFrame上，第0层上的iR的平均值，以平均值的倒数作为尺度因子
 *  3. 遍历第0层上的点pt，若0层点大于要求点，则进行随机删除到要求点的数目
 *      3.1 将Pnt --> immaturePoint --> PointHessian
 *      3.2 将PointHessian --> 放入ef中，并设置为激活状态
 *  4. 设置firstFrame和newFrame的状态（FEJ点状态存储，Tcw，Twc 和 状态的零空间求解）
 * @param newFrame 输入输出的初始化成功的帧
 */
void FullSystem::initializeFromInitializer(FrameHessian *newFrame) {
    boost::unique_lock<boost::mutex> lock(mapMutex);

    /// 把第一帧设置成关键帧, 加入队列, 加入EnergyFunctional
    FrameHessian *firstFrame = coarseInitializer->firstFrame; ///< 初始化器，第一帧
    firstFrame->idx = frameHessians.size();                   ///< idx，代表滑动窗口内的帧id
    frameHessians.push_back(firstFrame);                      ///< 将firstFrame放入frameHessians里面（所有滑动窗口）
    firstFrame->frameID = allKeyFramesHistory.size();         ///< frameID，所有历史关键帧id
    allKeyFramesHistory.push_back(firstFrame->shell);         ///< 将firstFrame放入allKeyFramesHistory里面 (所有关键帧记录)
    ef->insertFrame(firstFrame, &Hcalib);                     ///< energy function 中插入firstFrame
    setPrecalcValues();                                       ///< 设置预计算的值（关键帧与关键帧之间）

    firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);             ///< 预留20%的点作为收敛的点
    firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f); ///< 预留20%的点作为被边缘化的点
    firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);          ///< 预留20%的点作为被丢弃的外点

    /// 求第初始化器firstFrame上，第0层上的点期望的平均值，使用平均值的逆代表尺寸因子
    float sumID = 1e-5, numID = 1e-5;
    for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
        sumID += coarseInitializer->points[0][i].iR;
        numID++;
    }

    /// iR_new = iR * rescaleFactor --> iR_new 的均值为1，方差会变成 (re..) ^ 2 * var
    float rescaleFactor = 1 / (sumID / numID);

    /// 计算d / n，如果d / n > 1，则提取的点较少，如果d / n < 1，则提取的点较多
    float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    /// 调试信息，没什么用
    if (!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage, (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0]);

    /// 创建PointHessian, 点加入关键帧, 加入EnergyFunctional
    for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {

        /// d / n < 1 时，丢弃的点为 (1 - d / n) * n，剩下的点为 n - n + d --> d (期望点的数目)
        /// d / n > 1 时，不会进入continue中
        if (rand() / (float)RAND_MAX > keepPercentage)
            continue;

        /// 取出firstFrame第0层上的点 point
        Pnt *point = coarseInitializer->points[0] + i;

        /// 以point的像素坐标，第一帧，点的type（判断是哪一个金字塔层提取的），和系统的内参
        ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib);

        /// 如果进行pt点的差值[像素能量值color，dx，dy]，发现color如果是NAN，则删除该点
        if (!std::isfinite(pt->energyTH)) {
            delete pt;
            continue;
        }

        /// 将immaturePoint的逆深度范围设置为1，然后进行immaturePoint到PointHessian的转换
        pt->idepth_max = pt->idepth_min = 1;
        PointHessian *ph = new PointHessian(pt, &Hcalib);
        delete pt;

        ///! 在PointHessian的构造初始化阶段，并没有对energyTH进行重新计算，因此这里并不需要这部分的内容
        if (!std::isfinite(ph->energyTH)) {
            delete ph;
            continue;
        }

        ph->setIdepthScaled(point->iR * rescaleFactor); ///< 将point的iR * rescaleFactor作为初始的逆深度
        ph->setIdepthZero(ph->idepth);                  ///< 设置优化的初始值
        ph->hasDepthPrior = true;                       ///< 设置深度先验为true
        ph->setPointStatus(PointHessian::ACTIVE);       ///< 激活点ph

        firstFrame->pointHessians.push_back(ph); ///< 将firstFrame的点加入到pointHessians的集合中
        ef->insertPoint(ph);                     ///< energy function 中插入 ph 点
    }

    /// 设置第一帧和最新帧的待优化量, 参考帧
    SE3 firstToNew = coarseInitializer->thisToNext;
    firstToNew.translation() /= rescaleFactor;

    // really no lock required, as we are initializing.
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        firstFrame->shell->camToWorld = SE3();       ///< Twc
        firstFrame->shell->aff_g2l = AffLight(0, 0); ///< a, b
        firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(), firstFrame->shell->aff_g2l);
        firstFrame->shell->trackingRef = 0;
        firstFrame->shell->camToTrackingRef = SE3();

        newFrame->shell->camToWorld = firstToNew.inverse();
        newFrame->shell->aff_g2l = AffLight(0, 0);
        newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->aff_g2l);
        newFrame->shell->trackingRef = firstFrame->shell;
        newFrame->shell->camToTrackingRef = firstToNew.inverse();
    }

    initialized = true;
    printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

/**
 * @brief 提取某帧上的点，将其转换为未成熟点，供后续的Tracking中的优化和滑窗中的PH的后备力量
 * @see FullSystem::makeNewTraces
 *
 * @param newFrame  输入的待提取未成熟点的帧
 * @param gtDepth   没用到
 */
void FullSystem::makeNewTraces(FrameHessian *newFrame, float *gtDepth) {
    pixelSelector->allowFast = true;
    int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);

    newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
    newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
    newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

    for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
        for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
            int i = x + y * wG[0];
            if (selectionMap[i] == 0)
                continue;

            ImmaturePoint *impt = new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);
            if (!std::isfinite(impt->energyTH))
                delete impt;
            else
                newFrame->immaturePoints.push_back(impt);
        }
}

/**
 * @brief 更新滑动窗口中预计算的内容，线性化状态下的变换中间量和当前状态下的变换中间量，以及状态更新量
 *
 * @details
 *  1. 计算滑动窗口中，不同帧之间的变换中间量，包括线性化点处的变换中间量和最新状态的变换中间量 @see FrameFramePrecalc::set
 *  2. 计算滑动窗口中，各种状态的增量 @see EnergyFunctional::setDeltaF
 *
 */
void FullSystem::setPrecalcValues() {
    for (FrameHessian *fh : frameHessians) {
        fh->targetPrecalc.resize(frameHessians.size());
        for (unsigned int i = 0; i < frameHessians.size(); i++)

            /// 计算 fh 和 滑动窗口内所有帧之间的 优化前位姿、优化后位姿、优化后距离和 投影的中间量 @see FrameFramePrecalc::set
            fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
    }

    /// 计算滑动窗口中，各种状态的增量 @see EnergyFunctional::setDeltaF
    ef->setDeltaF(&Hcalib);
}

void FullSystem::printLogLine() {
    if (frameHessians.size() == 0)
        return;

    if (!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n", allKeyFramesHistory.back()->id,
               statistics_lastFineTrackRMSE, ef->resInA, ef->resInL, ef->resInM, (int)statistics_numForceDroppedResFwd, (int)statistics_numForceDroppedResBwd,
               allKeyFramesHistory.back()->aff_g2l.a, allKeyFramesHistory.back()->aff_g2l.b, frameHessians.back()->shell->id - frameHessians.front()->shell->id,
               (int)frameHessians.size());

    if (!setting_logStuff)
        return;

    if (numsLog != 0) {
        (*numsLog) << allKeyFramesHistory.back()->id << " " << statistics_lastFineTrackRMSE << " " << (int)statistics_numCreatedPoints << " "
                   << (int)statistics_numActivatedPoints << " " << (int)statistics_numDroppedPoints << " " << (int)statistics_lastNumOptIts << " " << ef->resInA
                   << " " << ef->resInL << " " << ef->resInM << " " << statistics_numMargResFwd << " " << statistics_numMargResBwd << " "
                   << statistics_numForceDroppedResFwd << " " << statistics_numForceDroppedResBwd << " " << frameHessians.back()->aff_g2l().a << " "
                   << frameHessians.back()->aff_g2l().b << " " << frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "
                   << (int)frameHessians.size() << " "
                   << "\n";
        numsLog->flush();
    }
}

void FullSystem::printEigenValLine() {
    if (!setting_logStuff)
        return;
    if (ef->lastHS.rows() < 12)
        return;

    MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
    MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
    int n = Hp.cols() / 8;
    assert(Hp.cols() % 8 == 0);

    // sub-select
    for (int i = 0; i < n; i++) {
        MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
        Hp.block(i * 6, 0, 6, n * 8) = tmp6;

        MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
        Ha.block(i * 2, 0, 2, n * 8) = tmp2;
    }
    for (int i = 0; i < n; i++) {
        MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
        Hp.block(0, i * 6, n * 8, 6) = tmp6;

        MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
        Ha.block(0, i * 2, n * 8, 2) = tmp2;
    }

    VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
    VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
    VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
    VecX diagonal = ef->lastHS.diagonal();

    std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
    std::sort(eigenP.data(), eigenP.data() + eigenP.size());
    std::sort(eigenA.data(), eigenA.data() + eigenA.size());

    int nz = std::max(100, setting_maxFrames * 10);

    if (eigenAllLog != 0) {
        VecX ea = VecX::Zero(nz);
        ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
        (*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        eigenAllLog->flush();
    }
    if (eigenALog != 0) {
        VecX ea = VecX::Zero(nz);
        ea.head(eigenA.size()) = eigenA;
        (*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        eigenALog->flush();
    }
    if (eigenPLog != 0) {
        VecX ea = VecX::Zero(nz);
        ea.head(eigenP.size()) = eigenP;
        (*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        eigenPLog->flush();
    }

    if (DiagonalLog != 0) {
        VecX ea = VecX::Zero(nz);
        ea.head(diagonal.size()) = diagonal;
        (*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        DiagonalLog->flush();
    }

    if (variancesLog != 0) {
        VecX ea = VecX::Zero(nz);
        ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
        (*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        variancesLog->flush();
    }

    std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
    (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
    for (unsigned int i = 0; i < nsp.size(); i++)
        (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " ";
    (*nullspacesLog) << "\n";
    nullspacesLog->flush();
}

void FullSystem::printFrameLifetimes() {
    if (!setting_logStuff)
        return;

    boost::unique_lock<boost::mutex> lock(trackMutex);

    std::ofstream *lg = new std::ofstream();
    lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
    lg->precision(15);

    for (FrameShell *s : allFrameHistory) {
        (*lg) << s->id << " " << s->marginalizedAt << " " << s->statistics_goodResOnThis << " " << s->statistics_outlierResOnThis << " " << s->movedByOpt;

        (*lg) << "\n";
    }

    lg->close();
    delete lg;
}

void FullSystem::printEvalLine() { return; }

} // namespace dso
