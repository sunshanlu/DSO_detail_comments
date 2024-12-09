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

#include <locale.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"

#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"
#include <boost/thread.hpp>

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/NumType.h"

#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start = 0;
int end = 100000;
bool prefetch = false;
float playbackSpeed = 0; // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping).
                         // otherwise, factor on timestamps.
bool preload = false;
bool useSampleOutput = false;

int mode = 0;

bool firstRosSpin = false;

using namespace dso;

void my_exit_handler(int s) {
    printf("Caught signal %d\n", s);
    exit(1);
}

void exitThread() {
    /// 这里的struct是用来指定sigaction是一个struct类型的函数，这是因为sigaction是从C语言中引入的
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    firstRosSpin = true;
    while (true)
        pause();
}

void settingsDefault(int preset) {
    printf("\n=============== PRESET Settings: ===============\n");
    if (preset == 0 || preset == 1) {
        printf("DEFAULT settings:\n"
               "- %s real-time enforcing\n"
               "- 2000 active points\n"
               "- 5-7 active frames\n"
               "- 1-6 LM iteration each KF\n"
               "- original image resolution\n",
               preset == 0 ? "no " : "1x");

        playbackSpeed = (preset == 0 ? 0 : 1);
        preload = preset == 1;
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        setting_logStuff = false;
    }

    if (preset == 2 || preset == 3) {
        printf("FAST settings:\n"
               "- %s real-time enforcing\n"
               "- 800 active points\n"
               "- 4-6 active frames\n"
               "- 1-4 LM iteration each KF\n"
               "- 424 x 320 image resolution\n",
               preset == 0 ? "no " : "5x");

        playbackSpeed = (preset == 2 ? 0 : 5);
        preload = preset == 3;
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations = 4;
        setting_minOptIterations = 1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}

/**
 * @brief 处理输入参数的字符串
 * @details
 * 	1. sampleoutput，1代表简单输出
 * 	2. quiet，1代表静默模式
 * 	3. preset，参数代表含义较多，@see settingsDefault
 * 	4. rec, 0代表禁止重新配置，猜测应该和pangolin界面有关
 * 	5. noros，1代表禁止ros和reconfigure
 *  6. nolog，1会使setting_logStuff置为true
 * 	7. reverse，1代表数据集的反向读取，会将reverse置为true
 *  8. nogui，1代表禁用gui，会将disableAllDsiplay置为true
 *  9. nomt，1代表禁用多线程，会将multiThreading置为false
 * 	10. prefetch，这个参数貌似没用上
 * 	11. start，数据读取的开始部分
 *  12. end，数据读取的结束部分
 * 	13. files，
 *  14. calib，
 *  15. vignette，
 *  16. gamma，
 *  17. rescale，这个参数貌似没有作用
 *  18. speed，以某个速度来播放数据，0代表线性化读取和处理，playbackSpeed实际控制速度的参数
 *  19. save，是否保存图片，若为1，则在当前运行路径下创建image_out，后续会将普通帧和关键帧信息输出到里面
 *  20. mode，根据mode取值，设置与光度矫正部分的相关参数
 * 		0代表有光度修正模型，vignette和gamma都存在
 * 		1代表没有光度矫正，setting_affineOptModeA和setting_affineOptModeB设为0
 * 		!3代表完美图像，不需要光度模型，并且可以固定光度参数a和b，setting_minGradHistAdd修改为3，但是目前并不止参数setting_minGradHistAdd的意义
 *
 * @param arg 参数
 */
void parseArgument(char *arg) {
    int option;
    float foption;
    char buf[1000];

    if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
        if (option == 1) {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "quiet=%d", &option)) {
        if (option == 1) {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "preset=%d", &option)) {
        settingsDefault(option);
        return;
    }

    if (1 == sscanf(arg, "rec=%d", &option)) {
        if (option == 0) {
            disableReconfigure = true;
            printf("DISABLE RECONFIGURE!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "noros=%d", &option)) {
        if (option == 1) {
            disableROS = true;
            disableReconfigure = true;
            printf("DISABLE ROS (AND RECONFIGURE)!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "nolog=%d", &option)) {
        if (option == 1) {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "reverse=%d", &option)) {
        if (option == 1) {
            reverse = true;
            printf("REVERSE!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nogui=%d", &option)) {
        if (option == 1) {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nomt=%d", &option)) {
        if (option == 1) {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "prefetch=%d", &option)) {
        if (option == 1) {
            prefetch = true;
            printf("PREFETCH!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "start=%d", &option)) {
        start = option;
        printf("START AT %d!\n", start);
        return;
    }
    if (1 == sscanf(arg, "end=%d", &option)) {
        end = option;
        printf("END AT %d!\n", end);
        return;
    }

    if (1 == sscanf(arg, "files=%s", buf)) {
        source = buf;
        printf("loading data from %s!\n", source.c_str());
        return;
    }

    if (1 == sscanf(arg, "calib=%s", buf)) {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }

    if (1 == sscanf(arg, "vignette=%s", buf)) {
        vignette = buf;
        printf("loading vignette from %s!\n", vignette.c_str());
        return;
    }

    if (1 == sscanf(arg, "gamma=%s", buf)) {
        gammaCalib = buf;
        printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
        return;
    }

    if (1 == sscanf(arg, "rescale=%f", &foption)) {
        rescale = foption;
        printf("RESCALE %f!\n", rescale);
        return;
    }

    if (1 == sscanf(arg, "speed=%f", &foption)) {
        playbackSpeed = foption;
        printf("PLAYBACK SPEED %f!\n", playbackSpeed);
        return;
    }

    if (1 == sscanf(arg, "save=%d", &option)) {
        if (option == 1) {
            debugSaveImages = true;
            /// 这里进行了两次，可能是防止可能出现的失败情况吧！
            if (42 == system("rm -rf images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("rm -rf images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            printf("SAVE IMAGES!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "mode=%d", &option)) {

        mode = option;
        if (option == 0) {
            printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
        if (option == 1) {
            printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
        }
        if (option == 2) {
            printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd = 3;
        }
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
}

int main(int argc, char **argv) {
    // setlocale(LC_ALL, "");
    /// 处理命令行参数的部分逻辑，即若有参数输入，则会修改默认配置内容
    for (int i = 1; i < argc; i++)
        parseArgument(argv[i]);

    // 用于处理ctl+c信号，使用了C语言中的sigaction结构体实现
    boost::thread exThread = boost::thread(exitThread);

    // 构造reader用到的参数默认值都是""，可以通过parseArgument函数传递内容
    ImageFolderReader *reader = new ImageFolderReader(source, calib, gammaCalib, vignette);

    // 计算金字塔每层的w、h、K、K_inv
    reader->setGlobalCalibration();

    /// 判断光度标定模型和给定的setting配置是否有问题，如果要求有光度模型，但是并没有Gamma函数，则error
    if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0) {
        printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }

    /// 如果reverse为true，则进行开始和结束之间的对调，并进行linc的配置，即读取间隔
    int lstart = start;
    int lend = end;
    int linc = 1;
    if (reverse) {
        printf("REVERSE!!!!");
        lstart = end - 1;
        if (lstart >= reader->getNumImages())
            lstart = reader->getNumImages() - 1;
        lend = start;
        linc = -1;
    }

    FullSystem *fullSystem = new FullSystem();
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed == 0);

    IOWrap::PangolinDSOViewer *viewer = 0;
    if (!disableAllDisplay) {
        viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if (useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        /// 下面的操作是将idsToPlay和timesToPlayAt进行填充
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        for (int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc) {
            idsToPlay.push_back(i);
            if (timesToPlayAt.size() == 0) {
                timesToPlayAt.push_back((double)0);
            } else {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
                timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / playbackSpeed);
            }
        }

        std::vector<ImageAndExposure *> preloadedImages;

        /// 提前加载所有图像的操作，preset为1或3会触发preload为true
        if (preload) {
            printf("LOADING ALL IMAGES!\n");
            for (int ii = 0; ii < (int)idsToPlay.size(); ii++) {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL); ///< 获取时间戳，精度在微妙级
        clock_t started = clock();     ///< cpu时间，通常用于性能分析
        double sInitializerOffset = 0;

        for (int ii = 0; ii < (int)idsToPlay.size(); ii++) {

            /// 统计DSO系统的初始化时刻，inited time offset
            if (!fullSystem->initialized) {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];

            ImageAndExposure *img;
            if (preload)
                img = preloadedImages[ii];
            else{
                /// 这里的reader的getImage函数会对原始图像进行光度矫正，并保存曝光时间和时间戳，其余不管
                img = reader->getImage(i);
            }

            bool skipFrame = false;

            /// 用于控制帧是否实时处理或者以某速度处理的代码内容
            if (playbackSpeed != 0) {
                struct timeval tv_now;
                gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) +
                                                           (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

                if (sSinceStart < timesToPlayAt[ii])
                    usleep((int)((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
                else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2)) {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame = true;
                }
            }

            if (!skipFrame)
                fullSystem->addActiveFrame(img, i);

            delete img;
            /// 系统重置的两种情况：
            /// 	1. 在初始化失败，并且ii已经走到第250帧时，依然进行重置
            /// 	2. 当有系统重置请求的时候，进行系统重置
            if (fullSystem->initFailed || setting_fullResetRequested) {
                if (ii < 250 || setting_fullResetRequested) {
                    printf("RESETTING!\n");

                    std::vector<IOWrap::Output3DWrapper *> wraps = fullSystem->outputWrapper;
                    delete fullSystem;

                    for (IOWrap::Output3DWrapper *ow : wraps)
                        ow->reset();

                    fullSystem = new FullSystem();
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed == 0);

                    fullSystem->outputWrapper = wraps;

                    setting_fullResetRequested = false;
                }
            }

            /// 如果系统跟踪丢失，系统直接退出（gg）
            if (fullSystem->isLost) {
                printf("LOST!!\n");
                break;
            }
        }

        /// 前端运行结束后，结束建图线程的操作
        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        fullSystem->printResult("result.txt");

        /// 后续都是一些log工作，不重要
        int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                                                           (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
        printf("\n======================"
               "\n%d Frames (%.1f fps)"
               "\n%.2fms per frame (single core); "
               "\n%.2fms per frame (multi core); "
               "\n%.3fx (single core); "
               "\n%.3fx (multi core); "
               "\n======================\n\n",
               numFramesProcessed, numFramesProcessed / numSecondsProcessed,
               MilliSecondsTakenSingle / numFramesProcessed, MilliSecondsTakenMT / (float)numFramesProcessed,
               1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
               1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        // fullSystem->printFrameLifetimes();
        if (setting_logStuff) {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC * reader->getNumImages()) << " "
                  << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) /
                         (float)reader->getNumImages()
                  << "\n";
            tmlog.flush();
            tmlog.close();
        }
    });

    if (viewer != 0)
        viewer->run();

    runthread.join();

    for (IOWrap::Output3DWrapper *ow : fullSystem->outputWrapper) {
        ow->join();
        delete ow;
    }

    printf("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    printf("DELETE READER!\n");
    delete reader;

    printf("EXIT NOW!\n");
    return 0;
}
