/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the BSD-2-Clause Plus Patent License (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSDplusPatent
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: BSD-2-Clause-Patent
*******************************************************************************/
#define GPU     1

// Constant parameters (inner loop bounds) of the design
#include "const-parameters.h"

#define K           256
#define I           256

#include <assert.h>
#include "cm_rt.h"
#include "common/cm_rt_helpers.h"
#include "common/isa_helpers.h"

// For printing output
#include <stdio.h>
#include <iostream>

#define TOTAL_I     VI*III*II*I
#define TOTAL_K     KK*K
#define SIZE_X      TOTAL_K
#define SIZE_A      TOTAL_I*TOTAL_K
#define SIZE_Y      TOTAL_I

using namespace std;

void check_correctness(float *A, float *x, float *y)
{
    for (int i = 0; i < I; i++) {
        for (int ii = 0; ii < II; ii++) {
            for (int vi = 0; vi < III; vi++) {
                size_t total_i = vi + VI * ii + VI * III * i;
                float golden = 0.0f;
                for (int k = 0; k < TOTAL_K; k++)
                    golden += x[k] * A[total_i + k*TOTAL_I];
                assert(fabs(golden - y[total_i]) < 0.005*fabs(golden));
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Creates a CmDevice from scratch.
    CmDevice *device = nullptr;
    unsigned int version = 0;
    cm_result_check(::CreateCmDevice(device, version));

    // Creates a CmProgram object consisting of the kernel loaded from the code buffer.
    CmProgram *program = nullptr;
    std::string isa_code = cm::util::isa::loadFile("gemv_genx.isa");
    cm_result_check(device->LoadProgram(const_cast<char*>(isa_code.data()), isa_code.size(), program));

    // Creates the cmNBody kernel.
    CmKernel *kernel = nullptr;
    cm_result_check(device->CreateKernel(program, "kernel_uA", kernel));

    // Create a task queue
    CmQueue* cmd_queue = NULL;
    cm_result_check(device->CreateQueue( cmd_queue ));
    srand(time(NULL));

    float *x = (float*)malloc(sizeof(float) * SIZE_X);
    for (int i = 0; i < SIZE_X; ++i) {
        x[i] = rand();
    }
    CmBuffer *surf_x = nullptr;
    SurfaceIndex *surf_x_idx = nullptr;
    cm_result_check(device->CreateBuffer(TOTAL_K*4, surf_x));
    cm_result_check(surf_x->WriteSurface((unsigned char*)x, NULL));
    cm_result_check(surf_x->GetIndex(surf_x_idx));

    float *A = (float*)malloc(sizeof(float) * SIZE_A);
    for (int i = 0; i < SIZE_A; ++i) {
        A[i] = rand();
    }
    CmSurface2D *surf_A = nullptr;
    SurfaceIndex *surf_A_idx = nullptr;
    cm_result_check(device->CreateSurface2D(TOTAL_I, TOTAL_K, CM_SURFACE_FORMAT_R32F, surf_A));
    cm_result_check(surf_A->WriteSurface((unsigned char*)A, NULL));
    cm_result_check(surf_A->GetIndex(surf_A_idx));

    CmBuffer *surf_y = nullptr;
    SurfaceIndex *surf_y_idx = nullptr;
    cm_result_check(device->CreateBuffer(TOTAL_I*4, surf_y));
    cm_result_check(surf_y->GetIndex(surf_y_idx));

    int _A_extent_1 = TOTAL_K;
    cm_result_check(kernel->SetKernelArg(0, sizeof(int), &_A_extent_1));
    cm_result_check(kernel->SetKernelArg(1, sizeof(SurfaceIndex), surf_A_idx));
    cm_result_check(kernel->SetKernelArg(2, sizeof(SurfaceIndex), surf_y_idx));
    cm_result_check(kernel->SetKernelArg(3, sizeof(SurfaceIndex), surf_x_idx));
    UINT64 kernel_ns = 0;
    UINT64 min_tkern = SIZE_MAX;
    // Creates a CmTask object.
    for (size_t i = 0; i < ITER; i++) {
        CmTask *task = nullptr;
        cm_result_check(device->CreateTask(task));
        cm_result_check(task->AddKernel(kernel));
        CmThreadGroupSpace *thread_group_space = nullptr;
        cm_result_check(device->CreateThreadGroupSpace(II, 1, I, 1, thread_group_space));

        UINT64 tmp_kern_time;
        CmEvent *sync_event = nullptr;
        device->InitPrintBuffer();
        double host_start = getTimeStamp();
        cm_result_check(cmd_queue->EnqueueWithGroup(task, sync_event, thread_group_space));
        cm_result_check(sync_event->WaitForTaskFinished(1000));
        double host_end = getTimeStamp();
        cm_result_check(sync_event->GetExecutionTime( tmp_kern_time ));
        device->FlushPrintBuffer();
        kernel_ns += tmp_kern_time;
        if (tmp_kern_time < min_tkern) {
            min_tkern = tmp_kern_time;
        }
        if (ITER == 1) {
            float *y = (float*)malloc(sizeof(float) * SIZE_Y);
            cm_result_check(surf_y->ReadSurface((unsigned char *)y, sync_event));
            check_correctness(A, x, y);
        }
        cm_result_check(device->DestroyTask(task));
    }
    double tkern = kernel_ns / ITER;
    double ops = (long)TOTAL_I*(long)TOTAL_K*2.0;

    cm_result_check(::DestroyCmDevice(device));

    if (ITER == 1) {
        printf("Pass!\n");
    } else {
        cout << "Size of matrix A: " << TOTAL_I << " * " << TOTAL_K << "\n";
        printf("Average GFlops: %lf\n", ops / tkern);
        printf("Max GFlops: %lf\n", ops / min_tkern);
    }

    return 0;
}
