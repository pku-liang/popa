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

#define K           512
#define Ku          8191
#define Kl          8191

#include <assert.h>
#include "cm_rt.h"
#include "common/cm_rt_helpers.h"
#include "common/isa_helpers.h"

// For printing output
#include <stdio.h>
#include <iostream>

#define I           ((Ku + Kl + 1 + (VI*II-1)) / (VI*II))
#define TOTAL_I     (VI*II*I)
#define TOTAL_K     (KKK*KK*K)
#define SIZE_X      TOTAL_K
#define SIZE_A      TOTAL_K*TOTAL_K
#define SIZE_BAND_A TOTAL_K*TOTAL_I

#define SIZE_TOP      (TOTAL_K*II*I)
#define SIZE_RIGHT    (TOTAL_I*K)
#define SIZE_RESULTS  (TOTAL_I + TOTAL_K - 1)

using namespace std;

void check_correctness(float *A, float *x, float *result_top, float *result_right)
{
    float *result = (float*)malloc(sizeof(float) * SIZE_RESULTS);
    memset(result, 0, SIZE_RESULTS * sizeof(float));
    int addr_top = 0, addr_right = 0;
    for (int k = 0; k < K; k++)
      for (int i = 0; i < I; i++)
        for (int kk = 0; kk < KK; kk++)
          for (int ii = 0; ii < II; ii++) {
            int total_i = VI*ii + VI*II*i;
            int total_k = KKK*kk + KKK*KK*k;
            for (int kkk = 0; kkk < KKK; kkk++) {
                result[total_i + total_k + kkk] += result_top[addr_top++];
            }
            for (int vi = 0; vi < VI; vi++) {
                result[total_i + total_k + vi + KKK-1] += result_right[addr_right++];
            }
        }
    for (int i = 0; i < TOTAL_K; i++) {
        float golden = 0.0f;
        for (int k = 0; k < TOTAL_K; k++) {
            golden += A[i + k*TOTAL_K] * x[k];
        }
        if (fabs(golden - result[i + Ku]) >= 0.005*fabs(golden)) {
            printf("%d, %lf, %lf\n", i, golden, result[i + Ku]);
        }
        assert(fabs(golden - result[i + Ku]) < 0.005*fabs(golden));
    }
}

int main(int argc, char *argv[]) {
    // Creates a CmDevice from scratch.
    CmDevice *device = nullptr;
    unsigned int version = 0;
    cm_result_check(::CreateCmDevice(device, version));

    // Creates a CmProgram object consisting of the kernel loaded from the code buffer.
    CmProgram *program = nullptr;
    std::string isa_code = cm::util::isa::loadFile("gbmv_genx.isa");
    cm_result_check(device->LoadProgram(const_cast<char*>(isa_code.data()), isa_code.size(), program));

    // Creates the cmNBody kernel.
    CmKernel *kernel = nullptr;
    cm_result_check(device->CreateKernel(program, "kernel_fX", kernel));

    // Create a task queue
    CmQueue* cmd_queue = NULL;
    cm_result_check(device->CreateQueue( cmd_queue ));
    srand(time(NULL));

    float *A = (float*)malloc(sizeof(float) * SIZE_A);
    for (int i = 0; i < TOTAL_K; i++) {
        for (int k = 0; k < TOTAL_K; k++) {
            if (k - i > Ku || i - k > Kl) continue;
            A[i + k*TOTAL_K] = 1;
        }
    }
    float *banded_A = (float*)malloc(sizeof(float) * SIZE_BAND_A);
    memset(banded_A, 0, SIZE_BAND_A * sizeof(float));
    for (int k = 0; k < TOTAL_K; k++) {
        int j = Ku - k;
        for (int i = max(0, k-Ku); i < min(TOTAL_K, k+Kl+1); i++) {
            banded_A[(i+j) + k*TOTAL_I] = A[i + k*TOTAL_K];
        }
    }
    CmSurface2D *surf_a = nullptr;
    SurfaceIndex *surf_a_idx = nullptr;
    cm_result_check(device->CreateSurface2D(TOTAL_I, TOTAL_K, CM_SURFACE_FORMAT_R32F, surf_a));
    cm_result_check(surf_a->WriteSurface((unsigned char*)banded_A, NULL));
    cm_result_check(surf_a->GetIndex(surf_a_idx));

    float *x = (float*)malloc(sizeof(float) * SIZE_X);
    memset(x, 0, SIZE_X * sizeof(float));
    for (int k = 0; k < TOTAL_K; ++k) {
        x[k] = 1;
    }
    CmBuffer *surf_x = nullptr;
    SurfaceIndex *surf_x_idx = nullptr;
    cm_result_check(device->CreateBuffer(TOTAL_K*4, surf_x));
    cm_result_check(surf_x->WriteSurface((unsigned char*)x, NULL));
    cm_result_check(surf_x->GetIndex(surf_x_idx));

    CmBuffer *surf_right = nullptr;
    SurfaceIndex *surf_right_idx = nullptr;
    cm_result_check(device->CreateBuffer(SIZE_RIGHT*4, surf_right));
    cm_result_check(surf_right->GetIndex(surf_right_idx));

    CmBuffer *surf_top = nullptr;
    SurfaceIndex *surf_top_idx = nullptr;
    cm_result_check(device->CreateBuffer(SIZE_TOP*4, surf_top));
    cm_result_check(surf_top->GetIndex(surf_top_idx));

    int _A_extent_0 = TOTAL_I;
    cm_result_check(kernel->SetKernelArg(0, sizeof(int), &_A_extent_0));
    cm_result_check(kernel->SetKernelArg(1, sizeof(SurfaceIndex), surf_a_idx));
    cm_result_check(kernel->SetKernelArg(2, sizeof(SurfaceIndex), surf_right_idx));
    cm_result_check(kernel->SetKernelArg(3, sizeof(SurfaceIndex), surf_top_idx));
    cm_result_check(kernel->SetKernelArg(4, sizeof(SurfaceIndex), surf_x_idx));
    UINT64 kernel_ns = 0;
    UINT64 min_tkern = SIZE_MAX;
    // Creates a CmTask object.
    for (size_t i = 0; i < ITER; i++) {
        CmTask *task = nullptr;
        cm_result_check(device->CreateTask(task));
        cm_result_check(task->AddKernel(kernel));
        CmThreadGroupSpace *thread_group_space = nullptr;
        cm_result_check(device->CreateThreadGroupSpace(II, 1, I, K, thread_group_space));

        UINT64 tmp_kern_time;
        CmEvent *sync_event = nullptr;
        device->InitPrintBuffer();
        cm_result_check(cmd_queue->EnqueueWithGroup(task, sync_event, thread_group_space));
        cm_result_check(sync_event->WaitForTaskFinished(1000));
        cm_result_check(sync_event->GetExecutionTime( tmp_kern_time ));
        device->FlushPrintBuffer();
        kernel_ns += tmp_kern_time;
        if (tmp_kern_time < min_tkern) {
            min_tkern = tmp_kern_time;
        }
        if (ITER == 1) {
            float *result_top = (float*)malloc(sizeof(float) * SIZE_TOP);
            float *result_right = (float*)malloc(sizeof(float) * SIZE_RIGHT);
            cm_result_check(surf_top->ReadSurface((unsigned char *)result_top, sync_event));
            cm_result_check(surf_right->ReadSurface((unsigned char *)result_right, sync_event));
            check_correctness(A, x, result_top, result_right);
        }
        cm_result_check(device->DestroyTask(task));
    }
    double tkern = kernel_ns / ITER;
    double ops = (long)TOTAL_I*(long)TOTAL_K*2.0;

    cm_result_check(::DestroyCmDevice(device));

    if (ITER == 1) {
        printf("Pass!\n");
    } else {
        cout << "Size of matrix bandA: " << TOTAL_K << " * " << TOTAL_I << "\n";
        printf("Average GFlops: %lf\n", ops / tkern);
        printf("Max GFlops: %lf\n", ops / min_tkern);
    }

    return 0;
}
