# POPA
A framework for portable performance across FPGAs and GPUs.

# Introduction
This paper aims at high and portable performance for tensor computations across spatial (e.g., FPGAs) and vector architectures (e.g., GPUs). The state-of-the-art usually address performance portability across vector architectures (CPUs and GPUs). However, they either miss FPGAs or do not achieve high performance. Without a common architectural abstraction, they program and optimize spatial and vector devices separately, causing low portability. We propose a unified programming framework, POPA, which achieves portability via architectural abstraction and performance via specialization. A parallel dataflow machine is proposed as a unified, abstract hardware target that hides differences of concrete architectures. The machine consists of software-defined systolic arrays and a tensor-specific cache hierarchy, which captures pipeline parallelism and customizable memories on FPGAs, as well as multithreading parallelism on GPUs. The machine is specified in a unified programming model as two dataflow graphs for scheduling compute and data movement, respectively. A compiler then specializes the abstract machine to exploit the properties of FPGAs and GPUs, bridging the gap between the abstract machine and a concrete architecture. We evaluate POPA on several Intel FPGAs and GPUs with high-profile tensor kernels, and this is the first system that achieves >=80% performance of expert-written code or machine peak across architectures, to the best of our knowledge.

# Quick Start Guide (Artifact Evaluation)
1. Open a DevCloud account by following the steps below.
2. Clone our repository into two separate directories:
   
   ```
   git clone -b popa git@github.com:haoxiaochen/t2sp.git t2sp
   git clone -b popa git@github.com:haoxiaochen/t2sp.git t2sp-s10
   ```
   - This step requires setting up your GitHub certificate, needed when fetching pre-generated bitstreams from Git LFS. You have the option to clone using an HTTPS link and manually enter your password, as shown below.
   - You may encounter a checkout failure due to git-lfs not found. The next step will install it and perform the checkout again.
4. Install dependencies and compile. Since A10 and S10 machines have different system environments, it is recommended to install them separately:
  
   ```
   qsub -q batch@v-qsvr-fpga -l nodes=1:arria10:ppn=2 -d $HOME/t2sp $HOME/t2sp/install-tools.sh
   qsub -q batch@v-qsvr-fpga -l nodes=1:stratix10:ppn=2 -d $HOME/t2sp-s10 $HOME/t2sp/install-tools.sh
   ```
   A job is submitted. You can check its completion status with `qstatus`.
5. Run our tests on FPGAs with pre-generated bitstreams:

   ```
   cd t2s/tests/popa
   ./devcloud_jobs.sh [a10|s10] bitstream
   ```
   This will submit 6 jobs to run GEMM, Conv, Capsule, PairHMM, GEMV, and GBMV. After the jobs are completed, you can find a file named 'job.sh.o[job_id]'.
   Alternatively, you can log into a DevCloud compute node and test each one separately:

   ```
   devcloud_login
   cd t2s/tests/popa
   ./test.sh devcloud gemm a10 large hw bitstream
   ```
   You might be prompted to enter your Github username and password if you choose to clone using an HTTPS link.
   Open this file, and you will see:

   ```
   ------------------- Testing devcloud gemm a10 large hw bitstream
   ```
   indicating a GEMM test. At the end of this file, you can see:

   ```
   GFlops: 620.383645
   ```
   demonstrating the achieved throughput.
7. [Optional] Run our test on FPGAs by synthesizing a bitstream:

   ```
   ./devcloud_job.sh devcloud gemm [a10|s10] large hw
   ```
   This will submit a job to synthesize our GEMM.
8. Run our tests on GEN 9 GPU:

   ```
   ./devcloud_jobs.sh gen9
   ```
   Unfortunately, GEN 12 GPU has been retired from DevCloud.


# [DevCloud] Open an account (once)

 + Register at the [Intel's FPGA DevCloud](https://software.intel.com/content/www/us/en/develop/tools/devcloud/fpga.html). This will enable access to both the FPGAs and the GPUs in the cloud. Currently, the cloud offers Arria 10  and Stratix 10 FPGAs, and GEN 9.5 (Intel UHD Graphics P630).

 + Connect to DevCloud by following the [document](https://devcloud.intel.com/oneapi/documentation/connect-with-ssh-linux-macos/). Now you are at the **head node** named `login-2`.
   
 + Add the following to your .bashrc:
   
   ```
    if [ -f /data/intel_fpga/devcloudLoginToolSetup.sh ]; then
        source /data/intel_fpga/devcloudLoginToolSetup.sh
    fi
   ```
   Then
   ```
    source .bashrc
   ```

# Open a terminal on a compute node

[DevCloud] from the head node, log into a **compute node**:

+ FPGA: 
  ```
    devcloud_login
  ```
    Choose         
    ```
    6) Enter Specific Node Number
    ```
    Enter the name of a node with Arria 10 Release 1.2.1, or with Stratix 10.

+ GPU: to request a compute node with GEN 9.5 or GEN 12,
  
    ```
    qsub -I -l nodes=1:gen9:ppn=2  
    ```
    
    or 
    
    ```
    qsub -I -l nodes=1:iris_xe_max:ppn=2
    ```

# Set up the environment (whenever a terminal is open)

```
cd $HOME/t2sp
source ./setenv.sh (devcloud|local) (fpga|gpu)
```
The options say if you are working on DevCloud or locally, and to use an FPGA or a GPU. 

# Build T2SP (whenever you change the source code)

```
cd $HOME/t2sp/Halide
make -j
```

# Publications

+ **POPA: Expressing High and Portable Performance across Spatial and Vector Architectures for Tensor Computations**.  
Xiaochen Hao, Hongbo Rong, Mingzhe Zhang, Ce Sun, Hong Jiang, Yun Liang. FPGA, 2024.

+ **Lasa: Abstraction and Specialization for Productive and Performant Linear Algebra on FPGAs**. 
Xiaochen Hao, Mingzhe Zhang, Ce Sun, Zhuofu Tao, Hongbo Rong, Yu Zhang, Lei He, Eric Petit, Wenguang Chen, Yun Liang. FCCM, 2023. [Link](https://ieeexplore.ieee.org/abstract/document/10171577)

+ **SuSy: a programming model for productive construction of high-performance systolic arrays on FPGAs**. 
Yi-Hsiang Lai, Hongbo Rong, Size Zheng, Weihao Zhang, Xiuping Cui, Yunshan Jia, Jie Wang, Brendan Sullivan, Zhiru Zhang, Yun Liang, Youhui Zhang, Jason Cong, Nithin George, Jose Alvarez, Christopher Hughes, and Pradeep Dubey. 2020.  ICCAD 2020. [Link](https://ieeexplore.ieee.org/document/9256583) 

+ **T2S-Tensor: Productively Generating High-Performance Spatial Hardware for Dense Tensor Computations**. 
Nitish Srivastava, Hongbo Rong, Prithayan Barua, Guanyu Feng, Huanqi Cao, Zhiru Zhang, David Albonesi,Vivek Sarkar, Wenguang Chen, Paul Petersen, Geoff Lowney, Adam Herr, Christopher Hughes,Timothy Mattson, Pradeep Dubey. FCCM, 2019. [Link](https://ieeexplore.ieee.org/document/8735529)

