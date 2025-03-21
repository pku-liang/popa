#pragma once

// Inner loop bounds, which are static constant parameters of the design
#ifdef GPU
    #define KKK         8
    #define JJJ         8
    #define III         32
    #define JJ          8
    #define II          2
    #define KK          1
#else // FPGA
    #ifdef TINY // For verifying correctness only
        #define KKK         4
        #define JJJ         4
        #define III         4
        #define JJ          4
        #define II          4
        #define KK          4
    #elif S10
        #define KKK         16
        #define JJJ         16
        #define III         14
        #define JJ          32
        #define II          32
        #define KK          32
    #else   // For A10
        #define KKK         8
        #define JJJ         4
        #define III         10
        #define JJ          32
        #define II          32
        #define KK          32
    #endif
#endif
