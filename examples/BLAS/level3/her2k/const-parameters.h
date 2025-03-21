#pragma once

#ifdef TINY // For verifying correctness only
    #define KKK         4
    #define JJJ         4
    #define III         4
    #define JJ          4
    #define II          4
    #define KK          4
#elif S10
    #define KKK         16
    #define JJJ         8
    #define III         8
    #define JJ          16
    #define II          16
    #define KK          8
#else   // For A10
    #define KKK         8
    #define JJJ         4
    #define III         4
    #define JJ          32
    #define II          32
    #define KK          16
#endif
