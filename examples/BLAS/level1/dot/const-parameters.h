#pragma once

#ifdef TINY // For verifying correctness only
    #define KKK         4
    #define KK          4
    #ifdef T2SP_DDOT
        #define TTYPE Float(64)
    #elif defined(T2SP_CDOT)
        #define TTYPE Complex(32)
    #elif defined(T2SP_ZDOT)
        #define TTYPE Complex(64)
    #else
        #define TTYPE Float(32)
    #endif
#else
    #define KK 64
    #ifdef T2SP_DDOT
        #define KKK 8
        #define TTYPE Float(64)
    #elif defined(T2SP_CDOT)
        #define KKK 8
        #define TTYPE Complex(32)
    #elif defined(T2SP_ZDOT)
        #define KKK 4
        #define TTYPE Complex(64)
    #else
        #define KKK 16
        #define TTYPE Float(32)
    #endif
#endif
