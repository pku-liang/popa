#ifndef HALIDE_COMPLEX_H
#define HALIDE_COMPLEX_H
#include "Util.h"
#include "runtime/HalideRuntime.h"
#include <stdint.h>
#include <string>

namespace Halide {

namespace Internal {
template<typename T, typename CType>
CType data_to_complex(T re, T im) {
    T farray[2] = {re, im};
    CType *p = (CType *)&farray[0];
    CType bits = *p;
    return bits;
}

template<typename T, typename CType>
T re_part(CType data) {
    T farray[2];
    CType *p = (CType *)&farray[0];
    *p = data;
    return farray[0];
}

template<typename T, typename CType>
T im_part(CType data) {
    T farray[2];
    CType *p = (CType *)&farray[0];
    *p = data;
    return farray[1];
}
}  // namespace Internal

using namespace Internal;

template<typename T, typename CType>
struct complex_t {

    /// \name Constructors
    /// @{

    /** Construct from a float or double using
     * round-to-nearest-ties-to-even. Out-of-range values become +/-
     * infinity.
     */
    // @{
    explicit complex_t<T, CType>(T re, T im) : data(data_to_complex<T, CType>(re, im)) {
    }

    // @}

    /** Construct a complex_t with the bits initialised to 0. This represents
     * positive zero.*/
    complex_t<T, CType>() = default;

    /// @}

    /** Return a new complex_t after conjugation*/
    complex_t<T, CType> conj() {
        return complex_t<T, CType>(re_part<T, CType>(data), -im_part<T, CType>(data));
    }

    /** Return a new complex_t with a negated sign bit*/
    complex_t<T, CType> operator-() const {
        return complex_t<T, CType>(-re_part<T, CType>(data), -im_part<T, CType>(data));
    }

    /** Arithmetic operators. */
    // @{
    complex_t<T, CType> operator+(complex_t<T, CType> rhs) const {
        return complex_t<T, CType>(re_part<T, CType>(data) + re_part<T, CType>(rhs.data), im_part<T, CType>(data) + im_part<T, CType>(rhs.data));
    }
    complex_t<T, CType> operator-(complex_t<T, CType> rhs) const {
        return complex_t<T, CType>(re_part<T, CType>(data) - re_part<T, CType>(rhs.data), im_part<T, CType>(data) - im_part<T, CType>(rhs.data));
    }
    complex_t<T, CType> operator*(complex_t<T, CType> rhs) const {
        return complex_t<T, CType>(re_part<T, CType>(data) * re_part<T, CType>(rhs.data) - im_part<T, CType>(data) * im_part<T, CType>(rhs.data),
                           re_part<T, CType>(data) * im_part<T, CType>(rhs.data) + im_part<T, CType>(data) * re_part<T, CType>(rhs.data));
    }
    complex_t<T, CType> operator/(complex_t<T, CType> rhs) const {
        T abs_square = re_part<T, CType>(rhs.data) * re_part<T, CType>(rhs.data) + im_part<T, CType>(rhs.data) * im_part<T, CType>(rhs.data);
        return complex_t<T, CType>((re_part<T, CType>(data) * re_part<T, CType>(rhs.data) + im_part<T, CType>(data) * im_part<T, CType>(rhs.data)) / abs_square,
                           (im_part<T, CType>(data) * re_part<T, CType>(rhs.data) - re_part<T, CType>(data) * im_part<T, CType>(rhs.data)) / abs_square);
    }

    complex_t<T, CType> operator+=(complex_t<T, CType> rhs) {
        return (*this = *this + rhs);
    }
    complex_t<T, CType> operator-=(complex_t<T, CType> rhs) {
        return (*this = *this - rhs);
    }
    complex_t<T, CType> operator*=(complex_t<T, CType> rhs) {
        return (*this = *this * rhs);
    }
    complex_t<T, CType> operator/=(complex_t<T, CType> rhs) {
        return (*this = *this / rhs);
    }
    // @}

    T re() const {
        return re_part<T, CType>(data);
    }
    T im() const {
        return im_part<T, CType>(data);
    }

    /** Comparison operators */
    // @{
    bool operator==(complex_t<T, CType> rhs) const {
        return re_part<T, CType>(data) == re_part<T, CType>(rhs.data) && im_part<T, CType>(data) == im_part<T, CType>(rhs.data);
    }
    bool operator!=(complex_t<T, CType> rhs) const {
        return !(*this == rhs);
    }
    // @}

    /** Returns the bits that represent this complex.
     *
     *  An alternative method to access the bits is to cast a pointer
     *  to this instance as a pointer to a CType.
     **/
    CType to_bits() const {
        return data;
    }

private:
    // The raw bits.
    CType data = 0;
};

typedef complex_t<float, uint64_t> complex32_t;
typedef complex_t<double, __uint128_t> complex64_t;
static_assert(sizeof(complex32_t) == 8, "complex32_t should occupy eight bytes");
static_assert(sizeof(complex64_t) == 16, "complex64_t should occupy sixteen bytes");

}  // namespace Halide

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<Halide::complex32_t>() {
    return halide_type_t(halide_type_complex, 64, 1);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<Halide::complex64_t>() {
    return halide_type_t(halide_type_complex, 128, 1);
}

#endif
