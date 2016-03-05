//
//  implementation.h
//
//
//  Created by Alexander zywicki on 1/23/16.
//
//

#ifndef _implementation_h
#define _implementation_h

#include <type_traits>
#include "types.h"
#ifdef __unix__
typedef unsigned long uintbig_t __attribute__((mode(TI)));
#elif __APPLE__
typedef unsigned long uintbig_t __attribute__((mode(TI)));

#else
typedef unsigned long uintbig_t;

#endif

namespace zmath {
    
    template<typename T> constexpr T _pi = 3.14159265358979323846264338327;
    template<typename T> constexpr T _two_pi = 6.28318530717958647692528676656;
    template<typename T> constexpr T _half_pi = _pi<zmath::scalar> * 0.5;
    constexpr zmath::scalar pi{
        _pi<zmath::scalar>
    };
    constexpr zmath::scalar two_pi{
        _two_pi<zmath::scalar>
    };
    constexpr zmath::scalar half_pi{
        _half_pi<zmath::scalar>
    };
    
    
    template<typename T>
    struct range{
        constexpr explicit range(T _lower,T _upper):lower(_lower),upper(_upper){}
        const T lower;
        const T upper;
    };
    
    
    template <typename T>
    constexpr T abs(T const& x) {
        static_assert(std::is_arithmetic<T>::value,"type provided is not an arithmetic type!");
        return x < 0 ? -1 * x : x;
    }
    template<typename T>
    constexpr T abs(range<T> const& r){
        return zmath::abs(r.upper - r.lower);
    }
    template<typename T>
    constexpr T enforce(T const& value,range<T> const& r){
        if (value > r.upper) {
            return r.upper;
        }else if(value < r.lower){
            return r.lower;
        }else{
            return value;
        }
    }
    
    template<typename T>
    constexpr T signum(T const& x){
        return (((static_cast<long>(x) > 0) & 1) *2)-1;
    }
    template<unsigned long N>
    zmath::vector<N> signum(zmath::vector<N> const& x){
        zmath::vector<N> res;
        for (unsigned long i=0; i<x.length(); ++i) {
            res[i] = signum(x[i]);
        }
        return res;
    }
    template<typename T>
    constexpr T inverse(T const& x){
        static_assert(std::is_floating_point<T>::value,"type provided is not floating point!");
        return x == 0.0 ? 0.0 : 1.0 / x;
    }
    
    constexpr inline uintbig_t factorial (uintbig_t n);
    constexpr inline unsigned long max_factorial(){
        uintbig_t N=1;
        while (factorial(N)<factorial(N+1)) {
            ++N;
        }
        return N;
    }
    
    constexpr inline uintbig_t factorial (uintbig_t n){
        return n > 0 ? n * factorial( n - 1 ) : 1;
    }
    template<typename T>
    constexpr inline zmath::scalar power(T x,unsigned long n){
        T result(1);
        while (n != 0) {
            if (!((n%2)==0)) {
                //odd
                result *= x;
                n-=1;
            }
            x*=x;
            n/=2;
        }
        return result;
    }
    
    constexpr inline zmath::scalar constrain(zmath::scalar const& value, zmath::scalar const& minimum, zmath::scalar const& maximum){
        const zmath::scalar d = signum(value) * (maximum - minimum);
        return (value - (static_cast<long>(value / d) * d));
    }
    constexpr inline zmath::scalar constrain(zmath::scalar const& value,zmath::range<zmath::scalar> const& r){
        const zmath::scalar d = signum(value) * zmath::abs(r);
        return (value - (static_cast<long>(value / d) * d));
    }

    template<unsigned long N>
    constexpr inline zmath::vector<N> constrain(zmath::vector<N> const& value, zmath::vector<N> const& minimum, zmath::vector<N> const& maximum){
        const zmath::vector<N> d = signum(value) * (maximum - minimum);
        return (value - (zmath::floor(value / d) * d));
    }
    
    
    
    
    
}
#endif
