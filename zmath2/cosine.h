//
//  cosine.h
//  zmath2
//
//  Created by Alexander zywicki on 1/29/16.
//  Copyright (c) 2016 Alexander zywicki. All rights reserved.
//

#ifndef zmath2_cosine_h
#define zmath2_cosine_h
#include "implementation.h"

#ifdef __SSE4_1__
#include <immintrin.h>

#endif

#include "types.h"

namespace zmath {
    
    template<unsigned long N>
    inline zmath::vector<N> cosine(zmath::vector<N> const& x){
        zmath::vector<N> _x(x);
        
        
        _x = zmath::constrain(_x, zmath::vector<N>(-zmath::pi), zmath::vector<N>(zmath::pi));
        _x = zmath::abs(_x);
        
        constexpr zmath::scalar _shaper2 = -inverse((zmath::scalar)factorial(2));
        constexpr zmath::scalar _shaper4 =  inverse((zmath::scalar)factorial(4));
        constexpr zmath::scalar _shaper6 = -inverse((zmath::scalar)factorial(6));
        constexpr zmath::scalar _shaper8 =  inverse((zmath::scalar)factorial(8));
        constexpr zmath::scalar _shaper10 =  -inverse((zmath::scalar)factorial(10));
        constexpr zmath::scalar _shaper12 =  inverse((zmath::scalar)factorial(12));
        constexpr zmath::scalar _shaper14 =  -inverse((zmath::scalar)factorial(14));
        constexpr zmath::scalar _shaper16 =  inverse((zmath::scalar)factorial(16));
        
        
        zmath::vector<N> x_2 (_x*_x);
        zmath::vector<N> power(x_2);
        zmath::vector<N> result(1.0);
        result += (_shaper2 * power);
        power*=x_2;
        result += (_shaper4 * power);
        power*=x_2;
        result += (_shaper6 * power);
        power*=x_2;
        result += (_shaper8 * power);
        power*=x_2;
        result += (_shaper10 * power);
        power*=x_2;
        result += (_shaper12 * power);
        power*=x_2;
        result += (_shaper14 * power);
        power*=x_2;
        result += (_shaper16 * power);
        
        return result;
    }
    
    
    constexpr zmath::scalar cosine(zmath::scalar x){
        
        x = constrain(x, -pi, pi);
        x = abs(x);
        zmath::scalar result = 1;
        {
            constexpr zmath::scalar _shaper2 = -inverse((zmath::scalar)factorial(2));
            constexpr zmath::scalar _shaper4 =  inverse((zmath::scalar)factorial(4));
            constexpr zmath::scalar _shaper6 = -inverse((zmath::scalar)factorial(6));
            constexpr zmath::scalar _shaper8 =  inverse((zmath::scalar)factorial(8));
            constexpr zmath::scalar _shaper10 =  -inverse((zmath::scalar)factorial(10));
            constexpr zmath::scalar _shaper12 =  inverse((zmath::scalar)factorial(12));
            constexpr zmath::scalar _shaper14 =  -inverse((zmath::scalar)factorial(14));
            constexpr zmath::scalar _shaper16 =  inverse((zmath::scalar)factorial(16));
            
            zmath::scalar power_1 = (x*x);
            const zmath::scalar x_4 = power_1*power_1;
            zmath::scalar power_2 = x_4;
            
            result += (_shaper2 * power_1);
            result +=(_shaper4 * power_2);
            power_1*=x_4;
            power_2*=x_4;
            result += (_shaper6 * power_1);
            result +=(_shaper8 * power_2);
            power_1*=x_4;
            power_2*=x_4;
            result += (_shaper10 * power_1);
            result +=(_shaper12 * power_2);
            power_1*=x_4;
            power_2*=x_4;
            result += (_shaper14 * power_1);
            result +=(_shaper16 * power_2);
        }
        return result;
    }
#ifdef __SSE4_1__
    namespace simd{
        zmath::scalar cosine(zmath::scalar x){
            using __a =  zmath::vector_attributes;
            
            
            zmath::scalar result[2];
            x = constrain(x, -pi, pi);
            x = abs(x);
            
            constexpr zmath::scalar _shaper2 = -inverse((zmath::scalar)factorial(2));
            constexpr zmath::scalar _shaper4 =  inverse((zmath::scalar)factorial(4));
            constexpr zmath::scalar _shaper6 = -inverse((zmath::scalar)factorial(6));
            constexpr zmath::scalar _shaper8 =  inverse((zmath::scalar)factorial(8));
            constexpr zmath::scalar _shaper10 =  -inverse((zmath::scalar)factorial(10));
            constexpr zmath::scalar _shaper12 =  inverse((zmath::scalar)factorial(12));
            constexpr zmath::scalar _shaper14 =  -inverse((zmath::scalar)factorial(14));
            constexpr zmath::scalar _shaper16 =  inverse((zmath::scalar)factorial(16));
            
            __a::__vector_type X_4,power;
            __a::__vector_type temp = __a::set(0);
            {
                const zmath::scalar x_2 = x*x;
                X_4 = __a::set(x_2*x_2);
                power = __a::set(x_2, x_2*x_2);
            }
            
            
            temp = __a::add(temp,  (__a::multiply(__a::set(_shaper2, _shaper4),power)));
            power = __a::multiply(power, X_4);
            temp = __a::add(temp,  (__a::multiply(__a::set(_shaper6, _shaper8),power)));
            power = __a::multiply(power, X_4);
            temp = __a::add(temp,  (__a::multiply(__a::set(_shaper10, _shaper12),power)));
            power = __a::multiply(power, X_4);
            temp = __a::add(temp,  (__a::multiply(__a::set(_shaper14, _shaper16),power)));
            
            __a::store(__a::hadd(temp, temp),result);
            return 1 + result[0];
        }
        
        namespace avx512{
            double cosine(double x){
                SSE_ALIGN( double result[8]);
                x = zmath::constrain(x, -pi, pi);
                x = zmath::abs(x);
                
                constexpr double _shaper[8] ={
                    -inverse((long double)factorial(2)),
                    inverse((long double)factorial(4)),
                    -inverse((long double)factorial(6)),
                    inverse((long double)factorial(8)),
                    -inverse((long double)factorial(10)),
                    inverse((long double)factorial(12)),
                    -inverse((long double)factorial(14)),
                    inverse((long double)factorial(16))
                };
                
                SSE_ALIGN( __m512d power);
                SSE_ALIGN( __m512d temp )= _mm512_set1_pd(0);
                {
                    SSE_ALIGN (double pwr[8]);
                    const double x_2 = x*x;
                    pwr[0] = x_2;
                    for (unsigned long i=1; i<8; ++i) {
                        pwr[i] = pwr[i-1]*x_2;
                    }
                    power = _mm512_loadu_pd(pwr);
                }
                {
                    SSE_ALIGN( __m512d shaper)  = _mm512_loadu_pd(_shaper);
                    
                    temp = _mm512_add_pd(temp, (_mm512_mul_pd(shaper, power)));
                    
                    _mm512_store_pd(result, temp);
                }
                double r=1;
                for (unsigned long i=0; i<8; ++i) {
                    r+=result[i];
                }
                return r;
            }
        }

    }
#endif
    
    
}

#endif
