//
//  sine.h
//  vec
//
//  Created by Alexander zywicki on 2/7/16.
//  Copyright (c) 2016 Alexander zywicki. All rights reserved.
//

#ifndef vec_sine_h
#define vec_sine_h
#include "types.h"
#include "implementation.h"
namespace zmath{
    template<unsigned long N>
    inline zmath::vector<N> sine(zmath::vector<N> const& x){
        zmath::vector<N> result(x);
        result+=zmath::half_pi;
        result = zmath::constrain(result, zmath::vector<N>(-zmath::pi), zmath::vector<N>(zmath::pi));
        result = zmath::abs(result);
        result-=zmath::half_pi;
        
        constexpr zmath::scalar _shaper3 = -inverse((zmath::scalar)factorial(3));
        constexpr zmath::scalar _shaper5 =  inverse((zmath::scalar)factorial(5));
        constexpr zmath::scalar _shaper7 = -inverse((zmath::scalar)factorial(7));
        constexpr zmath::scalar _shaper9 =  inverse((zmath::scalar)factorial(9));
        constexpr zmath::scalar _shaper11 =  -inverse((zmath::scalar)factorial(11));
        constexpr zmath::scalar _shaper13 =  inverse((zmath::scalar)factorial(13));
        constexpr zmath::scalar _shaper15 =  -inverse((zmath::scalar)factorial(15));
        constexpr zmath::scalar _shaper17 =  inverse((zmath::scalar)factorial(17));
        
        zmath::vector<N> power(result);
        zmath::vector<N> x_2 (power*power);
        power*=x_2;
        result += (_shaper3 * power);
        power*=x_2;
        result += (_shaper5 * power);
        power*=x_2;
        result += (_shaper7 * power);
        power*=x_2;
        result += (_shaper9 * power);
        power*=x_2;
        result += (_shaper11 * power);
        power*=x_2;
        result += (_shaper13 * power);
        power*=x_2;
        result += (_shaper15 * power);
        power*=x_2;
        result += (_shaper17 * power);
        
        return result;
    }
    constexpr inline zmath::scalar sine(zmath::scalar x){
        constexpr zmath::range<zmath::scalar> domain{-pi,pi};
        x+=half_pi;
        x = constrain(x, domain);
        x = abs(x);
        x-=half_pi;
        zmath::scalar result = x;
        {
            constexpr zmath::scalar _shaper3 = -inverse((zmath::scalar)factorial(3));
            constexpr zmath::scalar _shaper5 =  inverse((zmath::scalar)factorial(5));
            constexpr zmath::scalar _shaper7 = -inverse((zmath::scalar)factorial(7));
            constexpr zmath::scalar _shaper9 =  inverse((zmath::scalar)factorial(9));
            constexpr zmath::scalar _shaper11 =  -inverse((zmath::scalar)factorial(11));
            constexpr zmath::scalar _shaper13 =  inverse((zmath::scalar)factorial(13));
            constexpr zmath::scalar _shaper15 =  -inverse((zmath::scalar)factorial(15));
            constexpr zmath::scalar _shaper17 =  inverse((zmath::scalar)factorial(17));
            
            zmath::scalar power_1 = (x*x)*x;
            const zmath::scalar x_4 = power_1*x;
            zmath::scalar power_2 = x_4*x;
            
            result += (_shaper3 * power_1);
            result += (_shaper5 * power_2);
            power_1*=x_4;
            power_2*=x_4;
            result += (_shaper7 * power_1);
            result += (_shaper9 * power_2);
            power_1*=x_4;
            power_2*=x_4;
            result += (_shaper11 * power_1);
            result += (_shaper13 * power_2);
            power_1*=x_4;
            power_2*=x_4;
            result += (_shaper15 * power_1);
            result += (_shaper17 * power_2);
        }
        return result;
    }
#ifdef __SSE4_1__
    namespace simd{
        inline zmath::scalar sine(zmath::scalar x){
            using __a =  zmath::vector_attributes;
            constexpr zmath::scalar _shapers[4][2] ={
                {-inverse((zmath::scalar)factorial(3)),inverse((zmath::scalar)factorial(5))},
                {-inverse((zmath::scalar)factorial(7)),inverse((zmath::scalar)factorial(9))},
                {-inverse((zmath::scalar)factorial(11)),inverse((zmath::scalar)factorial(13))},
                {-inverse((zmath::scalar)factorial(15)),inverse((zmath::scalar)factorial(17))}
            };
            
            x+=half_pi;
            x = zmath::constrain(x, -pi, pi);
            x = zmath::abs(x);
            x-=half_pi;
            
            __a::__vector_type X_4,power;
            __a::__vector_type temp = __a::set(0);
            {
                const zmath::scalar x_2 = x*x;
                const zmath::scalar x_4 = x_2*x_2;
                X_4 = __a::set(x_4);
                power = __a::set(x_2*x, x_4*x);
            }
            temp = __a::add(temp,(__a::multiply(__a::loadr(_shapers[0]),power)));
            power = __a::multiply(power, X_4);
            temp = __a::add(temp,(__a::multiply(__a::loadr(_shapers[1]),power)));
            power = __a::multiply(power, X_4);
            temp = __a::add(temp,(__a::multiply(__a::loadr(_shapers[2]),power)));
            power = __a::multiply(power, X_4);
            temp = __a::add(temp,(__a::multiply(__a::loadr(_shapers[3]),power)));
            
            
            zmath::scalar result[2];
            __a::store(__a::hadd(temp, temp),result);
            return x + result[0];
        }
        namespace avx{
            inline double sine(double x){
                SSE_ALIGN( double result[4]);
                x+=half_pi;
                x = zmath::constrain(x, -pi, pi);
                x = zmath::abs(x);
                x-=half_pi;
                
                constexpr zmath::scalar _shaper3 = -inverse((long double)factorial(3));
                constexpr zmath::scalar _shaper5 =  inverse((long double)factorial(5));
                constexpr zmath::scalar _shaper7 = -inverse((long double)factorial(7));
                constexpr zmath::scalar _shaper9 =  inverse((long double)factorial(9));
                constexpr zmath::scalar _shaper11 =  -inverse((long double)factorial(11));
                constexpr zmath::scalar _shaper13 =  inverse((long double)factorial(13));
                constexpr zmath::scalar _shaper15 =  -inverse((long double)factorial(15));
                constexpr zmath::scalar _shaper17 =  inverse((long double)factorial(17));
                
                SSE_ALIGN (__m256d X_8),SSE_ALIGN(power);
                SSE_ALIGN (__m256d temp) = _mm256_set1_pd(0);
                {
                    const double x_2 = x*x;
                    SSE_ALIGN( double pwr[4]);
                    pwr[0] = x*x_2;
                    for (unsigned long i=1; i<4; ++i) {
                        pwr[i] = pwr[i-1]*x_2;
                    }
                    X_8 = _mm256_set1_pd(pwr[2]*x);
                    std::swap(pwr[0], pwr[3]);
                    std::swap(pwr[1], pwr[2]);
                    power = _mm256_load_pd(pwr);
                }
                temp = _mm256_add_pd(temp, (_mm256_mul_pd(_mm256_set_pd(_shaper3, _shaper5, _shaper7, _shaper9), power)));
                power = _mm256_mul_pd(power, X_8);
                temp = _mm256_add_pd(temp, (_mm256_mul_pd(_mm256_set_pd(_shaper11, _shaper13, _shaper15, _shaper17), power)));
                
                _mm256_store_pd(result, temp);
                return x+result[0]+result[1]+result[2]+result[3];
                
                
            }
        }
        namespace avx512{
            inline double sine(double x){
                SSE_ALIGN( double result[8]);
                x+=half_pi;
                x = zmath::constrain(x, -pi, pi);
                x = zmath::abs(x);
                x-=half_pi;
                
                constexpr double _shaper[8] ={
                    -inverse((long double)factorial(3)),
                    inverse((long double)factorial(5)),
                    -inverse((long double)factorial(7)),
                    inverse((long double)factorial(9)),
                    -inverse((long double)factorial(11)),
                    inverse((long double)factorial(13)),
                    -inverse((long double)factorial(15)),
                    inverse((long double)factorial(17))
                };
                
                SSE_ALIGN( __m512d power);
                SSE_ALIGN( __m512d temp) = _mm512_set1_pd(0);
                {
                    SSE_ALIGN( double pwr[8]);
                    const double x_2 = x*x;
                    
                    pwr[0] = x*x_2;
                    
                    for (unsigned long i=1; i<8; ++i) {
                        pwr[i] = pwr[i-1]*x_2;
                    }
                    power = _mm512_loadu_pd(pwr);
                }
                {
                    temp = _mm512_add_pd(temp, _mm512_mul_pd(_mm512_loadu_pd(_shaper), power));
                    
                    _mm512_store_pd(result, temp);
                }
                double r=x;
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
