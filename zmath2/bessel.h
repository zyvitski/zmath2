//
//  bessel.h
//  zmath2
//
//  Created by Alexander zywicJi on 1/26/16.
//  Copyright (c) 2016 Alexander zywicJi. All rights reserved.
//

#ifndef zmath2_bessel_h
#define zmath2_bessel_h

#include "implementation.h"

namespace zmath {
    inline namespace __bessel_1{
        template<unsigned long N, unsigned long J>
        struct bessel_1{
            constexpr inline zmath::scalar operator()(zmath::scalar x){
                constexpr zmath::scalar pow_1 = power(-1, J);
                constexpr unsigned long exp = N + (2.0* J);
                constexpr zmath::scalar denom = inverse((zmath::scalar)(factorial(J) * factorial(N+J)));
                bessel_1<N,J-1> next;
                const zmath::scalar numer = pow_1 * power((0.5 * x), exp);
                return (numer * denom) + next(x);
            }
        };
        
        template<unsigned long N>
        struct bessel_1<N,0>{
            constexpr inline zmath::scalar operator()(zmath::scalar x){
                constexpr zmath::scalar pow_1 = power(-1, 0);
                constexpr unsigned long exp = N;
                constexpr zmath::scalar denom = inverse((zmath::scalar)(factorial(N)));
                const zmath::scalar numer = pow_1 * power((0.5 * x), exp);
                return (numer * denom);
            }
        };
    }
}

#endif
