//
//  legendere.h
//  zmath2
//
//  Created by Alexander zywicki on 1/26/16.
//  Copyright (c) 2016 Alexander zywicki. All rights reserved.
//

#ifndef zmath2_legendere_h
#define zmath2_legendere_h

#include "implementation.h"

namespace zmath{
    inline namespace __legendere{
        template<unsigned N>
        constexpr inline zmath::scalar legendere(zmath::scalar x){
            constexpr zmath::scalar _2n1 = (2*N)+1;
            constexpr zmath::scalar _in1 = inverse((zmath::scalar)(N+1));
            return ((_2n1 * x * legendere<N-1>(x)) - (N*legendere<N-2>(x))) * _in1;
            
        }
        template<>
        constexpr inline zmath::scalar legendere<0>(zmath::scalar x){
            return 1.0;
        }
        template<>
        constexpr inline zmath::scalar legendere<1>(zmath::scalar x){
            return x;
        }
        
    }
}

#endif
