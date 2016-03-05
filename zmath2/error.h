//
//  error.h
//  zmath2
//
//  Created by Alexander zywicki on 1/26/16.
//  Copyright (c) 2016 Alexander zywicki. All rights reserved.
//

#ifndef zmath2_error_h
#define zmath2_error_h
#include "implementation.h"
#include <vector>

namespace zmath {
    template<typename _1,typename _2>
    inline zmath::scalar absolute_error(_1 const& actual, _2 const& approximation, zmath::scalar const& x){
        return abs(actual(x) - approximation(x));
    }
    template<typename _1,typename _2>
    inline std::vector<zmath::scalar> absolute_error(_1 const& actual,
                                              _2 const& approximation,
                                              zmath::range<zmath::scalar> const& r,
                                              unsigned long const& N){
        zmath::scalar x=0;
        zmath::scalar stp = abs(r)/N;
        std::vector<zmath::scalar> result(N);
        for (auto&& indx:result) {
            indx = absolute_error(actual, approximation, x);
            x+=stp;
        }
        return result;
    }
    
    
    template<typename _1,typename _2>
    inline zmath::scalar relative_error(_1 const& actual, _2 const& approximation, zmath::scalar const& x){
        const zmath::scalar a = absolute_error(actual, approximation, x);
        return a * inverse(abs(actual(x)));
    }
    template<typename _1,typename _2>
    inline std::vector<zmath::scalar> relative_error(_1 const& actual,
                                              _2 const& approximation,
                                              zmath::range<zmath::scalar> const& r,
                                              unsigned long const& N){
        zmath::scalar x=0;
        zmath::scalar stp = abs(r)/N;
        std::vector<zmath::scalar> result(N);
        for (auto&& indx:result) {
            indx = relative_error(actual, approximation, x);
            x+=stp;
        }
        return result;
    }
}


#endif
