//
//  chebyshev.h
//
//
//  Created by Alexander zywicki on 1/24/16.
//
//

#ifndef _chebyshev_h
#define _chebyshev_h
namespace zmath{
    inline namespace chebyshev{
        inline namespace __chebyshev_1{
            template<unsigned N>
            constexpr inline zmath::scalar chebyshev_1(zmath::scalar x){
                return (2.0 * x * chebyshev_1<N-1>(x)) - chebyshev_1<N-2>(x);
            }//chebyshev polynomial of the first kind
            
            template<>
            constexpr inline zmath::scalar chebyshev_1<0>(zmath::scalar x){
                return 1;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_1<1>(zmath::scalar x){
                return x;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_1<2>(zmath::scalar x){
                return (2.0 * (x*x))-1.0;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_1<3>(zmath::scalar x){
                const zmath::scalar x_3 = x*x*x;
                return (4.0 * x_3) - (3.0*x);
            }
            template<>
            constexpr inline zmath::scalar chebyshev_1<4>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                return (8.0 * (x_2*x_2)) - (8.0*x_2) + 1.0;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_1<5>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                const zmath::scalar x_3 = x * x_2;
                return (16.0 * (x_3*x_2)) - (20.0 * x_3) +(5.0*x);
            }
            template<>
            constexpr inline zmath::scalar chebyshev_1<6>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar result = -1;
                zmath::scalar power = x_2;
                result += (18.0*power);
                power*=x_2;
                result += (-48.0 * power);
                power*=x_2;
                result += (32.0 * power);
                return result;
            }
            
            template<>
            constexpr inline zmath::scalar chebyshev_1<7>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar power =x;
                zmath::scalar result = -7.0 * power;
                power*=x_2;
                result += (56.0 * power);
                power*=x_2;
                result += (-112.0 * power);
                power*=x_2;
                result+= (64.0 * power);
                return result;
            }
            
            template<>
            constexpr inline zmath::scalar chebyshev_1<8>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar power = x_2;
                zmath::scalar result = 1.0;
                result += (-32.0*power);
                power*=x_2;
                result+=(160.0*power);
                power*=x_2;
                result+=(-256.0*power);
                power*=x_2;
                result+=(128.0*power);
                return result;
            }
            
            
            template<>
            constexpr inline zmath::scalar chebyshev_1<9>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar power = x;
                zmath::scalar result = 9.0 * power;
                power*=x_2;
                result+=(-120.0 * power);
                power*=x_2;
                result+=(432.0*power);
                power*=x_2;
                result+=(-576.0 * power);
                power*=x_2;
                result+=(256.0 * power);
                return result;
            }
            
        }
        
        namespace __chebyshev_2{
            template<unsigned N>
            constexpr inline zmath::scalar chebyshev_2(zmath::scalar x){
                return (2.0 * x * chebyshev_2<N-1>(x)) - chebyshev_2<N-2>(x);
            }//chebyshev polynomial of the second kind
            
            template<>
            constexpr inline zmath::scalar chebyshev_2<0>(zmath::scalar x){
                return 1.0;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<1>(zmath::scalar x){
                return 2.0*x;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<2>(zmath::scalar x){
                return (4.0 * (x*x))-1.0;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<3>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                return (8.0 * (x*x_2)) - (4.0*x);
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<4>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar result = 1.0;
                zmath::scalar power = x_2;
                result += (-12.0 * power);
                power*=x_2;
                result += (16.0 * power);
                return result;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<5>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar power = x;
                zmath::scalar result = 6.0*power;
                power *= x_2;
                result += (-32.0 * power);
                power *= x_2;
                result += (32.0 * power);
                return result;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<6>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar result = -1.0;
                zmath::scalar power = x_2;
                result += (24.0 * power);
                power*=x_2;
                result+=(-80.0 * power);
                power*=x_2;
                result+=(64.0 * power);
                return result;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<7>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar power = x;
                zmath::scalar result = (-8.0 * power);
                power*=x_2;
                result+= (80.0 * power);
                power*=x_2;
                result+= (-192.0 * power);
                power*=x_2;
                result += (128.0 * power);
                return result;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<8>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar result = 1.0;
                zmath::scalar power = x_2;
                result += (-40.0 * power);
                power*=x_2;
                result += (240.0 * power);
                power*=x_2;
                result += (-448.0 * power);
                power*=x_2;
                result+= (256.0 * power);
                return result;
            }
            template<>
            constexpr inline zmath::scalar chebyshev_2<9>(zmath::scalar x){
                const zmath::scalar x_2 = x*x;
                zmath::scalar power = x;
                zmath::scalar result = 10 * power;
                power*=x_2;
                result+= (-160.0 * power);
                power*=x_2;
                result+=(672.0 * power);
                power*=x_2;
                result+=(-1024.0 * power);
                power*=x_2;
                result+= (512.0 * power);
                return result;
            }
        }
    }
}

#endif
