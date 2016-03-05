//
//  vector_types.h
//  vectorized
//
//  Created by Alexander zywicki on 2/9/16.
//  Copyright (c) 2016 Alexander zywicki. All rights reserved.
//

#ifndef vectorized_vector_types_h
#define vectorized_vector_types_h

#include <immintrin.h>
#include <smmintrin.h>
#include <avxintrin.h>
#include <avx512fintrin.h>
#include <ostream>

#ifdef __unix__
#define SSE_ALIGN __attribute__((aligned (16)))
#elif __APPLE__
#define SSE_ALIGN __attribute__((aligned (16)))
#elif _WIN32
#define SSE_ALIGN __declspec( align(16) )
#endif

namespace detail {

    using __128d = __m128d SSE_ALIGN;
    using __128f  = __m128  SSE_ALIGN;
    
    template<typename T>
    struct vector_traits;
    
    template<>
    struct vector_traits<__128d>{
        typedef __128d vector_type;
        typedef double scalar_type;
        typedef unsigned long size_type;
        static constexpr size_type length = 2;
        
        static inline vector_type add(vector_type const& lhs,vector_type const& rhs){
            return _mm_add_pd(lhs,rhs);
        }
        static inline vector_type subtract(vector_type const& lhs,vector_type const& rhs){
            return _mm_sub_pd(lhs,rhs);
        }
        static inline vector_type multiply(vector_type const& lhs,vector_type const& rhs){
            return _mm_mul_pd(lhs,rhs);
        }
        static inline vector_type divide(vector_type const& lhs,vector_type const& rhs){
            return _mm_div_pd(lhs,rhs);
        }
        static inline vector_type set(scalar_type const& value){
            return _mm_set1_pd(value);
        }
        static inline vector_type set_zero(){
            return _mm_setzero_pd();
        }
        static inline vector_type set(scalar_type const& _1, scalar_type const& _2){
            return _mm_set_pd(_1, _2);
        }
        static inline vector_type load( scalar_type const (&_1)[length]){
            return _mm_load_pd(_1);
        }
        static inline vector_type loadr( scalar_type const (&_1)[length]){
            return _mm_loadr_pd(_1);
        }
        static inline vector_type is_equal(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_EQ_OQ), _mm_set1_pd(static_cast<size_type>(true)));
        }
        static inline vector_type not_equal(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_NEQ_OQ),_mm_set1_pd(static_cast<size_type>(true)));
        }
        static inline vector_type greater_than(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_GT_OQ),_mm_set1_pd(static_cast<size_type>(true)));
        }
        static inline vector_type less_than(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_LT_OQ),_mm_set1_pd(static_cast<size_type>(true)));
        }
        static inline vector_type less_than_equal_to(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_LE_OQ),_mm_set1_pd(static_cast<size_type>(true)));
        }
        static inline vector_type greater_than_equal_to(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_GE_OQ),_mm_set1_pd(static_cast<size_type>(true)));
        }
        static inline vector_type bitwise_and(vector_type const& lhs, vector_type const& rhs){
            return _mm_and_pd(lhs,rhs);
        }
        static inline vector_type bitwise_or(vector_type const& lhs, vector_type const& rhs){
            return _mm_or_pd(lhs, rhs);
        }
        static inline vector_type bitwise_xor(vector_type const& lhs, vector_type const& rhs){
            return _mm_xor_pd(lhs, rhs);
        }
        static inline vector_type bitwise_not( vector_type const& rhs){
            return _mm_xor_pd(rhs,_mm_castsi128_pd(_mm_set1_epi32(-1)));
        }
        static inline vector_type floor(vector_type const& _1){
            return _mm_floor_pd(_1);
        }
        static inline vector_type ceil(vector_type const& _1){
            return _mm_ceil_pd(_1);
        }
        static inline vector_type hadd(vector_type const& _1,vector_type const& _2){
            return _mm_hadd_pd(_1, _2);
        }
        static inline void store(vector_type const& _1,scalar_type(&_2)[length]){
            _mm_store_pd(_2, _1);
        }
        static inline vector_type abs(vector_type const& value){
            static const vector_type sign_mask = _mm_set1_pd(-0.); // -0. = 1 << 63
            return _mm_andnot_pd(sign_mask, value); // !sign_mask & x
        }
    };
    template<>
    struct vector_traits<__128f>{
        typedef __128f vector_type;
        typedef float scalar_type;
        typedef unsigned long size_type;
        static constexpr size_type length = 4;
        
        static inline vector_type add(vector_type const& lhs,vector_type const& rhs){
            return _mm_add_ps(lhs,rhs);
        }
        static inline vector_type subtract(vector_type const& lhs,vector_type const& rhs){
            return _mm_sub_ps(lhs,rhs);
        }
        static inline vector_type multiply(vector_type const& lhs,vector_type const& rhs){
            return _mm_mul_ps(lhs,rhs);
        }
        static inline vector_type divide(vector_type const& lhs,vector_type const& rhs){
            return _mm_div_ps(lhs,rhs);
        }
        static inline vector_type set(scalar_type const& value){
            return _mm_set1_ps(value);
        }
        static inline vector_type set_zero(){
            return _mm_setzero_ps();
        }
        static inline vector_type set(scalar_type const& _1,
                                      scalar_type const& _2,
                                      scalar_type const& _3,
                                      scalar_type const& _4){
            return _mm_set_ps(_1, _2,_3,_4);
        }
        static inline vector_type load( scalar_type const (&_1)[length]){
            return _mm_load_ps(_1);
        }
        static inline vector_type loadr( scalar_type const (&_1)[length]){
            return _mm_loadr_ps(_1);
        }
        static inline vector_type is_equal(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_ps(_mm_cmp_ps(lhs, rhs,_CMP_EQ_OQ), _mm_set1_ps(static_cast<size_type>(true)));
        }
        static inline vector_type not_equal(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_ps(_mm_cmp_ps(lhs, rhs,_CMP_NEQ_OQ),_mm_set1_ps(static_cast<size_type>(true)));
        }
        static inline vector_type greater_than(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_ps(_mm_cmp_ps(lhs, rhs,_CMP_GT_OQ),_mm_set1_ps(static_cast<size_type>(true)));
        }
        static inline vector_type less_than(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_ps(_mm_cmp_ps(lhs, rhs,_CMP_LT_OQ),_mm_set1_ps(static_cast<size_type>(true)));
        }
        static inline vector_type less_than_equal_to(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_ps(_mm_cmp_ps(lhs, rhs,_CMP_LE_OQ),_mm_set1_ps(static_cast<size_type>(true)));
        }
        static inline vector_type greater_than_equal_to(vector_type const& lhs, vector_type const&  rhs){
            return _mm_and_ps(_mm_cmp_ps(lhs, rhs,_CMP_GE_OQ),_mm_set1_ps(static_cast<size_type>(true)));
        }
        static inline vector_type bitwise_and(vector_type const& lhs, vector_type const& rhs){
            return _mm_and_ps(lhs,rhs);
        }
        static inline vector_type bitwise_or(vector_type const& lhs, vector_type const& rhs){
            return _mm_or_ps(lhs, rhs);
        }
        static inline vector_type bitwise_xor(vector_type const& lhs, vector_type const& rhs){
            return _mm_xor_ps(lhs, rhs);
        }
        static inline vector_type bitwise_not( vector_type const& rhs){
            return _mm_xor_ps(rhs,_mm_castsi128_ps(_mm_set1_epi32(-1)));
        }
        static inline vector_type floor(vector_type const& _1){
            return _mm_floor_ps(_1);
        }
        static inline vector_type ceil(vector_type const& _1){
            return _mm_ceil_ps(_1);
        }
        static inline vector_type hadd(vector_type const& _1,vector_type const& _2){
            return _mm_hadd_ps(_1, _2);
        }
        static inline void store(vector_type const& _1,scalar_type(&_2)[length]){
            _mm_store_ps(_2, _1);
        }
        static inline vector_type abs(vector_type const& value){
            static const vector_type sign_mask = _mm_set1_ps(-0.); // -0. = 1 << 63
            return _mm_andnot_ps(sign_mask, value); // !sign_mask & x
        }
    };
    
    template<typename T>
    struct __vector{
        using traits = vector_traits<T>;
        using self_type = __vector<T>;
        using vector_type = typename traits::vector_type;
        using scalar_type = typename traits::scalar_type;
        using size_type   = typename traits::size_type;
        constexpr size_type length()const{return traits::length;}
        vector_type __data;
        
        explicit inline __vector():__data(traits::set_zero()){}
        explicit inline __vector(scalar_type const& _1):__data(traits::set(_1)){}
        explicit inline __vector(const scalar_type (& _1)[traits::length]):__data(traits::load(_1)){}
        explicit inline __vector(vector_type const& _1):__data(_1){}
        template<typename...args_t>
        inline __vector(args_t... args):__data(traits::set(args...)){}
        
        
        //access
        scalar_type& operator[](size_type const& index){
            return reinterpret_cast<scalar_type(&)[traits::length]>(__data)[index];
        }
        scalar_type const& operator[](size_type const& index)const {
            return reinterpret_cast<const scalar_type(&)[traits::length]>(__data)[index];
        }
        
        
        //add
        friend self_type operator+(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::add(_1.__data,_2.__data));
        }
        friend self_type operator+(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::add(traits::set(_1),_2.__data));
        }
        friend self_type operator+(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::add(_1.__data,traits::set(_2)));
        }
        self_type& operator+=(self_type const& other){
            __data = __data + other.__data;
            return *this;
        }
        self_type& operator+=(scalar_type const& other){
            __data = __data + traits::set(other);
            return *this;
        }
        
        //sub
        friend self_type operator-(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::subtract(_1.__data,_2.__data));
        }
        friend self_type operator-(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::subtract(traits::set(_1),_2.__data));
        }
        friend self_type operator-(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::subtract(_1.__data,traits::set(_2)));
        }
        self_type& operator-=(self_type const& other){
            __data = __data - other.__data;
            return *this;
        }
        self_type& operator-=(scalar_type const& other){
            __data = __data - traits::set(other);
            return *this;
        }
        
        //mul
        friend self_type operator*(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::multiply(_1.__data,_2.__data));
        }
        friend self_type operator*(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::multiply(traits::set(_1),_2.__data));
        }
        friend self_type operator*(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::multiply(_1.__data,traits::set(_2)));
        }
        self_type& operator*=(self_type const& other){
            __data = __data * other.__data;
            return *this;
        }
        self_type& operator*=(scalar_type const& other){
            __data = __data * traits::set(other);
            return *this;
        }
        
        //divide
        friend self_type operator/(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::divide(_1.__data,_2.__data));
        }
        friend self_type operator/(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::divide(traits::set(_1),_2.__data));
        }
        friend self_type operator/(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::divide(_1.__data,traits::set(_2)));
        }
        self_type& operator/=(self_type const& other){
            __data = __data / other.__data;
            return *this;
        }
        self_type& operator/=(scalar_type const& other){
            __data = __data / traits::set(other);
            return *this;
        }
        
        
        //and
        friend self_type operator&(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_and(_1.__data,_2.__data));
        }
        friend self_type operator&(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_and(traits::set(_1),_2.__data));
        }
        friend self_type operator&(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::bitwise_and(_1.__data,traits::set(_2)));
        }
        self_type& operator&=(self_type const& other){
            __data = __data & other.__data;
            return *this;
        }
        self_type& operator&=(scalar_type const& other){
            __data = __data & traits::set(other);
            return *this;
        }
        
        //or
        friend self_type operator|(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_or(_1.__data,_2.__data));
        }
        friend self_type operator|(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_or(traits::set(_1),_2.__data));
        }
        friend self_type operator|(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::bitwise_or(_1.__data,traits::set(_2)));
        }
        self_type& operator|=(self_type const& other){
            __data = __data | other.__data;
            return *this;
        }
        self_type& operator|=(scalar_type const& other){
            __data = __data | traits::set(other);
            return *this;
        }
        
        //xor
        friend self_type operator^(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_xor(_1.__data,_2.__data));
        }
        friend self_type operator^(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_xor(traits::set(_1),_2.__data));
        }
        friend self_type operator^(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::bitwise_xor(_1.__data,traits::set(_2)));
        }
        self_type& operator^=(self_type const& other){
            __data = __data ^ other.__data;
            return *this;
        }
        self_type& operator^=(scalar_type const& other){
            __data = __data ^ traits::set(other);
            return *this;
        }
        
        //greater than
        friend self_type operator>(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::greater_than(_1.__data,_2.__data));
        }
        friend self_type operator>(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::greater_than(traits::set(_1),_2.__data));
        }
        friend self_type operator>(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::greater_than(_1.__data,traits::set(_2)));
        }
        
        //greater than equal to
        friend self_type operator>=(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::greater_than_equal_to(_1.__data,_2.__data));
        }
        friend self_type operator>=(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::greater_than_equal_to(traits::set(_1),_2.__data));
        }
        friend self_type operator>=(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::greater_than_equal_to(_1.__data,traits::set(_2)));
        }
        
        //less than
        friend self_type operator<(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::less_than(_1.__data,_2.__data));
        }
        friend self_type operator<(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::less_than(traits::set(_1),_2.__data));
        }
        friend self_type operator<(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::less_than(_1.__data,traits::set(_2)));
        }
        
        //less than equal to
        friend self_type operator<=(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::less_than_equal_to(_1.__data,_2.__data));
        }
        friend self_type operator<=(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::less_than_equal_to(traits::set(_1),_2.__data));
        }
        friend self_type operator<=(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::less_than_equal_to(_1.__data,traits::set(_2)));
        }
        
        //equal to
        friend self_type operator==(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::is_equal(_1.__data,_2.__data));
        }
        friend self_type operator==(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::is_equal(traits::set(_1),_2.__data));
        }
        friend self_type operator==(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::is_equal(_1.__data,traits::set(_2)));
        }
        
        //not equal
        friend self_type operator!=(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::not_equal(_1.__data,_2.__data));
        }
        friend self_type operator!=(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::not_equal(traits::set(_1),_2.__data));
        }
        friend self_type operator!=(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::not_equal(_1.__data,traits::set(_2)));
        }
        
        friend self_type operator&&(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_and(_1.__data,_2.__data));
        }
        friend self_type operator&&(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_and(traits::set(_1),_2.__data));
        }
        friend self_type operator&&(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::bitwise_and(_1.__data,traits::set(_2)));
        }
        friend self_type operator||(self_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_or(_1.__data,_2.__data));
        }
        friend self_type operator||(scalar_type const& _1,self_type const& _2){
            return static_cast<self_type>(traits::bitwise_or(traits::set(_1),_2.__data));
        }
        friend self_type operator||(self_type const& _1,scalar_type const& _2){
            return static_cast<self_type>(traits::bitwise_or(_1.__data,traits::set(_2)));
        }
        
        //increment
        self_type& operator++(){
            (*this)+=1;
            return *this;
        }
        self_type operator++(int){
            self_type temp(*this);
            ++temp;
            return temp;
        }
        //decrement
        self_type& operator--(){
            (*this)-=1;
            return *this;
        }
        self_type operator--(int){
            self_type temp(*this);
            --temp;
            return temp;
        }
        
        //unary
        
        self_type operator+(){
            return static_cast<self_type>(traits::abs(__data));
        }
        self_type operator-(){
            return (*this)*-1;
        }
        self_type operator~(){
            return static_cast<self_type>(traits::bitwise_not(__data));
        }
        self_type operator!(){
            return static_cast<self_type>(traits::bitwise_not(__data));
        }
        
        friend std::ostream& operator<<(std::ostream& os,self_type const& v){
            for(size_type i=0;i<v.length();++i){
                os<<v[(v.length()-1)-i]<<"\t";
            }
            return os;
        }
        
        inline operator vector_type()const{
            return __data;
        }
        
        bool any(bool const& value){
            for(size_type i=0;i<length();++i){
                if((*this)[i]==value){
                    return true;
                }
            }
            return false;
        }
        bool all(bool const& value){
            bool result = true;
            for(size_type i=0;i<length();++i){
                if((*this)[i]!=value){
                    result = false;
                }
            }
            return result;
        }
        bool none(bool const& value){
            for(size_type i=0;i<length();++i){
                if((*this)[i]==value){
                    return false;
                }
            }
            return true;
        }
        

    };
}

typedef detail::__vector<detail::__128d> vector_2d;
typedef detail::__vector<detail::__128f>  vector_4f;

template<typename T>
detail::__vector<T> hadd(detail::__vector<T> const& _1,detail::__vector<T> const& _2){
    using traits = detail::vector_traits<T>;
    return static_cast<detail::__vector<T>>(traits::hadd(_1.__data,_2.__data));
}




#endif
