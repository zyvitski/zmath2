//
//  types.h
//  vec
//
//  Created by Alexander zywicki on 2/5/16.
//  Copyright (c) 2016 Alexander zywicki. All rights reserved.
//

#ifndef vec_types_h
#define vec_types_h

#include <immintrin.h>
#include <smmintrin.h>
#include <avxintrin.h>
#include <avx512fintrin.h>
#include <ostream>

#ifdef __unix__
#define SSE_ALIGN(x) x __attribute__((aligned (16)))
#elif __APPLE__
#define SSE_ALIGN(x) x __attribute__((aligned (16)))
#elif _WIN32
#define SSE_ALIGN(x) __declspec( align(16) )x
#endif

namespace __detail {
    using __128d = SSE_ALIGN(__m128d);
    using __scalar_d = double;
    
    
    template<typename T>
    struct __vector_attributes;
    
    template<>
    struct __vector_attributes<__detail::__128d>{
        typedef __detail::__128d __vector_type;
        typedef __detail::__scalar_d __scalar_type;
        static constexpr unsigned long length=2;
        
        static inline __vector_type add(__vector_type const& lhs,__vector_type const& rhs){
            return _mm_add_pd(lhs,rhs);
        }
        static inline __vector_type subtract(__vector_type const& lhs,__vector_type const& rhs){
            return _mm_sub_pd(lhs,rhs);
        }
        static inline __vector_type multiply(__vector_type const& lhs,__vector_type const& rhs){
            return _mm_mul_pd(lhs,rhs);
        }
        static inline __vector_type divide(__vector_type const& lhs,__vector_type const& rhs){
            return _mm_div_pd(lhs,rhs);
        }
        static inline __vector_type set(__scalar_type const& value){
            return _mm_set1_pd(value);
        }
        static inline __vector_type set_zero(){
            return _mm_setzero_pd();
        }
        static inline __vector_type set(__scalar_type const& _1, __scalar_type const& _2){
            return _mm_set_pd(_1, _2);
        }
        static inline __vector_type load( __scalar_type const (&_1)[2]){
            return _mm_load_pd(_1);
        }
        static inline __vector_type loadr( __scalar_type const (&_1)[2]){
            return _mm_loadr_pd(_1);
        }
        static inline __vector_type is_equal(__vector_type const& lhs, __vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_EQ_OQ), _mm_set1_pd(static_cast<unsigned long>(true)));
        }
        static inline __vector_type not_equal(__vector_type const& lhs, __vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_NEQ_OQ),_mm_set1_pd(static_cast<unsigned long>(true)));
        }
        static inline __vector_type greater_than(__vector_type const& lhs, __vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_GT_OQ),_mm_set1_pd(static_cast<unsigned long>(true)));
        }
        static inline __vector_type less_than(__vector_type const& lhs, __vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_LT_OQ),_mm_set1_pd(static_cast<unsigned long>(true)));
        }
        static inline __vector_type less_than_equal_to(__vector_type const& lhs, __vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_LE_OQ),_mm_set1_pd(static_cast<unsigned long>(true)));
        }
        static inline __vector_type greater_than_equal_to(__vector_type const& lhs, __vector_type const&  rhs){
            return _mm_and_pd(_mm_cmp_pd(lhs, rhs,_CMP_GE_OQ),_mm_set1_pd(static_cast<unsigned long>(true)));
        }
        static inline __vector_type bitwise_and(__vector_type const& lhs, __vector_type const& rhs){
            return _mm_and_pd(lhs,rhs);
        }
        static inline __vector_type bitwise_or(__vector_type const& lhs, __vector_type const& rhs){
            return _mm_or_pd(lhs, rhs);
        }
        static inline __vector_type bitwise_xor(__vector_type const& lhs, __vector_type const& rhs){
            return _mm_xor_pd(lhs, rhs);
        }
        static inline __vector_type bitwise_not( __vector_type const& rhs){
            return _mm_xor_pd(rhs,_mm_castsi128_pd(_mm_set1_epi32(-1)));
        }
        static inline __vector_type floor(__vector_type const& _1){
            return _mm_floor_pd(_1);
        }
        static inline __vector_type ceil(__vector_type const& _1){
            return _mm_ceil_pd(_1);
        }
        static inline __vector_type hadd(__vector_type const& _1,__vector_type const& _2){
            return _mm_hadd_pd(_1, _2);
        }
        static inline void store(__vector_type const& _1,__scalar_type(&_2)[2]){
            _mm_store_pd(_2, _1);
        }
        static inline __vector_type abs(__vector_type const& value){
            static const __vector_type sign_mask = _mm_set1_pd(-0.); // -0. = 1 << 63
            return _mm_andnot_pd(sign_mask, value); // !sign_mask & x
        }
        static inline __vector_type mod(__vector_type const& lhs,__vector_type const& rhs){
            __vector_type sign,d,temp;
            sign= _mm_cmpge_pd(lhs, _mm_set1_pd(0.0));
            sign = _mm_and_pd(sign, _mm_set1_pd(2.0));
            sign = _mm_sub_pd(sign, _mm_set1_pd(1.0));
            d = _mm_mul_pd(sign,rhs);
            temp = _mm_div_pd(lhs, d);
            temp = _mm_floor_pd(temp);
            temp = _mm_mul_pd(temp, d);
            temp = _mm_sub_pd(lhs, temp);
            return temp;
        }
    };

}
namespace zmath {
    typedef __detail::__vector_attributes<__detail::__128d>::__scalar_type scalar;
    typedef __detail::__vector_attributes<__detail::__128d> vector_attributes;
    template<unsigned long _length>
    class vector{
        private:
        using attributes =  __detail::__vector_attributes<__detail::__128d>;
        using scalar_type = attributes::__scalar_type;
        using base_vector = attributes::__vector_type;
        static constexpr unsigned long base_vector_length = attributes::length;
        static constexpr unsigned long array_length = _length/base_vector_length;
        
        using self_type = vector<_length>;
        
        static_assert(_length%2==0,"vector length must be even!");
        
        
        base_vector __data[array_length];
    public:
        explicit vector(){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::set_zero();
            }
        }
        explicit vector(scalar_type const& value){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::set(value);
            }
        }
        vector(self_type const& other){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = other.__data[i];
            }
        }
        self_type& operator=(self_type const& other){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = other.__data[i];
            }
            return *this;
        }
        scalar_type const& operator[](unsigned long const& index)const{
            unsigned long i = index/2;
            const scalar_type (&temp)[2] = reinterpret_cast<const scalar_type(&)[2]>(__data[i]);
            return (index%2==0) ? temp[0]:temp[1];
        }
        scalar_type& operator[](unsigned long const& index){
            unsigned long i = index/2;
            scalar_type (&temp)[2] = reinterpret_cast<scalar_type(&)[2]>(__data[i]);
            return (index%2==0) ? temp[0]:temp[1];
        }
        //add
        friend self_type operator+(scalar_type const& _1,self_type const& _2){
            self_type temp(_2);
            base_vector t = attributes::set(_1);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::add(temp.__data[i], t);
            }
            return temp;
        }
        friend self_type operator+(self_type const& _1,scalar_type const& _2){
            self_type temp(_1);
            base_vector t = attributes::set(_2);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::add(temp.__data[i], t);
            }
            return temp;
        }
        self_type operator+(self_type const& rhs)const{
            self_type temp(*this);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::add(temp.__data[i], rhs.__data[i]);
            }
            return temp;
        }
        self_type& operator+=(self_type const& rhs){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::add(__data[i], rhs.__data[i]);
            }
            return *this;
        }
        self_type& operator+=(scalar_type const& rhs){
            base_vector t = attributes::set(rhs);
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::add(__data[i], t);
            }
            return *this;
        }

        //sub
        
        friend self_type operator-(scalar_type const& _1,self_type const& _2){
            self_type temp(_2);
            base_vector t = attributes::set(_1);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::subtract( t,temp.__data[i]);
            }
            return temp;
        }
        friend self_type operator-(self_type const& _1,scalar_type const& _2){
            self_type temp(_1);
            base_vector t = attributes::set(_2);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::subtract(temp.__data[i], t);
            }
            return temp;
        }
        self_type operator-(self_type const& rhs)const{
            self_type temp(*this);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::subtract(temp.__data[i], rhs.__data[i]);
            }
            return temp;
        }
        self_type& operator-=(self_type const& rhs){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::subtract(__data[i], rhs.__data[i]);
            }
            return *this;
        }
        self_type& operator-=(scalar_type const& rhs){
            base_vector t = attributes::set(rhs);
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::subtract(__data[i], t);
            }
            return *this;
        }
        
        //mult
        friend self_type operator*(scalar_type const& _1,self_type const& _2){
            self_type temp(_2);
            base_vector t = attributes::set(_1);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::multiply( t,temp.__data[i]);
            }
            return temp;
        }
        friend self_type operator*(self_type const& _1,scalar_type const& _2){
            self_type temp(_1);
            base_vector t = attributes::set(_2);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::multiply(temp.__data[i], t);
            }
            return temp;
        }
        self_type operator*(self_type const& rhs)const{
            self_type temp(*this);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::multiply(temp.__data[i], rhs.__data[i]);
            }
            return temp;
        }
        self_type& operator*=(self_type const& rhs){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::multiply(__data[i], rhs.__data[i]);
            }
            return *this;
        }
        self_type& operator*=(scalar_type const& rhs){
            base_vector t = attributes::set(rhs);
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::multiply(__data[i], t);
            }
            return *this;
        }
        //divide
        friend self_type operator/(scalar_type const& _1,self_type const& _2){
            self_type temp(_2);
            base_vector t = attributes::set(_1);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::divide( t,temp.__data[i]);
            }
            return temp;
        }
        friend self_type operator/(self_type const& _1,scalar_type const& _2){
            self_type temp(_1);
            base_vector t = attributes::set(_2);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::divide(temp.__data[i], t);
            }
            return temp;
        }
        self_type operator/(self_type const& rhs)const{
            self_type temp(*this);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::divide(temp.__data[i], rhs.__data[i]);
            }
            return temp;
        }
        self_type& operator/=(self_type const& rhs){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::divide(__data[i], rhs.__data[i]);
            }
            return *this;
        }
        self_type& operator/=(scalar_type const& rhs){
            base_vector t = attributes::set(rhs);
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::divide(__data[i], t);
            }
            return *this;
        }
        
        //mod
        friend self_type operator%(scalar_type const& _1,self_type const& _2){
            self_type temp(_2);
            base_vector t = attributes::set(_1);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::mod( t,temp.__data[i]);
            }
            return temp;
        }
        friend self_type operator%(self_type const& _1,scalar_type const& _2){
            self_type temp(_1);
            base_vector t = attributes::set(_2);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::mod(temp.__data[i], t);
            }
            return temp;
        }
        self_type operator%(self_type const& rhs)const{
            self_type temp(*this);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::mod(temp.__data[i], rhs.__data[i]);
            }
            return temp;
        }
        self_type& operator%=(self_type const& rhs){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::mod(__data[i], rhs.__data[i]);
            }
            return *this;
        }
        self_type& operator%=(scalar_type const& rhs){
            base_vector t = attributes::set(rhs);
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::mod(__data[i], t);
            }
            return *this;
        }
        
        //and
        
        friend self_type operator&(scalar_type const& _1,self_type const& _2){
            self_type temp(_2);
            base_vector t = attributes::set(_1);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_and( t,temp.__data[i]);
            }
            return temp;
        }
        friend self_type operator&(self_type const& _1,scalar_type const& _2){
            self_type temp(_1);
            base_vector t = attributes::set(_2);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_and(temp.__data[i], t);
            }
            return temp;
        }
        self_type operator&(self_type const& rhs){
            self_type temp(*this);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_and(temp.__data[i], rhs.__data[i]);
            }
            return temp;
        }
        self_type& operator&=(self_type const& rhs){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::bitwise_and(__data[i], rhs.__data[i]);
            }
            return *this;
        }
        self_type& operator&=(scalar_type const& rhs){
            base_vector t = attributes::set(rhs);
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::bitwise_and(__data[i], t);
            }
            return *this;
        }
        
        //or
        
        friend self_type operator|(scalar_type const& _1,self_type const& _2){
            self_type temp(_2);
            base_vector t = attributes::set(_1);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_or( t,temp.__data[i]);
            }
            return temp;
        }
        friend self_type operator|(self_type const& _1,scalar_type const& _2){
            self_type temp(_1);
            base_vector t = attributes::set(_2);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_or(temp.__data[i], t);
            }
            return temp;
        }
        self_type operator|(self_type const& rhs){
            self_type temp(*this);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_or(temp.__data[i], rhs.__data[i]);
            }
            return temp;
        }
        self_type& operator|=(self_type const& rhs){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::bitwise_or(__data[i], rhs.__data[i]);
            }
            return *this;
        }
        self_type& operator|=(scalar_type const& rhs){
            base_vector t = attributes::set(rhs);
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::bitwise_or(__data[i], t);
            }
            return *this;
        }
        //xor
        
        friend self_type operator^(scalar_type const& _1,self_type const& _2){
            self_type temp(_2);
            base_vector t = attributes::set(_1);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_xor( t,temp.__data[i]);
            }
            return temp;
        }
        friend self_type operator^(self_type const& _1,scalar_type const& _2){
            self_type temp(_1);
            base_vector t = attributes::set(_2);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_xor(temp.__data[i], t);
            }
            return temp;
        }
        self_type operator^(self_type const& rhs){
            self_type temp(*this);
            for (unsigned long i=0; i<temp.array_length; ++i) {
                temp.__data[i] = attributes::bitwise_xor(temp.__data[i], rhs.__data[i]);
            }
            return temp;
        }
        self_type& operator^=(self_type const& rhs){
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::bitwise_xor(__data[i], rhs.__data[i]);
            }
            return *this;
        }
        self_type& operator^=(scalar_type const& rhs){
            base_vector t = attributes::set(rhs);
            for (unsigned long i=0; i<array_length; ++i) {
                __data[i] = attributes::bitwise_xor(__data[i], t);
            }
            return *this;
        }
        
        
        
        
        //comparisons
        
        friend self_type operator==(self_type const& _1, self_type const& _2){
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::is_equal(_1.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator==(scalar_type const& _1, self_type const& _2){
            self_type temp(_1);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::is_equal(temp.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator==(self_type const& _1, scalar_type const& _2){
            self_type temp(_2);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::is_equal(_1.__data[i], temp.__data[i]);
            }
            return temp;
        }
        
        friend self_type operator!=(self_type const& _1, self_type const& _2){
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                (*temp[i]) = attributes::not_equal(_1.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator!=(scalar_type const& _1, self_type const& _2){
            self_type temp(_1);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::not_equal(temp.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator!=(self_type const& _1, scalar_type const& _2){
            self_type temp(_2);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::not_equal(_1.__data[i], temp.__data[i]);
            }
            return temp;
        }
        
        friend self_type operator>(self_type const& _1, self_type const& _2){
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                (*temp[i]) = attributes::greater_than(_1.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator>(scalar_type const& _1, self_type const& _2){
            self_type temp(_1);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::greater_than(temp.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator>(self_type const& _1, scalar_type const& _2){
            self_type temp(_2);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::greater_than(_1.__data[i], temp.__data[i]);
            }
            return temp;
        }
        
        friend self_type operator>=(self_type const& _1, self_type const& _2){
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                (*temp[i]) = attributes::greater_than_equal_to(_1.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator>=(scalar_type const& _1, self_type const& _2){
            self_type temp(_1);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::greater_than_equal_to(temp.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator>=(self_type const& _1, scalar_type const& _2){
            self_type temp(_2);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::greater_than_equal_to(_1.__data[i], temp.__data[i]);
            }
            return temp;
        }
        
        
        friend self_type operator<(self_type const& _1, self_type const& _2){
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                (*temp[i]) = attributes::less_than(_1.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator<(scalar_type const& _1, self_type const& _2){
            self_type temp(_1);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::less_than(temp.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator<(self_type const& _1, scalar_type const& _2){
            self_type temp(_2);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::less_than(_1.__data[i], temp.__data[i]);
            }
            return temp;
        }
        
        friend self_type operator<=(self_type const& _1, self_type const& _2){
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                (*temp[i]) = attributes::less_than_equal_to(_1.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator<=(scalar_type const& _1, self_type const& _2){
            self_type temp(_1);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::less_than_equal_to(temp.__data[i], _2.__data[i]);
            }
            return temp;
        }
        friend self_type operator<=(self_type const& _1, scalar_type const& _2){
            self_type temp(_2);
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::less_than_equal_to(_1.__data[i], temp.__data[i]);
            }
            return temp;
        }

        
        
        
        
        
        
        self_type operator+(){
            self_type temp;
            for (unsigned long i =0; i<array_length; ++i) {
                temp[i]=attributes::abs(__data[i]);
            }
            return temp;
        }
        self_type operator-(){
            return -1.0 * (*this);
        }
        self_type operator!(){
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                temp[i] = attributes::bitwise_not(__data[i]);
            }
            return temp;
        }
        
        self_type& operator++(){
            (*this)+=1;
            return *this;
        }
        self_type& operator++(int){
            self_type temp(*this);
            (*this)+=1;
            return temp;
        }
        self_type& operator--(){
            (*this)-=1;
            return *this;
        }
        self_type& operator--(int){
            self_type temp(*this);
            (*this)-=1;
            return temp;
        }


        constexpr inline unsigned long length()const{
            return _length;
        }
        self_type floor()const{
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::floor(__data[i]);
            }
            return temp;
        }
        self_type ceil()const{
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::ceil(__data[i]);
            }
            return temp;
        }
        self_type abs()const{
            self_type temp;
            for (unsigned long i=0; i<array_length; ++i) {
                temp.__data[i] = attributes::abs(__data[i]);
            }
            return temp;
        }
        
        
        void write(unsigned long const& index,scalar_type const& _1,scalar_type const& _2){
            __data[index]=attributes::set(_1, _2);
        }
        
        base_vector* operator*(){
            return __data;
        }
        
        friend std::ostream& operator<<(std::ostream& os,self_type const& v){
            for (unsigned long i=0; i<v.length(); ++i) {
                os<<v[i]<<std::endl;
            }
            return os;
        }
        
        
    };
    
    template<unsigned long N>
    vector<N> abs(vector<N> const& x){
        return x.abs();
    }
    template<unsigned long N>
    vector<N> floor(vector<N> const& x){
        return x.floor();
    }
    template<unsigned long N>
    vector<N> ceil(vector<N> const& x){
        return x.ceil();
    }
}

#endif
