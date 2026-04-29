#ifndef RPP_CPU_OPS_SINGLE_THREAD_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_HPP

#include <rpp/cpu/ops/single_thread/vector_set_constant.hpp>
#include <rpp/cpu/ops/single_thread/vector_set_zero.hpp>
#include <rpp/cpu/ops/single_thread/vector_assign.hpp>
#include <rpp/cpu/ops/single_thread/vector_inplace_add.hpp>
#include <rpp/cpu/ops/single_thread/tensor_add_identity.hpp>
#include <rpp/cpu/ops/single_thread/tensor_set_identity.hpp>
#include <rpp/cpu/ops/single_thread/tensor_pairing.hpp>
#include <rpp/cpu/ops/single_thread/tensor_antipode.hpp>
#include <rpp/cpu/ops/single_thread/tensor_reflect.hpp>
#include <rpp/cpu/ops/single_thread/ft_adj_lmul.hpp>
#include <rpp/cpu/ops/single_thread/ft_adj_rmul.hpp>
#include <rpp/cpu/ops/single_thread/ft_inplace_fma.hpp>
#include <rpp/cpu/ops/single_thread/ft_fma.hpp>
#include <rpp/cpu/ops/single_thread/ft_mul.hpp>
#include <rpp/cpu/ops/single_thread/ft_inplace_mul.hpp>
#include <rpp/cpu/ops/single_thread/st_fma.hpp>
#include <rpp/cpu/ops/single_thread/st_inplace_fma.hpp>
#include <rpp/cpu/ops/single_thread/st_mul.hpp>
#include <rpp/cpu/ops/single_thread/st_adj_mul.hpp>
#include <rpp/cpu/ops/single_thread/ft_exp.hpp>
#include <rpp/cpu/ops/single_thread/ft_fmexp.hpp>
#include <rpp/cpu/ops/single_thread/ft_log.hpp>

#endif // RPP_CPU_OPS_SINGLE_THREAD_HPP
