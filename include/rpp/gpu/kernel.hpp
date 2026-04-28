// ReSharper disable CppRedundantTypenameKeyword
#ifndef RPP_GPU_KERNEL_HPP
#define RPP_GPU_KERNEL_HPP

#include <rpp/gpu/ops/block/vector_set_constant.hpp>
#include <rpp/gpu/ops/block/vector_set_zero.hpp>
#include <rpp/gpu/ops/block/vector_assign.hpp>
#include <rpp/gpu/ops/block/vector_inplace_add.hpp>
#include <rpp/gpu/ops/block/tensor_add_identity.hpp>
#include <rpp/gpu/ops/block/tensor_set_identity.hpp>
#include <rpp/gpu/ops/block/tensor_antipode.hpp>
#include <rpp/gpu/ops/block/tensor_reflect.hpp>
#include <rpp/gpu/ops/block/ft_adj_lmul.hpp>
#include <rpp/gpu/ops/block/ft_adj_rmul.hpp>
#include <rpp/gpu/ops/block/ft_inplace_fma.hpp>
#include <rpp/gpu/ops/block/ft_fma.hpp>
#include <rpp/gpu/ops/block/ft_mul.hpp>
#include <rpp/gpu/ops/block/ft_inplace_mul.hpp>
#include <rpp/gpu/ops/block/st_fma.hpp>
#include <rpp/gpu/ops/block/st_inplace_fma.hpp>
#include <rpp/gpu/ops/block/st_mul.hpp>
#include <rpp/gpu/ops/block/st_adj_mul.hpp>
#include <rpp/gpu/ops/block/ft_exp.hpp>
#include <rpp/gpu/ops/block/ft_fmexp.hpp>
#include <rpp/gpu/ops/block/ft_log.hpp>

#endif // RPP_GPU_KERNEL_HPP
