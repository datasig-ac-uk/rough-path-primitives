#ifndef RPP_OPERATIONS_HPP
#define RPP_OPERATIONS_HPP

/*****************************************************************************
 *                           Vector Operations                               *
 *****************************************************************************/

#include <rpp/operations/basic/sparse_matrix_vector.hpp>
#include <rpp/operations/basic/vector_add.hpp>
#include <rpp/operations/basic/vector_assign.hpp>
#include <rpp/operations/basic/vector_inplace_add.hpp>
#include <rpp/operations/basic/vector_scalar_multiply.hpp>
#include <rpp/operations/basic/vector_set_constant.hpp>


/*****************************************************************************
 *                        Elementary Tensor Operations                       *
 *****************************************************************************/

#include <rpp/operations/basic/tensor_add_identity.hpp>
#include <rpp/operations/basic/tensor_set_identity.hpp>



/*****************************************************************************
 *                        Basic Algebra Operations                           *
 *****************************************************************************/

#include <rpp/operations/basic/tensor_generalised_antipode.hpp>
#include <rpp/operations/basic/tensor_antipode.hpp>
#include <rpp/operations/basic/tensor_reflect.hpp>
#include <rpp/operations/basic/ft_inplace_fma.hpp>
#include <rpp/operations/basic/ft_mul.hpp>
#include <rpp/operations/basic/ft_fma.hpp>
#include <rpp/operations/basic/ft_inplace_mul.hpp>
#include <rpp/operations/basic/ft_adj_lmul.hpp>
#include <rpp/operations/basic/ft_adj_rmul.hpp>
#include <rpp/operations/basic/st_mul.hpp>
#include <rpp/operations/basic/st_inplace_fma.hpp>
#include <rpp/operations/basic/st_fma.hpp>
#include <rpp/operations/basic/st_adj_mul.hpp>
#include <rpp/operations/basic/tensor_pairing.hpp>
#include <rpp/operations/basic/lie_to_tensor.hpp>
#include <rpp/operations/basic/tensor_to_lie.hpp>


/*****************************************************************************
 *                    Intermediate Algebra Operations                        *
 *****************************************************************************/

#include <rpp/operations/intermediate/ft_exp.hpp>
#include <rpp/operations/intermediate/ft_fmexp.hpp>
#include <rpp/operations/intermediate/ft_log.hpp>



#endif // RPP_OPERATIONS_HPP
