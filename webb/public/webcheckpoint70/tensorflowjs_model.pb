
<
g_0/g_0_bn_5/betaConst*
value
B@*
dtype0
I
g_0/g_0_deconv_5wConst*
dtype0* 
valueB@�
=
g_0/g_0_bn_4/betaConst*
dtype0*
valueB	�
J
g_0/g_0_deconv_4wConst*
dtype0*!
valueB��
=
g_0/g_0_bn_3/betaConst*
valueB	�*
dtype0
=
g_0/g_0_bn_2/betaConst*
valueB	�*
dtype0
<
g_0/g_0_bn_1/betaConst*
value
B@*
dtype0
F
g_0/g_0_conv_1wConst*
valueB@*
dtype0
?
g_0/g_0_conv_1biasesConst*
dtype0*
value
B@
=
g_0/g_0_bn_1/gammaConst*
value
B@*
dtype0
G
g_0/g_0_conv_2wConst* 
valueB@�*
dtype0
@
g_0/g_0_conv_2biasesConst*
dtype0*
valueB	�
>
g_0/g_0_bn_2/gammaConst*
valueB	�*
dtype0
H
g_0/g_0_conv_3wConst*!
valueB��*
dtype0
@
g_0/g_0_conv_3biasesConst*
valueB	�*
dtype0
>
g_0/g_0_bn_3/gammaConst*
valueB	�*
dtype0
B
g_0/g_0_deconv_4biasesConst*
dtype0*
valueB	�
>
g_0/g_0_bn_4/gammaConst*
valueB	�*
dtype0
A
g_0/g_0_deconv_5biasesConst*
dtype0*
value
B@
=
g_0/g_0_bn_5/gammaConst*
value
B@*
dtype0
F
g_0/g_0_conv_6wConst*
valueB@*
dtype0
?
g_0/g_0_conv_6biasesConst*
value
B*
dtype0
<
g_1/g_1_bn_5/betaConst*
value
B@*
dtype0
I
g_1/g_1_deconv_5wConst* 
valueB@�*
dtype0
=
g_1/g_1_bn_4/betaConst*
valueB	�*
dtype0
6
g_0_2/ConstConst*
dtype0*
value
B
J
g_1/g_1_deconv_4wConst*
dtype0*!
valueB��
=
g_1/g_1_bn_3/betaConst*
dtype0*
valueB	�
=
g_1/g_1_bn_2/betaConst*
valueB	�*
dtype0
>
g_0_2/LeakyRelu_4/alphaConst*
valueB *
dtype0
<
g_1/g_1_bn_1/betaConst*
value
B@*
dtype0
F
g_1/g_1_conv_1wConst*
valueB@*
dtype0
?
g_1/g_1_conv_1biasesConst*
value
B@*
dtype0
8
g_0_2/Const_1Const*
value
B*
dtype0
=
g_1/g_1_bn_1/gammaConst*
value
B@*
dtype0
G
g_1/g_1_conv_2wConst* 
valueB@�*
dtype0
@
g_1/g_1_conv_2biasesConst*
valueB	�*
dtype0
>
g_1/g_1_bn_2/gammaConst*
valueB	�*
dtype0
H
g_1/g_1_conv_3wConst*!
valueB��*
dtype0
@
g_1/g_1_conv_3biasesConst*
valueB	�*
dtype0
>
g_1/g_1_bn_3/gammaConst*
valueB	�*
dtype0
B
g_1/g_1_deconv_4biasesConst*
valueB	�*
dtype0
>
g_1/g_1_bn_4/gammaConst*
valueB	�*
dtype0
A
g_1/g_1_deconv_5biasesConst*
value
B@*
dtype0
X
-g_0_2/g_0_bn_1/moments/mean/reduction_indicesConst*
value
B*
dtype0
H
!g_0_2/g_0_bn_1/instancenorm/add/yConst*
dtype0*
valueB 
=
g_1/g_1_bn_5/gammaConst*
value
B@*
dtype0
F
g_1/g_1_conv_6wConst*
valueB@*
dtype0
?
g_1/g_1_conv_6biasesConst*
value
B*
dtype0
A
pred_0Placeholder*
shape:��*
dtype0
A
pred_1Placeholder*
shape:��*
dtype0
�
g_0_2/g_0_conv_1Conv2Dpred_0g_0/g_0_conv_1w*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
g_1_2/g_1_conv_1Conv2Dpred_1g_1/g_1_conv_1w*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
`
g_0_2/BiasAddBiasAddg_0_2/g_0_conv_1g_0/g_0_conv_1biases*
T0*
data_formatNHWC
`
g_1_2/BiasAddBiasAddg_1_2/g_1_conv_1g_1/g_1_conv_1biases*
T0*
data_formatNHWC
M
g_0_2/ReshapeReshapeg_0_2/BiasAddg_0_2/Const_1*
T0*
Tshape0
M
g_1_2/ReshapeReshapeg_1_2/BiasAddg_0_2/Const_1*
Tshape0*
T0
�
g_0_2/g_0_bn_1/moments/meanMeang_0_2/Reshape-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_1/moments/meanMeang_1_2/Reshape-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
r
(g_0_2/g_0_bn_1/moments/SquaredDifferenceSquaredDifferenceg_0_2/Reshapeg_0_2/g_0_bn_1/moments/mean*
T0
r
(g_1_2/g_1_bn_1/moments/SquaredDifferenceSquaredDifferenceg_1_2/Reshapeg_1_2/g_1_bn_1/moments/mean*
T0
�
g_0_2/g_0_bn_1/moments/varianceMean(g_0_2/g_0_bn_1/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_1/moments/varianceMean(g_1_2/g_1_bn_1/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
s
g_0_2/g_0_bn_1/instancenorm/addAddg_0_2/g_0_bn_1/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
s
g_1_2/g_1_bn_1/instancenorm/addAddg_1_2/g_1_bn_1/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
T
!g_0_2/g_0_bn_1/instancenorm/RsqrtRsqrtg_0_2/g_0_bn_1/instancenorm/add*
T0
T
!g_1_2/g_1_bn_1/instancenorm/RsqrtRsqrtg_1_2/g_1_bn_1/instancenorm/add*
T0
f
g_0_2/g_0_bn_1/instancenorm/mulMul!g_0_2/g_0_bn_1/instancenorm/Rsqrtg_0/g_0_bn_1/gamma*
T0
f
g_1_2/g_1_bn_1/instancenorm/mulMul!g_1_2/g_1_bn_1/instancenorm/Rsqrtg_1/g_1_bn_1/gamma*
T0
a
!g_0_2/g_0_bn_1/instancenorm/mul_1Mulg_0_2/Reshapeg_0_2/g_0_bn_1/instancenorm/mul*
T0
o
!g_0_2/g_0_bn_1/instancenorm/mul_2Mulg_0_2/g_0_bn_1/moments/meang_0_2/g_0_bn_1/instancenorm/mul*
T0
a
!g_1_2/g_1_bn_1/instancenorm/mul_1Mulg_1_2/Reshapeg_1_2/g_1_bn_1/instancenorm/mul*
T0
o
!g_1_2/g_1_bn_1/instancenorm/mul_2Mulg_1_2/g_1_bn_1/moments/meang_1_2/g_1_bn_1/instancenorm/mul*
T0
e
g_0_2/g_0_bn_1/instancenorm/subSubg_0/g_0_bn_1/beta!g_0_2/g_0_bn_1/instancenorm/mul_2*
T0
e
g_1_2/g_1_bn_1/instancenorm/subSubg_1/g_1_bn_1/beta!g_1_2/g_1_bn_1/instancenorm/mul_2*
T0
u
!g_0_2/g_0_bn_1/instancenorm/add_1Add!g_0_2/g_0_bn_1/instancenorm/mul_1g_0_2/g_0_bn_1/instancenorm/sub*
T0
u
!g_1_2/g_1_bn_1/instancenorm/add_1Add!g_1_2/g_1_bn_1/instancenorm/mul_1g_1_2/g_1_bn_1/instancenorm/sub*
T0
_
g_0_2/LeakyRelu/mulMulg_0_2/LeakyRelu_4/alpha!g_0_2/g_0_bn_1/instancenorm/add_1*
T0
_
g_1_2/LeakyRelu/mulMulg_0_2/LeakyRelu_4/alpha!g_1_2/g_1_bn_1/instancenorm/add_1*
T0
[
g_0_2/LeakyReluMaximumg_0_2/LeakyRelu/mul!g_0_2/g_0_bn_1/instancenorm/add_1*
T0
[
g_1_2/LeakyReluMaximumg_1_2/LeakyRelu/mul!g_1_2/g_1_bn_1/instancenorm/add_1*
T0
�
g_0_2/g_0_conv_2Conv2Dg_0_2/LeakyRelug_0/g_0_conv_2w*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
g_1_2/g_1_conv_2Conv2Dg_1_2/LeakyRelug_1/g_1_conv_2w*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
b
g_0_2/BiasAdd_1BiasAddg_0_2/g_0_conv_2g_0/g_0_conv_2biases*
T0*
data_formatNHWC
b
g_1_2/BiasAdd_1BiasAddg_1_2/g_1_conv_2g_1/g_1_conv_2biases*
T0*
data_formatNHWC
�
g_0_2/g_0_bn_2/moments/meanMeang_0_2/BiasAdd_1-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_2/moments/meanMeang_1_2/BiasAdd_1-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
t
(g_0_2/g_0_bn_2/moments/SquaredDifferenceSquaredDifferenceg_0_2/BiasAdd_1g_0_2/g_0_bn_2/moments/mean*
T0
t
(g_1_2/g_1_bn_2/moments/SquaredDifferenceSquaredDifferenceg_1_2/BiasAdd_1g_1_2/g_1_bn_2/moments/mean*
T0
�
g_0_2/g_0_bn_2/moments/varianceMean(g_0_2/g_0_bn_2/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(
�
g_1_2/g_1_bn_2/moments/varianceMean(g_1_2/g_1_bn_2/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(
s
g_0_2/g_0_bn_2/instancenorm/addAddg_0_2/g_0_bn_2/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
s
g_1_2/g_1_bn_2/instancenorm/addAddg_1_2/g_1_bn_2/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
T
!g_0_2/g_0_bn_2/instancenorm/RsqrtRsqrtg_0_2/g_0_bn_2/instancenorm/add*
T0
T
!g_1_2/g_1_bn_2/instancenorm/RsqrtRsqrtg_1_2/g_1_bn_2/instancenorm/add*
T0
f
g_0_2/g_0_bn_2/instancenorm/mulMul!g_0_2/g_0_bn_2/instancenorm/Rsqrtg_0/g_0_bn_2/gamma*
T0
f
g_1_2/g_1_bn_2/instancenorm/mulMul!g_1_2/g_1_bn_2/instancenorm/Rsqrtg_1/g_1_bn_2/gamma*
T0
c
!g_0_2/g_0_bn_2/instancenorm/mul_1Mulg_0_2/BiasAdd_1g_0_2/g_0_bn_2/instancenorm/mul*
T0
o
!g_0_2/g_0_bn_2/instancenorm/mul_2Mulg_0_2/g_0_bn_2/moments/meang_0_2/g_0_bn_2/instancenorm/mul*
T0
c
!g_1_2/g_1_bn_2/instancenorm/mul_1Mulg_1_2/BiasAdd_1g_1_2/g_1_bn_2/instancenorm/mul*
T0
o
!g_1_2/g_1_bn_2/instancenorm/mul_2Mulg_1_2/g_1_bn_2/moments/meang_1_2/g_1_bn_2/instancenorm/mul*
T0
e
g_0_2/g_0_bn_2/instancenorm/subSubg_0/g_0_bn_2/beta!g_0_2/g_0_bn_2/instancenorm/mul_2*
T0
e
g_1_2/g_1_bn_2/instancenorm/subSubg_1/g_1_bn_2/beta!g_1_2/g_1_bn_2/instancenorm/mul_2*
T0
u
!g_0_2/g_0_bn_2/instancenorm/add_1Add!g_0_2/g_0_bn_2/instancenorm/mul_1g_0_2/g_0_bn_2/instancenorm/sub*
T0
u
!g_1_2/g_1_bn_2/instancenorm/add_1Add!g_1_2/g_1_bn_2/instancenorm/mul_1g_1_2/g_1_bn_2/instancenorm/sub*
T0
a
g_0_2/LeakyRelu_1/mulMulg_0_2/LeakyRelu_4/alpha!g_0_2/g_0_bn_2/instancenorm/add_1*
T0
a
g_1_2/LeakyRelu_1/mulMulg_0_2/LeakyRelu_4/alpha!g_1_2/g_1_bn_2/instancenorm/add_1*
T0
_
g_0_2/LeakyRelu_1Maximumg_0_2/LeakyRelu_1/mul!g_0_2/g_0_bn_2/instancenorm/add_1*
T0
_
g_1_2/LeakyRelu_1Maximumg_1_2/LeakyRelu_1/mul!g_1_2/g_1_bn_2/instancenorm/add_1*
T0
�
g_0_2/g_0_conv_3Conv2Dg_0_2/LeakyRelu_1g_0/g_0_conv_3w*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
g_1_2/g_1_conv_3Conv2Dg_1_2/LeakyRelu_1g_1/g_1_conv_3w*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations

b
g_0_2/BiasAdd_2BiasAddg_0_2/g_0_conv_3g_0/g_0_conv_3biases*
T0*
data_formatNHWC
b
g_1_2/BiasAdd_2BiasAddg_1_2/g_1_conv_3g_1/g_1_conv_3biases*
T0*
data_formatNHWC
�
g_0_2/g_0_bn_3/moments/meanMeang_0_2/BiasAdd_2-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_3/moments/meanMeang_1_2/BiasAdd_2-g_0_2/g_0_bn_1/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(
t
(g_0_2/g_0_bn_3/moments/SquaredDifferenceSquaredDifferenceg_0_2/BiasAdd_2g_0_2/g_0_bn_3/moments/mean*
T0
t
(g_1_2/g_1_bn_3/moments/SquaredDifferenceSquaredDifferenceg_1_2/BiasAdd_2g_1_2/g_1_bn_3/moments/mean*
T0
�
g_0_2/g_0_bn_3/moments/varianceMean(g_0_2/g_0_bn_3/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_3/moments/varianceMean(g_1_2/g_1_bn_3/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
s
g_0_2/g_0_bn_3/instancenorm/addAddg_0_2/g_0_bn_3/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
s
g_1_2/g_1_bn_3/instancenorm/addAddg_1_2/g_1_bn_3/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
T
!g_0_2/g_0_bn_3/instancenorm/RsqrtRsqrtg_0_2/g_0_bn_3/instancenorm/add*
T0
T
!g_1_2/g_1_bn_3/instancenorm/RsqrtRsqrtg_1_2/g_1_bn_3/instancenorm/add*
T0
f
g_0_2/g_0_bn_3/instancenorm/mulMul!g_0_2/g_0_bn_3/instancenorm/Rsqrtg_0/g_0_bn_3/gamma*
T0
f
g_1_2/g_1_bn_3/instancenorm/mulMul!g_1_2/g_1_bn_3/instancenorm/Rsqrtg_1/g_1_bn_3/gamma*
T0
c
!g_0_2/g_0_bn_3/instancenorm/mul_1Mulg_0_2/BiasAdd_2g_0_2/g_0_bn_3/instancenorm/mul*
T0
o
!g_0_2/g_0_bn_3/instancenorm/mul_2Mulg_0_2/g_0_bn_3/moments/meang_0_2/g_0_bn_3/instancenorm/mul*
T0
c
!g_1_2/g_1_bn_3/instancenorm/mul_1Mulg_1_2/BiasAdd_2g_1_2/g_1_bn_3/instancenorm/mul*
T0
o
!g_1_2/g_1_bn_3/instancenorm/mul_2Mulg_1_2/g_1_bn_3/moments/meang_1_2/g_1_bn_3/instancenorm/mul*
T0
e
g_0_2/g_0_bn_3/instancenorm/subSubg_0/g_0_bn_3/beta!g_0_2/g_0_bn_3/instancenorm/mul_2*
T0
e
g_1_2/g_1_bn_3/instancenorm/subSubg_1/g_1_bn_3/beta!g_1_2/g_1_bn_3/instancenorm/mul_2*
T0
u
!g_0_2/g_0_bn_3/instancenorm/add_1Add!g_0_2/g_0_bn_3/instancenorm/mul_1g_0_2/g_0_bn_3/instancenorm/sub*
T0
u
!g_1_2/g_1_bn_3/instancenorm/add_1Add!g_1_2/g_1_bn_3/instancenorm/mul_1g_1_2/g_1_bn_3/instancenorm/sub*
T0
a
g_0_2/LeakyRelu_2/mulMulg_0_2/LeakyRelu_4/alpha!g_0_2/g_0_bn_3/instancenorm/add_1*
T0
a
g_1_2/LeakyRelu_2/mulMulg_0_2/LeakyRelu_4/alpha!g_1_2/g_1_bn_3/instancenorm/add_1*
T0
_
g_0_2/LeakyRelu_2Maximumg_0_2/LeakyRelu_2/mul!g_0_2/g_0_bn_3/instancenorm/add_1*
T0
_
g_1_2/LeakyRelu_2Maximumg_1_2/LeakyRelu_2/mul!g_1_2/g_1_bn_3/instancenorm/add_1*
T0
�
g_0_2/g_0_deconv_4Conv2DBackpropInputg_0_2/Constg_0/g_0_deconv_4wg_0_2/LeakyRelu_2*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
g_1_2/g_1_deconv_4Conv2DBackpropInputg_0_2/Constg_1/g_1_deconv_4wg_1_2/LeakyRelu_2*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
f
g_0_2/BiasAdd_3BiasAddg_0_2/g_0_deconv_4g_0/g_0_deconv_4biases*
data_formatNHWC*
T0
f
g_1_2/BiasAdd_3BiasAddg_1_2/g_1_deconv_4g_1/g_1_deconv_4biases*
T0*
data_formatNHWC
�
g_0_2/g_0_bn_4/moments/meanMeang_0_2/BiasAdd_3-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_4/moments/meanMeang_1_2/BiasAdd_3-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
t
(g_0_2/g_0_bn_4/moments/SquaredDifferenceSquaredDifferenceg_0_2/BiasAdd_3g_0_2/g_0_bn_4/moments/mean*
T0
t
(g_1_2/g_1_bn_4/moments/SquaredDifferenceSquaredDifferenceg_1_2/BiasAdd_3g_1_2/g_1_bn_4/moments/mean*
T0
�
g_0_2/g_0_bn_4/moments/varianceMean(g_0_2/g_0_bn_4/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_4/moments/varianceMean(g_1_2/g_1_bn_4/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(
s
g_0_2/g_0_bn_4/instancenorm/addAddg_0_2/g_0_bn_4/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
s
g_1_2/g_1_bn_4/instancenorm/addAddg_1_2/g_1_bn_4/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
T
!g_0_2/g_0_bn_4/instancenorm/RsqrtRsqrtg_0_2/g_0_bn_4/instancenorm/add*
T0
T
!g_1_2/g_1_bn_4/instancenorm/RsqrtRsqrtg_1_2/g_1_bn_4/instancenorm/add*
T0
f
g_0_2/g_0_bn_4/instancenorm/mulMul!g_0_2/g_0_bn_4/instancenorm/Rsqrtg_0/g_0_bn_4/gamma*
T0
f
g_1_2/g_1_bn_4/instancenorm/mulMul!g_1_2/g_1_bn_4/instancenorm/Rsqrtg_1/g_1_bn_4/gamma*
T0
c
!g_0_2/g_0_bn_4/instancenorm/mul_1Mulg_0_2/BiasAdd_3g_0_2/g_0_bn_4/instancenorm/mul*
T0
o
!g_0_2/g_0_bn_4/instancenorm/mul_2Mulg_0_2/g_0_bn_4/moments/meang_0_2/g_0_bn_4/instancenorm/mul*
T0
c
!g_1_2/g_1_bn_4/instancenorm/mul_1Mulg_1_2/BiasAdd_3g_1_2/g_1_bn_4/instancenorm/mul*
T0
o
!g_1_2/g_1_bn_4/instancenorm/mul_2Mulg_1_2/g_1_bn_4/moments/meang_1_2/g_1_bn_4/instancenorm/mul*
T0
e
g_0_2/g_0_bn_4/instancenorm/subSubg_0/g_0_bn_4/beta!g_0_2/g_0_bn_4/instancenorm/mul_2*
T0
e
g_1_2/g_1_bn_4/instancenorm/subSubg_1/g_1_bn_4/beta!g_1_2/g_1_bn_4/instancenorm/mul_2*
T0
u
!g_0_2/g_0_bn_4/instancenorm/add_1Add!g_0_2/g_0_bn_4/instancenorm/mul_1g_0_2/g_0_bn_4/instancenorm/sub*
T0
u
!g_1_2/g_1_bn_4/instancenorm/add_1Add!g_1_2/g_1_bn_4/instancenorm/mul_1g_1_2/g_1_bn_4/instancenorm/sub*
T0
a
g_0_2/LeakyRelu_3/mulMulg_0_2/LeakyRelu_4/alpha!g_0_2/g_0_bn_4/instancenorm/add_1*
T0
a
g_1_2/LeakyRelu_3/mulMulg_0_2/LeakyRelu_4/alpha!g_1_2/g_1_bn_4/instancenorm/add_1*
T0
_
g_0_2/LeakyRelu_3Maximumg_0_2/LeakyRelu_3/mul!g_0_2/g_0_bn_4/instancenorm/add_1*
T0
_
g_1_2/LeakyRelu_3Maximumg_1_2/LeakyRelu_3/mul!g_1_2/g_1_bn_4/instancenorm/add_1*
T0
�
g_0_2/g_0_deconv_5Conv2DBackpropInputg_0_2/Const_1g_0/g_0_deconv_5wg_0_2/LeakyRelu_3*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
g_1_2/g_1_deconv_5Conv2DBackpropInputg_0_2/Const_1g_1/g_1_deconv_5wg_1_2/LeakyRelu_3*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
f
g_0_2/BiasAdd_4BiasAddg_0_2/g_0_deconv_5g_0/g_0_deconv_5biases*
T0*
data_formatNHWC
f
g_1_2/BiasAdd_4BiasAddg_1_2/g_1_deconv_5g_1/g_1_deconv_5biases*
T0*
data_formatNHWC
�
g_0_2/g_0_bn_5/moments/meanMeang_0_2/BiasAdd_4-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_5/moments/meanMeang_1_2/BiasAdd_4-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
t
(g_0_2/g_0_bn_5/moments/SquaredDifferenceSquaredDifferenceg_0_2/BiasAdd_4g_0_2/g_0_bn_5/moments/mean*
T0
t
(g_1_2/g_1_bn_5/moments/SquaredDifferenceSquaredDifferenceg_1_2/BiasAdd_4g_1_2/g_1_bn_5/moments/mean*
T0
�
g_0_2/g_0_bn_5/moments/varianceMean(g_0_2/g_0_bn_5/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
g_1_2/g_1_bn_5/moments/varianceMean(g_1_2/g_1_bn_5/moments/SquaredDifference-g_0_2/g_0_bn_1/moments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(
s
g_0_2/g_0_bn_5/instancenorm/addAddg_0_2/g_0_bn_5/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
s
g_1_2/g_1_bn_5/instancenorm/addAddg_1_2/g_1_bn_5/moments/variance!g_0_2/g_0_bn_1/instancenorm/add/y*
T0
T
!g_0_2/g_0_bn_5/instancenorm/RsqrtRsqrtg_0_2/g_0_bn_5/instancenorm/add*
T0
T
!g_1_2/g_1_bn_5/instancenorm/RsqrtRsqrtg_1_2/g_1_bn_5/instancenorm/add*
T0
f
g_0_2/g_0_bn_5/instancenorm/mulMul!g_0_2/g_0_bn_5/instancenorm/Rsqrtg_0/g_0_bn_5/gamma*
T0
f
g_1_2/g_1_bn_5/instancenorm/mulMul!g_1_2/g_1_bn_5/instancenorm/Rsqrtg_1/g_1_bn_5/gamma*
T0
c
!g_0_2/g_0_bn_5/instancenorm/mul_1Mulg_0_2/BiasAdd_4g_0_2/g_0_bn_5/instancenorm/mul*
T0
o
!g_0_2/g_0_bn_5/instancenorm/mul_2Mulg_0_2/g_0_bn_5/moments/meang_0_2/g_0_bn_5/instancenorm/mul*
T0
c
!g_1_2/g_1_bn_5/instancenorm/mul_1Mulg_1_2/BiasAdd_4g_1_2/g_1_bn_5/instancenorm/mul*
T0
o
!g_1_2/g_1_bn_5/instancenorm/mul_2Mulg_1_2/g_1_bn_5/moments/meang_1_2/g_1_bn_5/instancenorm/mul*
T0
e
g_0_2/g_0_bn_5/instancenorm/subSubg_0/g_0_bn_5/beta!g_0_2/g_0_bn_5/instancenorm/mul_2*
T0
e
g_1_2/g_1_bn_5/instancenorm/subSubg_1/g_1_bn_5/beta!g_1_2/g_1_bn_5/instancenorm/mul_2*
T0
u
!g_0_2/g_0_bn_5/instancenorm/add_1Add!g_0_2/g_0_bn_5/instancenorm/mul_1g_0_2/g_0_bn_5/instancenorm/sub*
T0
u
!g_1_2/g_1_bn_5/instancenorm/add_1Add!g_1_2/g_1_bn_5/instancenorm/mul_1g_1_2/g_1_bn_5/instancenorm/sub*
T0
a
g_0_2/LeakyRelu_4/mulMulg_0_2/LeakyRelu_4/alpha!g_0_2/g_0_bn_5/instancenorm/add_1*
T0
a
g_1_2/LeakyRelu_4/mulMulg_0_2/LeakyRelu_4/alpha!g_1_2/g_1_bn_5/instancenorm/add_1*
T0
_
g_0_2/LeakyRelu_4Maximumg_0_2/LeakyRelu_4/mul!g_0_2/g_0_bn_5/instancenorm/add_1*
T0
_
g_1_2/LeakyRelu_4Maximumg_1_2/LeakyRelu_4/mul!g_1_2/g_1_bn_5/instancenorm/add_1*
T0
�
g_0_2/g_0_conv_6Conv2Dg_0_2/LeakyRelu_4g_0/g_0_conv_6w*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
g_1_2/g_1_conv_6Conv2Dg_1_2/LeakyRelu_4g_1/g_1_conv_6w*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC
b
g_0_2/BiasAdd_5BiasAddg_0_2/g_0_conv_6g_0/g_0_conv_6biases*
T0*
data_formatNHWC
b
g_1_2/BiasAdd_5BiasAddg_1_2/g_1_conv_6g_1/g_1_conv_6biases*
T0*
data_formatNHWC
/
add_10Addg_0_2/BiasAdd_5pred_0*
T0
/
add_11Addg_1_2/BiasAdd_5pred_1*
T0
$
output0Identityadd_10*
T0
$
output1Identityadd_11*
T0 " 