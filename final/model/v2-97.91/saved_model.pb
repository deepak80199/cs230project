Äú#
º
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8âª!

 Adam/lstm_71/lstm_cell_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*1
shared_name" Adam/lstm_71/lstm_cell_71/bias/v

4Adam/lstm_71/lstm_cell_71/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_71/lstm_cell_71/bias/v*
_output_shapes
:x*
dtype0
´
,Adam/lstm_71/lstm_cell_71/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*=
shared_name.,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v
­
@Adam/lstm_71/lstm_cell_71/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v*
_output_shapes

:x*
dtype0
 
"Adam/lstm_71/lstm_cell_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*3
shared_name$"Adam/lstm_71/lstm_cell_71/kernel/v

6Adam/lstm_71/lstm_cell_71/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_71/lstm_cell_71/kernel/v*
_output_shapes

:x*
dtype0

 Adam/lstm_70/lstm_cell_70/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*1
shared_name" Adam/lstm_70/lstm_cell_70/bias/v

4Adam/lstm_70/lstm_cell_70/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_70/lstm_cell_70/bias/v*
_output_shapes
:x*
dtype0
´
,Adam/lstm_70/lstm_cell_70/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*=
shared_name.,Adam/lstm_70/lstm_cell_70/recurrent_kernel/v
­
@Adam/lstm_70/lstm_cell_70/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_70/lstm_cell_70/recurrent_kernel/v*
_output_shapes

:x*
dtype0
 
"Adam/lstm_70/lstm_cell_70/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*3
shared_name$"Adam/lstm_70/lstm_cell_70/kernel/v

6Adam/lstm_70/lstm_cell_70/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_70/lstm_cell_70/kernel/v*
_output_shapes

:x*
dtype0

Adam/dense_123/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_123/bias/v
{
)Adam/dense_123/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_123/bias/v*
_output_shapes
:*
dtype0

Adam/dense_123/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_123/kernel/v

+Adam/dense_123/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_123/kernel/v*
_output_shapes

:d*
dtype0

Adam/dense_122/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_122/bias/v
{
)Adam/dense_122/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_122/bias/v*
_output_shapes
:d*
dtype0

Adam/dense_122/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_122/kernel/v

+Adam/dense_122/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_122/kernel/v*
_output_shapes

:d*
dtype0

 Adam/lstm_71/lstm_cell_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*1
shared_name" Adam/lstm_71/lstm_cell_71/bias/m

4Adam/lstm_71/lstm_cell_71/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_71/lstm_cell_71/bias/m*
_output_shapes
:x*
dtype0
´
,Adam/lstm_71/lstm_cell_71/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*=
shared_name.,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m
­
@Adam/lstm_71/lstm_cell_71/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m*
_output_shapes

:x*
dtype0
 
"Adam/lstm_71/lstm_cell_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*3
shared_name$"Adam/lstm_71/lstm_cell_71/kernel/m

6Adam/lstm_71/lstm_cell_71/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_71/lstm_cell_71/kernel/m*
_output_shapes

:x*
dtype0

 Adam/lstm_70/lstm_cell_70/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*1
shared_name" Adam/lstm_70/lstm_cell_70/bias/m

4Adam/lstm_70/lstm_cell_70/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_70/lstm_cell_70/bias/m*
_output_shapes
:x*
dtype0
´
,Adam/lstm_70/lstm_cell_70/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*=
shared_name.,Adam/lstm_70/lstm_cell_70/recurrent_kernel/m
­
@Adam/lstm_70/lstm_cell_70/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_70/lstm_cell_70/recurrent_kernel/m*
_output_shapes

:x*
dtype0
 
"Adam/lstm_70/lstm_cell_70/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*3
shared_name$"Adam/lstm_70/lstm_cell_70/kernel/m

6Adam/lstm_70/lstm_cell_70/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_70/lstm_cell_70/kernel/m*
_output_shapes

:x*
dtype0

Adam/dense_123/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_123/bias/m
{
)Adam/dense_123/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_123/bias/m*
_output_shapes
:*
dtype0

Adam/dense_123/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_123/kernel/m

+Adam/dense_123/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_123/kernel/m*
_output_shapes

:d*
dtype0

Adam/dense_122/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*&
shared_nameAdam/dense_122/bias/m
{
)Adam/dense_122/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_122/bias/m*
_output_shapes
:d*
dtype0

Adam/dense_122/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameAdam/dense_122/kernel/m

+Adam/dense_122/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_122/kernel/m*
_output_shapes

:d*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

lstm_71/lstm_cell_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x**
shared_namelstm_71/lstm_cell_71/bias

-lstm_71/lstm_cell_71/bias/Read/ReadVariableOpReadVariableOplstm_71/lstm_cell_71/bias*
_output_shapes
:x*
dtype0
¦
%lstm_71/lstm_cell_71/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*6
shared_name'%lstm_71/lstm_cell_71/recurrent_kernel

9lstm_71/lstm_cell_71/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_71/lstm_cell_71/recurrent_kernel*
_output_shapes

:x*
dtype0

lstm_71/lstm_cell_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*,
shared_namelstm_71/lstm_cell_71/kernel

/lstm_71/lstm_cell_71/kernel/Read/ReadVariableOpReadVariableOplstm_71/lstm_cell_71/kernel*
_output_shapes

:x*
dtype0

lstm_70/lstm_cell_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x**
shared_namelstm_70/lstm_cell_70/bias

-lstm_70/lstm_cell_70/bias/Read/ReadVariableOpReadVariableOplstm_70/lstm_cell_70/bias*
_output_shapes
:x*
dtype0
¦
%lstm_70/lstm_cell_70/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*6
shared_name'%lstm_70/lstm_cell_70/recurrent_kernel

9lstm_70/lstm_cell_70/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_70/lstm_cell_70/recurrent_kernel*
_output_shapes

:x*
dtype0

lstm_70/lstm_cell_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*,
shared_namelstm_70/lstm_cell_70/kernel

/lstm_70/lstm_cell_70/kernel/Read/ReadVariableOpReadVariableOplstm_70/lstm_cell_70/kernel*
_output_shapes

:x*
dtype0
t
dense_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_123/bias
m
"dense_123/bias/Read/ReadVariableOpReadVariableOpdense_123/bias*
_output_shapes
:*
dtype0
|
dense_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_123/kernel
u
$dense_123/kernel/Read/ReadVariableOpReadVariableOpdense_123/kernel*
_output_shapes

:d*
dtype0
t
dense_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_122/bias
m
"dense_122/bias/Read/ReadVariableOpReadVariableOpdense_122/bias*
_output_shapes
:d*
dtype0
|
dense_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_122/kernel
u
$dense_122/kernel/Read/ReadVariableOpReadVariableOpdense_122/kernel*
_output_shapes

:d*
dtype0

serving_default_lstm_70_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
Ï
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_70_inputlstm_70/lstm_cell_70/kernel%lstm_70/lstm_cell_70/recurrent_kernellstm_70/lstm_cell_70/biaslstm_71/lstm_cell_71/kernel%lstm_71/lstm_cell_71/recurrent_kernellstm_71/lstm_cell_71/biasdense_122/kerneldense_122/biasdense_123/kerneldense_123/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1298325

NoOpNoOp
ùV
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*´V
valueªVB§V B V

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
Á
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator
'cell
(
state_spec*
¥
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/_random_generator* 
¦
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
¦
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias*
J
@0
A1
B2
C3
D4
E5
66
77
>8
?9*
J
@0
A1
B2
C3
D4
E5
66
77
>8
?9*
* 
°
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
* 

Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_rate6mÂ7mÃ>mÄ?mÅ@mÆAmÇBmÈCmÉDmÊEmË6vÌ7vÍ>vÎ?vÏ@vÐAvÑBvÒCvÓDvÔEvÕ*

Xserving_default* 

@0
A1
B2*

@0
A1
B2*
* 


Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
_trace_0
`trace_1
atrace_2
btrace_3* 
6
ctrace_0
dtrace_1
etrace_2
ftrace_3* 
* 
ã
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
m_random_generator
n
state_size

@kernel
Arecurrent_kernel
Bbias*
* 
* 
* 
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ttrace_0
utrace_1* 

vtrace_0
wtrace_1* 
* 

C0
D1
E2*

C0
D1
E2*
* 


xstates
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
8
~trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
ë
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

state_size

Ckernel
Drecurrent_kernel
Ebias*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

60
71*

60
71*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_122/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_122/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

>0
?1*

>0
?1*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

£trace_0* 

¤trace_0* 
`Z
VARIABLE_VALUEdense_123/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_123/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_70/lstm_cell_70/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_70/lstm_cell_70/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_70/lstm_cell_70/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_71/lstm_cell_71/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_71/lstm_cell_71/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_71/lstm_cell_71/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

¥0
¦1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

@0
A1
B2*

@0
A1
B2*
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

¬trace_0
­trace_1* 

®trace_0
¯trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

'0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

C0
D1
E2*

C0
D1
E2*
* 

°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

µtrace_0
¶trace_1* 

·trace_0
¸trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
¹	variables
º	keras_api

»total

¼count*
M
½	variables
¾	keras_api

¿total

Àcount
Á
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

»0
¼1*

¹	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

¿0
À1*

½	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
}
VARIABLE_VALUEAdam/dense_122/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_122/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_123/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_123/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_70/lstm_cell_70/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_70/lstm_cell_70/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_70/lstm_cell_70/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_71/lstm_cell_71/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_71/lstm_cell_71/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_71/lstm_cell_71/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_122/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_122/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_123/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_123/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_70/lstm_cell_70/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_70/lstm_cell_70/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_70/lstm_cell_70/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_71/lstm_cell_71/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_71/lstm_cell_71/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_71/lstm_cell_71/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Æ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_122/kernel/Read/ReadVariableOp"dense_122/bias/Read/ReadVariableOp$dense_123/kernel/Read/ReadVariableOp"dense_123/bias/Read/ReadVariableOp/lstm_70/lstm_cell_70/kernel/Read/ReadVariableOp9lstm_70/lstm_cell_70/recurrent_kernel/Read/ReadVariableOp-lstm_70/lstm_cell_70/bias/Read/ReadVariableOp/lstm_71/lstm_cell_71/kernel/Read/ReadVariableOp9lstm_71/lstm_cell_71/recurrent_kernel/Read/ReadVariableOp-lstm_71/lstm_cell_71/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_122/kernel/m/Read/ReadVariableOp)Adam/dense_122/bias/m/Read/ReadVariableOp+Adam/dense_123/kernel/m/Read/ReadVariableOp)Adam/dense_123/bias/m/Read/ReadVariableOp6Adam/lstm_70/lstm_cell_70/kernel/m/Read/ReadVariableOp@Adam/lstm_70/lstm_cell_70/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_70/lstm_cell_70/bias/m/Read/ReadVariableOp6Adam/lstm_71/lstm_cell_71/kernel/m/Read/ReadVariableOp@Adam/lstm_71/lstm_cell_71/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_71/lstm_cell_71/bias/m/Read/ReadVariableOp+Adam/dense_122/kernel/v/Read/ReadVariableOp)Adam/dense_122/bias/v/Read/ReadVariableOp+Adam/dense_123/kernel/v/Read/ReadVariableOp)Adam/dense_123/bias/v/Read/ReadVariableOp6Adam/lstm_70/lstm_cell_70/kernel/v/Read/ReadVariableOp@Adam/lstm_70/lstm_cell_70/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_70/lstm_cell_70/bias/v/Read/ReadVariableOp6Adam/lstm_71/lstm_cell_71/kernel/v/Read/ReadVariableOp@Adam/lstm_71/lstm_cell_71/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_71/lstm_cell_71/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1300659
µ

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_122/kerneldense_122/biasdense_123/kerneldense_123/biaslstm_70/lstm_cell_70/kernel%lstm_70/lstm_cell_70/recurrent_kernellstm_70/lstm_cell_70/biaslstm_71/lstm_cell_71/kernel%lstm_71/lstm_cell_71/recurrent_kernellstm_71/lstm_cell_71/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_122/kernel/mAdam/dense_122/bias/mAdam/dense_123/kernel/mAdam/dense_123/bias/m"Adam/lstm_70/lstm_cell_70/kernel/m,Adam/lstm_70/lstm_cell_70/recurrent_kernel/m Adam/lstm_70/lstm_cell_70/bias/m"Adam/lstm_71/lstm_cell_71/kernel/m,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m Adam/lstm_71/lstm_cell_71/bias/mAdam/dense_122/kernel/vAdam/dense_122/bias/vAdam/dense_123/kernel/vAdam/dense_123/bias/v"Adam/lstm_70/lstm_cell_70/kernel/v,Adam/lstm_70/lstm_cell_70/recurrent_kernel/v Adam/lstm_70/lstm_cell_70/bias/v"Adam/lstm_71/lstm_cell_71/kernel/v,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v Adam/lstm_71/lstm_cell_71/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1300786§ß
J

D__inference_lstm_70_layer_call_and_return_conditional_losses_1297497

inputs=
+lstm_cell_70_matmul_readvariableop_resource:x?
-lstm_cell_70_matmul_1_readvariableop_resource:x:
,lstm_cell_70_biasadd_readvariableop_resource:x
identity¢#lstm_cell_70/BiasAdd/ReadVariableOp¢"lstm_cell_70/MatMul/ReadVariableOp¢$lstm_cell_70/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_70/MatMul/ReadVariableOpReadVariableOp+lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMulMatMulstrided_slice_2:output:0*lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMul_1MatMulzeros:output:0,lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_70/addAddV2lstm_cell_70/MatMul:product:0lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_70/BiasAddBiasAddlstm_cell_70/add:z:0+lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_70/splitSplit%lstm_cell_70/split/split_dim:output:0lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_70/SigmoidSigmoidlstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_1Sigmoidlstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_70/mulMullstm_cell_70/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_70/ReluRelulstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_1Mullstm_cell_70/Sigmoid:y:0lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_70/add_1AddV2lstm_cell_70/mul:z:0lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_2Sigmoidlstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_70/Relu_1Relulstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_2Mullstm_cell_70/Sigmoid_2:y:0!lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_70_matmul_readvariableop_resource-lstm_cell_70_matmul_1_readvariableop_resource,lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1297413*
condR
while_cond_1297412*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_70/BiasAdd/ReadVariableOp#^lstm_cell_70/MatMul/ReadVariableOp%^lstm_cell_70/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_70/BiasAdd/ReadVariableOp#lstm_cell_70/BiasAdd/ReadVariableOp2H
"lstm_cell_70/MatMul/ReadVariableOp"lstm_cell_70/MatMul/ReadVariableOp2L
$lstm_cell_70/MatMul_1/ReadVariableOp$lstm_cell_70/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1298033
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1298033___redundant_placeholder05
1while_while_cond_1298033___redundant_placeholder15
1while_while_cond_1298033___redundant_placeholder25
1while_while_cond_1298033___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
º
È
while_cond_1296915
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1296915___redundant_placeholder05
1while_while_cond_1296915___redundant_placeholder15
1while_while_cond_1296915___redundant_placeholder25
1while_while_cond_1296915___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


è
lstm_71_while_cond_1298881,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3.
*lstm_71_while_less_lstm_71_strided_slice_1E
Alstm_71_while_lstm_71_while_cond_1298881___redundant_placeholder0E
Alstm_71_while_lstm_71_while_cond_1298881___redundant_placeholder1E
Alstm_71_while_lstm_71_while_cond_1298881___redundant_placeholder2E
Alstm_71_while_lstm_71_while_cond_1298881___redundant_placeholder3
lstm_71_while_identity

lstm_71/while/LessLesslstm_71_while_placeholder*lstm_71_while_less_lstm_71_strided_slice_1*
T0*
_output_shapes
: [
lstm_71/while/IdentityIdentitylstm_71/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_71_while_identitylstm_71/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
º
È
while_cond_1300170
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1300170___redundant_placeholder05
1while_while_cond_1300170___redundant_placeholder15
1while_while_cond_1300170___redundant_placeholder25
1while_while_cond_1300170___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ì

I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1296857

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¨
µ
)__inference_lstm_70_layer_call_fn_1299011
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_70_layer_call_and_return_conditional_losses_1296985|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
9

D__inference_lstm_71_layer_call_and_return_conditional_losses_1297339

inputs&
lstm_cell_71_1297255:x&
lstm_cell_71_1297257:x"
lstm_cell_71_1297259:x
identity¢$lstm_cell_71/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskù
$lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_71_1297255lstm_cell_71_1297257lstm_cell_71_1297259*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1297209n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_71_1297255lstm_cell_71_1297257lstm_cell_71_1297259*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1297269*
condR
while_cond_1297268*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_71/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_71/StatefulPartitionedCall$lstm_cell_71/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£

(sequential_64_lstm_71_while_cond_1296543H
Dsequential_64_lstm_71_while_sequential_64_lstm_71_while_loop_counterN
Jsequential_64_lstm_71_while_sequential_64_lstm_71_while_maximum_iterations+
'sequential_64_lstm_71_while_placeholder-
)sequential_64_lstm_71_while_placeholder_1-
)sequential_64_lstm_71_while_placeholder_2-
)sequential_64_lstm_71_while_placeholder_3J
Fsequential_64_lstm_71_while_less_sequential_64_lstm_71_strided_slice_1a
]sequential_64_lstm_71_while_sequential_64_lstm_71_while_cond_1296543___redundant_placeholder0a
]sequential_64_lstm_71_while_sequential_64_lstm_71_while_cond_1296543___redundant_placeholder1a
]sequential_64_lstm_71_while_sequential_64_lstm_71_while_cond_1296543___redundant_placeholder2a
]sequential_64_lstm_71_while_sequential_64_lstm_71_while_cond_1296543___redundant_placeholder3(
$sequential_64_lstm_71_while_identity
º
 sequential_64/lstm_71/while/LessLess'sequential_64_lstm_71_while_placeholderFsequential_64_lstm_71_while_less_sequential_64_lstm_71_strided_slice_1*
T0*
_output_shapes
: w
$sequential_64/lstm_71/while/IdentityIdentity$sequential_64/lstm_71/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_64_lstm_71_while_identity-sequential_64/lstm_71/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
º
È
while_cond_1296724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1296724___redundant_placeholder05
1while_while_cond_1296724___redundant_placeholder15
1while_while_cond_1296724___redundant_placeholder25
1while_while_cond_1296724___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
º
È
while_cond_1299880
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1299880___redundant_placeholder05
1while_while_cond_1299880___redundant_placeholder15
1while_while_cond_1299880___redundant_placeholder25
1while_while_cond_1299880___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Æ

+__inference_dense_122_layer_call_fn_1300292

inputs
unknown:d
	unknown_0:d
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1297682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1297268
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1297268___redundant_placeholder05
1while_while_cond_1297268___redundant_placeholder15
1while_while_cond_1297268___redundant_placeholder25
1while_while_cond_1297268___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÚJ

D__inference_lstm_70_layer_call_and_return_conditional_losses_1299176
inputs_0=
+lstm_cell_70_matmul_readvariableop_resource:x?
-lstm_cell_70_matmul_1_readvariableop_resource:x:
,lstm_cell_70_biasadd_readvariableop_resource:x
identity¢#lstm_cell_70/BiasAdd/ReadVariableOp¢"lstm_cell_70/MatMul/ReadVariableOp¢$lstm_cell_70/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_70/MatMul/ReadVariableOpReadVariableOp+lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMulMatMulstrided_slice_2:output:0*lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMul_1MatMulzeros:output:0,lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_70/addAddV2lstm_cell_70/MatMul:product:0lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_70/BiasAddBiasAddlstm_cell_70/add:z:0+lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_70/splitSplit%lstm_cell_70/split/split_dim:output:0lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_70/SigmoidSigmoidlstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_1Sigmoidlstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_70/mulMullstm_cell_70/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_70/ReluRelulstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_1Mullstm_cell_70/Sigmoid:y:0lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_70/add_1AddV2lstm_cell_70/mul:z:0lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_2Sigmoidlstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_70/Relu_1Relulstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_2Mullstm_cell_70/Sigmoid_2:y:0!lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_70_matmul_readvariableop_resource-lstm_cell_70_matmul_1_readvariableop_resource,lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1299092*
condR
while_cond_1299091*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_70/BiasAdd/ReadVariableOp#^lstm_cell_70/MatMul/ReadVariableOp%^lstm_cell_70/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_70/BiasAdd/ReadVariableOp#lstm_cell_70/BiasAdd/ReadVariableOp2H
"lstm_cell_70/MatMul/ReadVariableOp"lstm_cell_70/MatMul/ReadVariableOp2L
$lstm_cell_70/MatMul_1/ReadVariableOp$lstm_cell_70/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


è
lstm_70_while_cond_1298433,
(lstm_70_while_lstm_70_while_loop_counter2
.lstm_70_while_lstm_70_while_maximum_iterations
lstm_70_while_placeholder
lstm_70_while_placeholder_1
lstm_70_while_placeholder_2
lstm_70_while_placeholder_3.
*lstm_70_while_less_lstm_70_strided_slice_1E
Alstm_70_while_lstm_70_while_cond_1298433___redundant_placeholder0E
Alstm_70_while_lstm_70_while_cond_1298433___redundant_placeholder1E
Alstm_70_while_lstm_70_while_cond_1298433___redundant_placeholder2E
Alstm_70_while_lstm_70_while_cond_1298433___redundant_placeholder3
lstm_70_while_identity

lstm_70/while/LessLesslstm_70_while_placeholder*lstm_70_while_less_lstm_70_strided_slice_1*
T0*
_output_shapes
: [
lstm_70/while/IdentityIdentitylstm_70/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_70_while_identitylstm_70/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


f
G__inference_dropout_68_layer_call_and_return_conditional_losses_1299632

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8
Ë
while_body_1299521
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_70_matmul_readvariableop_resource_0:xG
5while_lstm_cell_70_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_70_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_70_matmul_readvariableop_resource:xE
3while_lstm_cell_70_matmul_1_readvariableop_resource:x@
2while_lstm_cell_70_biasadd_readvariableop_resource:x¢)while/lstm_cell_70/BiasAdd/ReadVariableOp¢(while/lstm_cell_70/MatMul/ReadVariableOp¢*while/lstm_cell_70/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_70/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_70/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_70/addAddV2#while/lstm_cell_70/MatMul:product:0%while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_70/BiasAddBiasAddwhile/lstm_cell_70/add:z:01while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_70/splitSplit+while/lstm_cell_70/split/split_dim:output:0#while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_70/SigmoidSigmoid!while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_1Sigmoid!while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mulMul while/lstm_cell_70/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_70/ReluRelu!while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_1Mulwhile/lstm_cell_70/Sigmoid:y:0%while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/add_1AddV2while/lstm_cell_70/mul:z:0while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_2Sigmoid!while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_70/Relu_1Reluwhile/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_2Mul while/lstm_cell_70/Sigmoid_2:y:0'while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_70/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_70/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_70/BiasAdd/ReadVariableOp)^while/lstm_cell_70/MatMul/ReadVariableOp+^while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_70_biasadd_readvariableop_resource4while_lstm_cell_70_biasadd_readvariableop_resource_0"l
3while_lstm_cell_70_matmul_1_readvariableop_resource5while_lstm_cell_70_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_70_matmul_readvariableop_resource3while_lstm_cell_70_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_70/BiasAdd/ReadVariableOp)while/lstm_cell_70/BiasAdd/ReadVariableOp2T
(while/lstm_cell_70/MatMul/ReadVariableOp(while/lstm_cell_70/MatMul/ReadVariableOp2X
*while/lstm_cell_70/MatMul_1/ReadVariableOp*while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô

I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1300487

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
º
È
while_cond_1300025
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1300025___redundant_placeholder05
1while_while_cond_1300025___redundant_placeholder15
1while_while_cond_1300025___redundant_placeholder25
1while_while_cond_1300025___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
£

(sequential_64_lstm_70_while_cond_1296402H
Dsequential_64_lstm_70_while_sequential_64_lstm_70_while_loop_counterN
Jsequential_64_lstm_70_while_sequential_64_lstm_70_while_maximum_iterations+
'sequential_64_lstm_70_while_placeholder-
)sequential_64_lstm_70_while_placeholder_1-
)sequential_64_lstm_70_while_placeholder_2-
)sequential_64_lstm_70_while_placeholder_3J
Fsequential_64_lstm_70_while_less_sequential_64_lstm_70_strided_slice_1a
]sequential_64_lstm_70_while_sequential_64_lstm_70_while_cond_1296402___redundant_placeholder0a
]sequential_64_lstm_70_while_sequential_64_lstm_70_while_cond_1296402___redundant_placeholder1a
]sequential_64_lstm_70_while_sequential_64_lstm_70_while_cond_1296402___redundant_placeholder2a
]sequential_64_lstm_70_while_sequential_64_lstm_70_while_cond_1296402___redundant_placeholder3(
$sequential_64_lstm_70_while_identity
º
 sequential_64/lstm_70/while/LessLess'sequential_64_lstm_70_while_placeholderFsequential_64_lstm_70_while_less_sequential_64_lstm_70_strided_slice_1*
T0*
_output_shapes
: w
$sequential_64/lstm_70/while/IdentityIdentity$sequential_64/lstm_70/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_64_lstm_70_while_identity-sequential_64/lstm_70/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ì

I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1296711

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¢K

D__inference_lstm_71_layer_call_and_return_conditional_losses_1300111

inputs=
+lstm_cell_71_matmul_readvariableop_resource:x?
-lstm_cell_71_matmul_1_readvariableop_resource:x:
,lstm_cell_71_biasadd_readvariableop_resource:x
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1300026*
condR
while_cond_1300025*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
e
G__inference_dropout_68_layer_call_and_return_conditional_losses_1299620

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹ 

J__inference_sequential_64_layer_call_and_return_conditional_losses_1298184

inputs!
lstm_70_1298157:x!
lstm_70_1298159:x
lstm_70_1298161:x!
lstm_71_1298165:x!
lstm_71_1298167:x
lstm_71_1298169:x#
dense_122_1298173:d
dense_122_1298175:d#
dense_123_1298178:d
dense_123_1298180:
identity¢!dense_122/StatefulPartitionedCall¢!dense_123/StatefulPartitionedCall¢"dropout_68/StatefulPartitionedCall¢"dropout_69/StatefulPartitionedCall¢lstm_70/StatefulPartitionedCall¢lstm_71/StatefulPartitionedCall
lstm_70/StatefulPartitionedCallStatefulPartitionedCallinputslstm_70_1298157lstm_70_1298159lstm_70_1298161*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_70_layer_call_and_return_conditional_losses_1298118ó
"dropout_68/StatefulPartitionedCallStatefulPartitionedCall(lstm_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_68_layer_call_and_return_conditional_losses_1297959§
lstm_71/StatefulPartitionedCallStatefulPartitionedCall+dropout_68/StatefulPartitionedCall:output:0lstm_71_1298165lstm_71_1298167lstm_71_1298169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_1297930
"dropout_69/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0#^dropout_68/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_69_layer_call_and_return_conditional_losses_1297769
!dense_122/StatefulPartitionedCallStatefulPartitionedCall+dropout_69/StatefulPartitionedCall:output:0dense_122_1298173dense_122_1298175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1297682
!dense_123/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0dense_123_1298178dense_123_1298180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_1297699y
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^dense_122/StatefulPartitionedCall"^dense_123/StatefulPartitionedCall#^dropout_68/StatefulPartitionedCall#^dropout_69/StatefulPartitionedCall ^lstm_70/StatefulPartitionedCall ^lstm_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2H
"dropout_68/StatefulPartitionedCall"dropout_68/StatefulPartitionedCall2H
"dropout_69/StatefulPartitionedCall"dropout_69/StatefulPartitionedCall2B
lstm_70/StatefulPartitionedCalllstm_70/StatefulPartitionedCall2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1300519

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¢K

D__inference_lstm_71_layer_call_and_return_conditional_losses_1297930

inputs=
+lstm_cell_71_matmul_readvariableop_resource:x?
-lstm_cell_71_matmul_1_readvariableop_resource:x:
,lstm_cell_71_biasadd_readvariableop_resource:x
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1297845*
condR
while_cond_1297844*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢K

D__inference_lstm_71_layer_call_and_return_conditional_losses_1300256

inputs=
+lstm_cell_71_matmul_readvariableop_resource:x?
-lstm_cell_71_matmul_1_readvariableop_resource:x:
,lstm_cell_71_biasadd_readvariableop_resource:x
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1300171*
condR
while_cond_1300170*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


õ
%__inference_signature_wrapper_1298325
lstm_70_input
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:d
	unknown_6:d
	unknown_7:d
	unknown_8:
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCalllstm_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1296644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_70_input
ê
e
G__inference_dropout_68_layer_call_and_return_conditional_losses_1297510

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8

D__inference_lstm_70_layer_call_and_return_conditional_losses_1296985

inputs&
lstm_cell_70_1296903:x&
lstm_cell_70_1296905:x"
lstm_cell_70_1296907:x
identity¢$lstm_cell_70/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskù
$lstm_cell_70/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_70_1296903lstm_cell_70_1296905lstm_cell_70_1296907*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1296857n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_70_1296903lstm_cell_70_1296905lstm_cell_70_1296907*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1296916*
condR
while_cond_1296915*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_70/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_70/StatefulPartitionedCall$lstm_cell_70/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1299735
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1299735___redundant_placeholder05
1while_while_cond_1299735___redundant_placeholder15
1while_while_cond_1299735___redundant_placeholder25
1while_while_cond_1299735___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ì

I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1297061

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
õ	
f
G__inference_dropout_69_layer_call_and_return_conditional_losses_1297769

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ"
ä
while_body_1296725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_70_1296749_0:x.
while_lstm_cell_70_1296751_0:x*
while_lstm_cell_70_1296753_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_70_1296749:x,
while_lstm_cell_70_1296751:x(
while_lstm_cell_70_1296753:x¢*while/lstm_cell_70/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0·
*while/lstm_cell_70/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_70_1296749_0while_lstm_cell_70_1296751_0while_lstm_cell_70_1296753_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1296711Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_70/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_70/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_70/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_70/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_70_1296749while_lstm_cell_70_1296749_0":
while_lstm_cell_70_1296751while_lstm_cell_70_1296751_0":
while_lstm_cell_70_1296753while_lstm_cell_70_1296753_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_70/StatefulPartitionedCall*while/lstm_cell_70/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÿß
Ì
"__inference__wrapped_model_1296644
lstm_70_inputS
Asequential_64_lstm_70_lstm_cell_70_matmul_readvariableop_resource:xU
Csequential_64_lstm_70_lstm_cell_70_matmul_1_readvariableop_resource:xP
Bsequential_64_lstm_70_lstm_cell_70_biasadd_readvariableop_resource:xS
Asequential_64_lstm_71_lstm_cell_71_matmul_readvariableop_resource:xU
Csequential_64_lstm_71_lstm_cell_71_matmul_1_readvariableop_resource:xP
Bsequential_64_lstm_71_lstm_cell_71_biasadd_readvariableop_resource:xH
6sequential_64_dense_122_matmul_readvariableop_resource:dE
7sequential_64_dense_122_biasadd_readvariableop_resource:dH
6sequential_64_dense_123_matmul_readvariableop_resource:dE
7sequential_64_dense_123_biasadd_readvariableop_resource:
identity¢.sequential_64/dense_122/BiasAdd/ReadVariableOp¢-sequential_64/dense_122/MatMul/ReadVariableOp¢.sequential_64/dense_123/BiasAdd/ReadVariableOp¢-sequential_64/dense_123/MatMul/ReadVariableOp¢9sequential_64/lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp¢8sequential_64/lstm_70/lstm_cell_70/MatMul/ReadVariableOp¢:sequential_64/lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp¢sequential_64/lstm_70/while¢9sequential_64/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp¢8sequential_64/lstm_71/lstm_cell_71/MatMul/ReadVariableOp¢:sequential_64/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp¢sequential_64/lstm_71/whileX
sequential_64/lstm_70/ShapeShapelstm_70_input*
T0*
_output_shapes
:s
)sequential_64/lstm_70/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_64/lstm_70/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_64/lstm_70/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_64/lstm_70/strided_sliceStridedSlice$sequential_64/lstm_70/Shape:output:02sequential_64/lstm_70/strided_slice/stack:output:04sequential_64/lstm_70/strided_slice/stack_1:output:04sequential_64/lstm_70/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_64/lstm_70/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :µ
"sequential_64/lstm_70/zeros/packedPack,sequential_64/lstm_70/strided_slice:output:0-sequential_64/lstm_70/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_64/lstm_70/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_64/lstm_70/zerosFill+sequential_64/lstm_70/zeros/packed:output:0*sequential_64/lstm_70/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&sequential_64/lstm_70/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¹
$sequential_64/lstm_70/zeros_1/packedPack,sequential_64/lstm_70/strided_slice:output:0/sequential_64/lstm_70/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_64/lstm_70/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
sequential_64/lstm_70/zeros_1Fill-sequential_64/lstm_70/zeros_1/packed:output:0,sequential_64/lstm_70/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$sequential_64/lstm_70/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"           
sequential_64/lstm_70/transpose	Transposelstm_70_input-sequential_64/lstm_70/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
sequential_64/lstm_70/Shape_1Shape#sequential_64/lstm_70/transpose:y:0*
T0*
_output_shapes
:u
+sequential_64/lstm_70/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_64/lstm_70/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_64/lstm_70/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%sequential_64/lstm_70/strided_slice_1StridedSlice&sequential_64/lstm_70/Shape_1:output:04sequential_64/lstm_70/strided_slice_1/stack:output:06sequential_64/lstm_70/strided_slice_1/stack_1:output:06sequential_64/lstm_70/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_64/lstm_70/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#sequential_64/lstm_70/TensorArrayV2TensorListReserve:sequential_64/lstm_70/TensorArrayV2/element_shape:output:0.sequential_64/lstm_70/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Ksequential_64/lstm_70/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¢
=sequential_64/lstm_70/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_64/lstm_70/transpose:y:0Tsequential_64/lstm_70/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+sequential_64/lstm_70/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_64/lstm_70/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_64/lstm_70/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%sequential_64/lstm_70/strided_slice_2StridedSlice#sequential_64/lstm_70/transpose:y:04sequential_64/lstm_70/strided_slice_2/stack:output:06sequential_64/lstm_70/strided_slice_2/stack_1:output:06sequential_64/lstm_70/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskº
8sequential_64/lstm_70/lstm_cell_70/MatMul/ReadVariableOpReadVariableOpAsequential_64_lstm_70_lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0×
)sequential_64/lstm_70/lstm_cell_70/MatMulMatMul.sequential_64/lstm_70/strided_slice_2:output:0@sequential_64/lstm_70/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¾
:sequential_64/lstm_70/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOpCsequential_64_lstm_70_lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0Ñ
+sequential_64/lstm_70/lstm_cell_70/MatMul_1MatMul$sequential_64/lstm_70/zeros:output:0Bsequential_64/lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÍ
&sequential_64/lstm_70/lstm_cell_70/addAddV23sequential_64/lstm_70/lstm_cell_70/MatMul:product:05sequential_64/lstm_70/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¸
9sequential_64/lstm_70/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOpBsequential_64_lstm_70_lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0Ö
*sequential_64/lstm_70/lstm_cell_70/BiasAddBiasAdd*sequential_64/lstm_70/lstm_cell_70/add:z:0Asequential_64/lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxt
2sequential_64/lstm_70/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
(sequential_64/lstm_70/lstm_cell_70/splitSplit;sequential_64/lstm_70/lstm_cell_70/split/split_dim:output:03sequential_64/lstm_70/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
*sequential_64/lstm_70/lstm_cell_70/SigmoidSigmoid1sequential_64/lstm_70/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_64/lstm_70/lstm_cell_70/Sigmoid_1Sigmoid1sequential_64/lstm_70/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
&sequential_64/lstm_70/lstm_cell_70/mulMul0sequential_64/lstm_70/lstm_cell_70/Sigmoid_1:y:0&sequential_64/lstm_70/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential_64/lstm_70/lstm_cell_70/ReluRelu1sequential_64/lstm_70/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(sequential_64/lstm_70/lstm_cell_70/mul_1Mul.sequential_64/lstm_70/lstm_cell_70/Sigmoid:y:05sequential_64/lstm_70/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
(sequential_64/lstm_70/lstm_cell_70/add_1AddV2*sequential_64/lstm_70/lstm_cell_70/mul:z:0,sequential_64/lstm_70/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_64/lstm_70/lstm_cell_70/Sigmoid_2Sigmoid1sequential_64/lstm_70/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_64/lstm_70/lstm_cell_70/Relu_1Relu,sequential_64/lstm_70/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
(sequential_64/lstm_70/lstm_cell_70/mul_2Mul0sequential_64/lstm_70/lstm_cell_70/Sigmoid_2:y:07sequential_64/lstm_70/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3sequential_64/lstm_70/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ú
%sequential_64/lstm_70/TensorArrayV2_1TensorListReserve<sequential_64/lstm_70/TensorArrayV2_1/element_shape:output:0.sequential_64/lstm_70/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
sequential_64/lstm_70/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_64/lstm_70/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(sequential_64/lstm_70/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
sequential_64/lstm_70/whileWhile1sequential_64/lstm_70/while/loop_counter:output:07sequential_64/lstm_70/while/maximum_iterations:output:0#sequential_64/lstm_70/time:output:0.sequential_64/lstm_70/TensorArrayV2_1:handle:0$sequential_64/lstm_70/zeros:output:0&sequential_64/lstm_70/zeros_1:output:0.sequential_64/lstm_70/strided_slice_1:output:0Msequential_64/lstm_70/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_64_lstm_70_lstm_cell_70_matmul_readvariableop_resourceCsequential_64_lstm_70_lstm_cell_70_matmul_1_readvariableop_resourceBsequential_64_lstm_70_lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_64_lstm_70_while_body_1296403*4
cond,R*
(sequential_64_lstm_70_while_cond_1296402*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Fsequential_64/lstm_70/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
8sequential_64/lstm_70/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_64/lstm_70/while:output:3Osequential_64/lstm_70/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0~
+sequential_64/lstm_70/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-sequential_64/lstm_70/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_64/lstm_70/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%sequential_64/lstm_70/strided_slice_3StridedSliceAsequential_64/lstm_70/TensorArrayV2Stack/TensorListStack:tensor:04sequential_64/lstm_70/strided_slice_3/stack:output:06sequential_64/lstm_70/strided_slice_3/stack_1:output:06sequential_64/lstm_70/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask{
&sequential_64/lstm_70/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!sequential_64/lstm_70/transpose_1	TransposeAsequential_64/lstm_70/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_64/lstm_70/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
sequential_64/lstm_70/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
!sequential_64/dropout_68/IdentityIdentity%sequential_64/lstm_70/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
sequential_64/lstm_71/ShapeShape*sequential_64/dropout_68/Identity:output:0*
T0*
_output_shapes
:s
)sequential_64/lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_64/lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_64/lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_64/lstm_71/strided_sliceStridedSlice$sequential_64/lstm_71/Shape:output:02sequential_64/lstm_71/strided_slice/stack:output:04sequential_64/lstm_71/strided_slice/stack_1:output:04sequential_64/lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_64/lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :µ
"sequential_64/lstm_71/zeros/packedPack,sequential_64/lstm_71/strided_slice:output:0-sequential_64/lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_64/lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_64/lstm_71/zerosFill+sequential_64/lstm_71/zeros/packed:output:0*sequential_64/lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&sequential_64/lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¹
$sequential_64/lstm_71/zeros_1/packedPack,sequential_64/lstm_71/strided_slice:output:0/sequential_64/lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_64/lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
sequential_64/lstm_71/zeros_1Fill-sequential_64/lstm_71/zeros_1/packed:output:0,sequential_64/lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$sequential_64/lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ½
sequential_64/lstm_71/transpose	Transpose*sequential_64/dropout_68/Identity:output:0-sequential_64/lstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
sequential_64/lstm_71/Shape_1Shape#sequential_64/lstm_71/transpose:y:0*
T0*
_output_shapes
:u
+sequential_64/lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_64/lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_64/lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%sequential_64/lstm_71/strided_slice_1StridedSlice&sequential_64/lstm_71/Shape_1:output:04sequential_64/lstm_71/strided_slice_1/stack:output:06sequential_64/lstm_71/strided_slice_1/stack_1:output:06sequential_64/lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_64/lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#sequential_64/lstm_71/TensorArrayV2TensorListReserve:sequential_64/lstm_71/TensorArrayV2/element_shape:output:0.sequential_64/lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Ksequential_64/lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¢
=sequential_64/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_64/lstm_71/transpose:y:0Tsequential_64/lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+sequential_64/lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_64/lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_64/lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%sequential_64/lstm_71/strided_slice_2StridedSlice#sequential_64/lstm_71/transpose:y:04sequential_64/lstm_71/strided_slice_2/stack:output:06sequential_64/lstm_71/strided_slice_2/stack_1:output:06sequential_64/lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskº
8sequential_64/lstm_71/lstm_cell_71/MatMul/ReadVariableOpReadVariableOpAsequential_64_lstm_71_lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0×
)sequential_64/lstm_71/lstm_cell_71/MatMulMatMul.sequential_64/lstm_71/strided_slice_2:output:0@sequential_64/lstm_71/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¾
:sequential_64/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOpCsequential_64_lstm_71_lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0Ñ
+sequential_64/lstm_71/lstm_cell_71/MatMul_1MatMul$sequential_64/lstm_71/zeros:output:0Bsequential_64/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÍ
&sequential_64/lstm_71/lstm_cell_71/addAddV23sequential_64/lstm_71/lstm_cell_71/MatMul:product:05sequential_64/lstm_71/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¸
9sequential_64/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOpBsequential_64_lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0Ö
*sequential_64/lstm_71/lstm_cell_71/BiasAddBiasAdd*sequential_64/lstm_71/lstm_cell_71/add:z:0Asequential_64/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxt
2sequential_64/lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
(sequential_64/lstm_71/lstm_cell_71/splitSplit;sequential_64/lstm_71/lstm_cell_71/split/split_dim:output:03sequential_64/lstm_71/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
*sequential_64/lstm_71/lstm_cell_71/SigmoidSigmoid1sequential_64/lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_64/lstm_71/lstm_cell_71/Sigmoid_1Sigmoid1sequential_64/lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
&sequential_64/lstm_71/lstm_cell_71/mulMul0sequential_64/lstm_71/lstm_cell_71/Sigmoid_1:y:0&sequential_64/lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential_64/lstm_71/lstm_cell_71/ReluRelu1sequential_64/lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(sequential_64/lstm_71/lstm_cell_71/mul_1Mul.sequential_64/lstm_71/lstm_cell_71/Sigmoid:y:05sequential_64/lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
(sequential_64/lstm_71/lstm_cell_71/add_1AddV2*sequential_64/lstm_71/lstm_cell_71/mul:z:0,sequential_64/lstm_71/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_64/lstm_71/lstm_cell_71/Sigmoid_2Sigmoid1sequential_64/lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_64/lstm_71/lstm_cell_71/Relu_1Relu,sequential_64/lstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
(sequential_64/lstm_71/lstm_cell_71/mul_2Mul0sequential_64/lstm_71/lstm_cell_71/Sigmoid_2:y:07sequential_64/lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3sequential_64/lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   t
2sequential_64/lstm_71/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_64/lstm_71/TensorArrayV2_1TensorListReserve<sequential_64/lstm_71/TensorArrayV2_1/element_shape:output:0;sequential_64/lstm_71/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
sequential_64/lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_64/lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(sequential_64/lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
sequential_64/lstm_71/whileWhile1sequential_64/lstm_71/while/loop_counter:output:07sequential_64/lstm_71/while/maximum_iterations:output:0#sequential_64/lstm_71/time:output:0.sequential_64/lstm_71/TensorArrayV2_1:handle:0$sequential_64/lstm_71/zeros:output:0&sequential_64/lstm_71/zeros_1:output:0.sequential_64/lstm_71/strided_slice_1:output:0Msequential_64/lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_64_lstm_71_lstm_cell_71_matmul_readvariableop_resourceCsequential_64_lstm_71_lstm_cell_71_matmul_1_readvariableop_resourceBsequential_64_lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_64_lstm_71_while_body_1296544*4
cond,R*
(sequential_64_lstm_71_while_cond_1296543*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Fsequential_64/lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
8sequential_64/lstm_71/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_64/lstm_71/while:output:3Osequential_64/lstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elements~
+sequential_64/lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-sequential_64/lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_64/lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%sequential_64/lstm_71/strided_slice_3StridedSliceAsequential_64/lstm_71/TensorArrayV2Stack/TensorListStack:tensor:04sequential_64/lstm_71/strided_slice_3/stack:output:06sequential_64/lstm_71/strided_slice_3/stack_1:output:06sequential_64/lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask{
&sequential_64/lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!sequential_64/lstm_71/transpose_1	TransposeAsequential_64/lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_64/lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
sequential_64/lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
!sequential_64/dropout_69/IdentityIdentity.sequential_64/lstm_71/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_64/dense_122/MatMul/ReadVariableOpReadVariableOp6sequential_64_dense_122_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0½
sequential_64/dense_122/MatMulMatMul*sequential_64/dropout_69/Identity:output:05sequential_64/dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
.sequential_64/dense_122/BiasAdd/ReadVariableOpReadVariableOp7sequential_64_dense_122_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0¾
sequential_64/dense_122/BiasAddBiasAdd(sequential_64/dense_122/MatMul:product:06sequential_64/dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential_64/dense_122/ReluRelu(sequential_64/dense_122/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¤
-sequential_64/dense_123/MatMul/ReadVariableOpReadVariableOp6sequential_64_dense_123_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0½
sequential_64/dense_123/MatMulMatMul*sequential_64/dense_122/Relu:activations:05sequential_64/dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_64/dense_123/BiasAdd/ReadVariableOpReadVariableOp7sequential_64_dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_64/dense_123/BiasAddBiasAdd(sequential_64/dense_123/MatMul:product:06sequential_64/dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_64/dense_123/SoftmaxSoftmax(sequential_64/dense_123/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity)sequential_64/dense_123/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp/^sequential_64/dense_122/BiasAdd/ReadVariableOp.^sequential_64/dense_122/MatMul/ReadVariableOp/^sequential_64/dense_123/BiasAdd/ReadVariableOp.^sequential_64/dense_123/MatMul/ReadVariableOp:^sequential_64/lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp9^sequential_64/lstm_70/lstm_cell_70/MatMul/ReadVariableOp;^sequential_64/lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp^sequential_64/lstm_70/while:^sequential_64/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp9^sequential_64/lstm_71/lstm_cell_71/MatMul/ReadVariableOp;^sequential_64/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp^sequential_64/lstm_71/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2`
.sequential_64/dense_122/BiasAdd/ReadVariableOp.sequential_64/dense_122/BiasAdd/ReadVariableOp2^
-sequential_64/dense_122/MatMul/ReadVariableOp-sequential_64/dense_122/MatMul/ReadVariableOp2`
.sequential_64/dense_123/BiasAdd/ReadVariableOp.sequential_64/dense_123/BiasAdd/ReadVariableOp2^
-sequential_64/dense_123/MatMul/ReadVariableOp-sequential_64/dense_123/MatMul/ReadVariableOp2v
9sequential_64/lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp9sequential_64/lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp2t
8sequential_64/lstm_70/lstm_cell_70/MatMul/ReadVariableOp8sequential_64/lstm_70/lstm_cell_70/MatMul/ReadVariableOp2x
:sequential_64/lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp:sequential_64/lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp2:
sequential_64/lstm_70/whilesequential_64/lstm_70/while2v
9sequential_64/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp9sequential_64/lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp2t
8sequential_64/lstm_71/lstm_cell_71/MatMul/ReadVariableOp8sequential_64/lstm_71/lstm_cell_71/MatMul/ReadVariableOp2x
:sequential_64/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp:sequential_64/lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp2:
sequential_64/lstm_71/whilesequential_64/lstm_71/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_70_input
Ô

I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1300421

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1


f
G__inference_dropout_68_layer_call_and_return_conditional_losses_1297959

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
H
,__inference_dropout_68_layer_call_fn_1299610

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_68_layer_call_and_return_conditional_losses_1297510d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
J

D__inference_lstm_70_layer_call_and_return_conditional_losses_1299462

inputs=
+lstm_cell_70_matmul_readvariableop_resource:x?
-lstm_cell_70_matmul_1_readvariableop_resource:x:
,lstm_cell_70_biasadd_readvariableop_resource:x
identity¢#lstm_cell_70/BiasAdd/ReadVariableOp¢"lstm_cell_70/MatMul/ReadVariableOp¢$lstm_cell_70/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_70/MatMul/ReadVariableOpReadVariableOp+lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMulMatMulstrided_slice_2:output:0*lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMul_1MatMulzeros:output:0,lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_70/addAddV2lstm_cell_70/MatMul:product:0lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_70/BiasAddBiasAddlstm_cell_70/add:z:0+lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_70/splitSplit%lstm_cell_70/split/split_dim:output:0lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_70/SigmoidSigmoidlstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_1Sigmoidlstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_70/mulMullstm_cell_70/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_70/ReluRelulstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_1Mullstm_cell_70/Sigmoid:y:0lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_70/add_1AddV2lstm_cell_70/mul:z:0lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_2Sigmoidlstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_70/Relu_1Relulstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_2Mullstm_cell_70/Sigmoid_2:y:0!lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_70_matmul_readvariableop_resource-lstm_cell_70_matmul_1_readvariableop_resource,lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1299378*
condR
while_cond_1299377*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_70/BiasAdd/ReadVariableOp#^lstm_cell_70/MatMul/ReadVariableOp%^lstm_cell_70/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_70/BiasAdd/ReadVariableOp#lstm_cell_70/BiasAdd/ReadVariableOp2H
"lstm_cell_70/MatMul/ReadVariableOp"lstm_cell_70/MatMul/ReadVariableOp2L
$lstm_cell_70/MatMul_1/ReadVariableOp$lstm_cell_70/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬S

(sequential_64_lstm_71_while_body_1296544H
Dsequential_64_lstm_71_while_sequential_64_lstm_71_while_loop_counterN
Jsequential_64_lstm_71_while_sequential_64_lstm_71_while_maximum_iterations+
'sequential_64_lstm_71_while_placeholder-
)sequential_64_lstm_71_while_placeholder_1-
)sequential_64_lstm_71_while_placeholder_2-
)sequential_64_lstm_71_while_placeholder_3G
Csequential_64_lstm_71_while_sequential_64_lstm_71_strided_slice_1_0
sequential_64_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_71_tensorarrayunstack_tensorlistfromtensor_0[
Isequential_64_lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0:x]
Ksequential_64_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0:xX
Jsequential_64_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0:x(
$sequential_64_lstm_71_while_identity*
&sequential_64_lstm_71_while_identity_1*
&sequential_64_lstm_71_while_identity_2*
&sequential_64_lstm_71_while_identity_3*
&sequential_64_lstm_71_while_identity_4*
&sequential_64_lstm_71_while_identity_5E
Asequential_64_lstm_71_while_sequential_64_lstm_71_strided_slice_1
}sequential_64_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_71_tensorarrayunstack_tensorlistfromtensorY
Gsequential_64_lstm_71_while_lstm_cell_71_matmul_readvariableop_resource:x[
Isequential_64_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource:xV
Hsequential_64_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource:x¢?sequential_64/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp¢>sequential_64/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp¢@sequential_64/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp
Msequential_64/lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
?sequential_64/lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_64_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_71_tensorarrayunstack_tensorlistfromtensor_0'sequential_64_lstm_71_while_placeholderVsequential_64/lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0È
>sequential_64/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOpIsequential_64_lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0û
/sequential_64/lstm_71/while/lstm_cell_71/MatMulMatMulFsequential_64/lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_64/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÌ
@sequential_64/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOpKsequential_64_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0â
1sequential_64/lstm_71/while/lstm_cell_71/MatMul_1MatMul)sequential_64_lstm_71_while_placeholder_2Hsequential_64/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxß
,sequential_64/lstm_71/while/lstm_cell_71/addAddV29sequential_64/lstm_71/while/lstm_cell_71/MatMul:product:0;sequential_64/lstm_71/while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÆ
?sequential_64/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOpJsequential_64_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0è
0sequential_64/lstm_71/while/lstm_cell_71/BiasAddBiasAdd0sequential_64/lstm_71/while/lstm_cell_71/add:z:0Gsequential_64/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxz
8sequential_64/lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :±
.sequential_64/lstm_71/while/lstm_cell_71/splitSplitAsequential_64/lstm_71/while/lstm_cell_71/split/split_dim:output:09sequential_64/lstm_71/while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¦
0sequential_64/lstm_71/while/lstm_cell_71/SigmoidSigmoid7sequential_64/lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
2sequential_64/lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid7sequential_64/lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
,sequential_64/lstm_71/while/lstm_cell_71/mulMul6sequential_64/lstm_71/while/lstm_cell_71/Sigmoid_1:y:0)sequential_64_lstm_71_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_64/lstm_71/while/lstm_cell_71/ReluRelu7sequential_64/lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
.sequential_64/lstm_71/while/lstm_cell_71/mul_1Mul4sequential_64/lstm_71/while/lstm_cell_71/Sigmoid:y:0;sequential_64/lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
.sequential_64/lstm_71/while/lstm_cell_71/add_1AddV20sequential_64/lstm_71/while/lstm_cell_71/mul:z:02sequential_64/lstm_71/while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
2sequential_64/lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid7sequential_64/lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/sequential_64/lstm_71/while/lstm_cell_71/Relu_1Relu2sequential_64/lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
.sequential_64/lstm_71/while/lstm_cell_71/mul_2Mul6sequential_64/lstm_71/while/lstm_cell_71/Sigmoid_2:y:0=sequential_64/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential_64/lstm_71/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Å
@sequential_64/lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_64_lstm_71_while_placeholder_1Osequential_64/lstm_71/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_64/lstm_71/while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒc
!sequential_64/lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_64/lstm_71/while/addAddV2'sequential_64_lstm_71_while_placeholder*sequential_64/lstm_71/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_64/lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!sequential_64/lstm_71/while/add_1AddV2Dsequential_64_lstm_71_while_sequential_64_lstm_71_while_loop_counter,sequential_64/lstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: 
$sequential_64/lstm_71/while/IdentityIdentity%sequential_64/lstm_71/while/add_1:z:0!^sequential_64/lstm_71/while/NoOp*
T0*
_output_shapes
: Â
&sequential_64/lstm_71/while/Identity_1IdentityJsequential_64_lstm_71_while_sequential_64_lstm_71_while_maximum_iterations!^sequential_64/lstm_71/while/NoOp*
T0*
_output_shapes
: 
&sequential_64/lstm_71/while/Identity_2Identity#sequential_64/lstm_71/while/add:z:0!^sequential_64/lstm_71/while/NoOp*
T0*
_output_shapes
: È
&sequential_64/lstm_71/while/Identity_3IdentityPsequential_64/lstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_64/lstm_71/while/NoOp*
T0*
_output_shapes
: »
&sequential_64/lstm_71/while/Identity_4Identity2sequential_64/lstm_71/while/lstm_cell_71/mul_2:z:0!^sequential_64/lstm_71/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
&sequential_64/lstm_71/while/Identity_5Identity2sequential_64/lstm_71/while/lstm_cell_71/add_1:z:0!^sequential_64/lstm_71/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 sequential_64/lstm_71/while/NoOpNoOp@^sequential_64/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?^sequential_64/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpA^sequential_64/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_64_lstm_71_while_identity-sequential_64/lstm_71/while/Identity:output:0"Y
&sequential_64_lstm_71_while_identity_1/sequential_64/lstm_71/while/Identity_1:output:0"Y
&sequential_64_lstm_71_while_identity_2/sequential_64/lstm_71/while/Identity_2:output:0"Y
&sequential_64_lstm_71_while_identity_3/sequential_64/lstm_71/while/Identity_3:output:0"Y
&sequential_64_lstm_71_while_identity_4/sequential_64/lstm_71/while/Identity_4:output:0"Y
&sequential_64_lstm_71_while_identity_5/sequential_64/lstm_71/while/Identity_5:output:0"
Hsequential_64_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resourceJsequential_64_lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0"
Isequential_64_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resourceKsequential_64_lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0"
Gsequential_64_lstm_71_while_lstm_cell_71_matmul_readvariableop_resourceIsequential_64_lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0"
Asequential_64_lstm_71_while_sequential_64_lstm_71_strided_slice_1Csequential_64_lstm_71_while_sequential_64_lstm_71_strided_slice_1_0"
}sequential_64_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_71_tensorarrayunstack_tensorlistfromtensorsequential_64_lstm_71_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
?sequential_64/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp?sequential_64/lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp2
>sequential_64/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp>sequential_64/lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp2
@sequential_64/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp@sequential_64/lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ê
ô
.__inference_lstm_cell_71_layer_call_fn_1300438

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1297061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ö
³
)__inference_lstm_71_layer_call_fn_1299665

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_1297656o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Ò
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298262
lstm_70_input!
lstm_70_1298235:x!
lstm_70_1298237:x
lstm_70_1298239:x!
lstm_71_1298243:x!
lstm_71_1298245:x
lstm_71_1298247:x#
dense_122_1298251:d
dense_122_1298253:d#
dense_123_1298256:d
dense_123_1298258:
identity¢!dense_122/StatefulPartitionedCall¢!dense_123/StatefulPartitionedCall¢lstm_70/StatefulPartitionedCall¢lstm_71/StatefulPartitionedCall
lstm_70/StatefulPartitionedCallStatefulPartitionedCalllstm_70_inputlstm_70_1298235lstm_70_1298237lstm_70_1298239*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_70_layer_call_and_return_conditional_losses_1297497ã
dropout_68/PartitionedCallPartitionedCall(lstm_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_68_layer_call_and_return_conditional_losses_1297510
lstm_71/StatefulPartitionedCallStatefulPartitionedCall#dropout_68/PartitionedCall:output:0lstm_71_1298243lstm_71_1298245lstm_71_1298247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_1297656ß
dropout_69/PartitionedCallPartitionedCall(lstm_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_69_layer_call_and_return_conditional_losses_1297669
!dense_122/StatefulPartitionedCallStatefulPartitionedCall#dropout_69/PartitionedCall:output:0dense_122_1298251dense_122_1298253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1297682
!dense_123/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0dense_123_1298256dense_123_1298258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_1297699y
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp"^dense_122/StatefulPartitionedCall"^dense_123/StatefulPartitionedCall ^lstm_70/StatefulPartitionedCall ^lstm_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2B
lstm_70/StatefulPartitionedCalllstm_70/StatefulPartitionedCall2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_70_input
S

 __inference__traced_save_1300659
file_prefix/
+savev2_dense_122_kernel_read_readvariableop-
)savev2_dense_122_bias_read_readvariableop/
+savev2_dense_123_kernel_read_readvariableop-
)savev2_dense_123_bias_read_readvariableop:
6savev2_lstm_70_lstm_cell_70_kernel_read_readvariableopD
@savev2_lstm_70_lstm_cell_70_recurrent_kernel_read_readvariableop8
4savev2_lstm_70_lstm_cell_70_bias_read_readvariableop:
6savev2_lstm_71_lstm_cell_71_kernel_read_readvariableopD
@savev2_lstm_71_lstm_cell_71_recurrent_kernel_read_readvariableop8
4savev2_lstm_71_lstm_cell_71_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_122_kernel_m_read_readvariableop4
0savev2_adam_dense_122_bias_m_read_readvariableop6
2savev2_adam_dense_123_kernel_m_read_readvariableop4
0savev2_adam_dense_123_bias_m_read_readvariableopA
=savev2_adam_lstm_70_lstm_cell_70_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_70_lstm_cell_70_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_70_lstm_cell_70_bias_m_read_readvariableopA
=savev2_adam_lstm_71_lstm_cell_71_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_71_lstm_cell_71_bias_m_read_readvariableop6
2savev2_adam_dense_122_kernel_v_read_readvariableop4
0savev2_adam_dense_122_bias_v_read_readvariableop6
2savev2_adam_dense_123_kernel_v_read_readvariableop4
0savev2_adam_dense_123_bias_v_read_readvariableopA
=savev2_adam_lstm_70_lstm_cell_70_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_70_lstm_cell_70_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_70_lstm_cell_70_bias_v_read_readvariableopA
=savev2_adam_lstm_71_lstm_cell_71_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_71_lstm_cell_71_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Û
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueúB÷(B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH½
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ë
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_122_kernel_read_readvariableop)savev2_dense_122_bias_read_readvariableop+savev2_dense_123_kernel_read_readvariableop)savev2_dense_123_bias_read_readvariableop6savev2_lstm_70_lstm_cell_70_kernel_read_readvariableop@savev2_lstm_70_lstm_cell_70_recurrent_kernel_read_readvariableop4savev2_lstm_70_lstm_cell_70_bias_read_readvariableop6savev2_lstm_71_lstm_cell_71_kernel_read_readvariableop@savev2_lstm_71_lstm_cell_71_recurrent_kernel_read_readvariableop4savev2_lstm_71_lstm_cell_71_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_122_kernel_m_read_readvariableop0savev2_adam_dense_122_bias_m_read_readvariableop2savev2_adam_dense_123_kernel_m_read_readvariableop0savev2_adam_dense_123_bias_m_read_readvariableop=savev2_adam_lstm_70_lstm_cell_70_kernel_m_read_readvariableopGsavev2_adam_lstm_70_lstm_cell_70_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_70_lstm_cell_70_bias_m_read_readvariableop=savev2_adam_lstm_71_lstm_cell_71_kernel_m_read_readvariableopGsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_71_lstm_cell_71_bias_m_read_readvariableop2savev2_adam_dense_122_kernel_v_read_readvariableop0savev2_adam_dense_122_bias_v_read_readvariableop2savev2_adam_dense_123_kernel_v_read_readvariableop0savev2_adam_dense_123_bias_v_read_readvariableop=savev2_adam_lstm_70_lstm_cell_70_kernel_v_read_readvariableopGsavev2_adam_lstm_70_lstm_cell_70_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_70_lstm_cell_70_bias_v_read_readvariableop=savev2_adam_lstm_71_lstm_cell_71_kernel_v_read_readvariableopGsavev2_adam_lstm_71_lstm_cell_71_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_71_lstm_cell_71_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*§
_input_shapes
: :d:d:d::x:x:x:x:x:x: : : : : : : : : :d:d:d::x:x:x:x:x:x:d:d:d::x:x:x:x:x:x: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x:$ 

_output_shapes

:x:$	 

_output_shapes

:x: 


_output_shapes
:x:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x:$ 

_output_shapes

:d: 

_output_shapes
:d:$  

_output_shapes

:d: !

_output_shapes
::$" 

_output_shapes

:x:$# 

_output_shapes

:x: $

_output_shapes
:x:$% 

_output_shapes

:x:$& 

_output_shapes

:x: '

_output_shapes
:x:(

_output_shapes
: 
¨
µ
)__inference_lstm_70_layer_call_fn_1299000
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_70_layer_call_and_return_conditional_losses_1296794|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
J

D__inference_lstm_70_layer_call_and_return_conditional_losses_1298118

inputs=
+lstm_cell_70_matmul_readvariableop_resource:x?
-lstm_cell_70_matmul_1_readvariableop_resource:x:
,lstm_cell_70_biasadd_readvariableop_resource:x
identity¢#lstm_cell_70/BiasAdd/ReadVariableOp¢"lstm_cell_70/MatMul/ReadVariableOp¢$lstm_cell_70/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_70/MatMul/ReadVariableOpReadVariableOp+lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMulMatMulstrided_slice_2:output:0*lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMul_1MatMulzeros:output:0,lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_70/addAddV2lstm_cell_70/MatMul:product:0lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_70/BiasAddBiasAddlstm_cell_70/add:z:0+lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_70/splitSplit%lstm_cell_70/split/split_dim:output:0lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_70/SigmoidSigmoidlstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_1Sigmoidlstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_70/mulMullstm_cell_70/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_70/ReluRelulstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_1Mullstm_cell_70/Sigmoid:y:0lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_70/add_1AddV2lstm_cell_70/mul:z:0lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_2Sigmoidlstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_70/Relu_1Relulstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_2Mullstm_cell_70/Sigmoid_2:y:0!lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_70_matmul_readvariableop_resource-lstm_cell_70_matmul_1_readvariableop_resource,lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1298034*
condR
while_cond_1298033*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_70/BiasAdd/ReadVariableOp#^lstm_cell_70/MatMul/ReadVariableOp%^lstm_cell_70/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_70/BiasAdd/ReadVariableOp#lstm_cell_70/BiasAdd/ReadVariableOp2H
"lstm_cell_70/MatMul/ReadVariableOp"lstm_cell_70/MatMul/ReadVariableOp2L
$lstm_cell_70/MatMul_1/ReadVariableOp$lstm_cell_70/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
çÇ
¹	
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298989

inputsE
3lstm_70_lstm_cell_70_matmul_readvariableop_resource:xG
5lstm_70_lstm_cell_70_matmul_1_readvariableop_resource:xB
4lstm_70_lstm_cell_70_biasadd_readvariableop_resource:xE
3lstm_71_lstm_cell_71_matmul_readvariableop_resource:xG
5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource:xB
4lstm_71_lstm_cell_71_biasadd_readvariableop_resource:x:
(dense_122_matmul_readvariableop_resource:d7
)dense_122_biasadd_readvariableop_resource:d:
(dense_123_matmul_readvariableop_resource:d7
)dense_123_biasadd_readvariableop_resource:
identity¢ dense_122/BiasAdd/ReadVariableOp¢dense_122/MatMul/ReadVariableOp¢ dense_123/BiasAdd/ReadVariableOp¢dense_123/MatMul/ReadVariableOp¢+lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp¢*lstm_70/lstm_cell_70/MatMul/ReadVariableOp¢,lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp¢lstm_70/while¢+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp¢*lstm_71/lstm_cell_71/MatMul/ReadVariableOp¢,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp¢lstm_71/whileC
lstm_70/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_70/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_70/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_70/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_70/strided_sliceStridedSlicelstm_70/Shape:output:0$lstm_70/strided_slice/stack:output:0&lstm_70/strided_slice/stack_1:output:0&lstm_70/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_70/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_70/zeros/packedPacklstm_70/strided_slice:output:0lstm_70/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_70/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_70/zerosFilllstm_70/zeros/packed:output:0lstm_70/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_70/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_70/zeros_1/packedPacklstm_70/strided_slice:output:0!lstm_70/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_70/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_70/zeros_1Filllstm_70/zeros_1/packed:output:0lstm_70/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_70/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_70/transpose	Transposeinputslstm_70/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_70/Shape_1Shapelstm_70/transpose:y:0*
T0*
_output_shapes
:g
lstm_70/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_70/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_70/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_70/strided_slice_1StridedSlicelstm_70/Shape_1:output:0&lstm_70/strided_slice_1/stack:output:0(lstm_70/strided_slice_1/stack_1:output:0(lstm_70/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_70/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_70/TensorArrayV2TensorListReserve,lstm_70/TensorArrayV2/element_shape:output:0 lstm_70/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_70/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_70/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_70/transpose:y:0Flstm_70/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_70/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_70/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_70/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_70/strided_slice_2StridedSlicelstm_70/transpose:y:0&lstm_70/strided_slice_2/stack:output:0(lstm_70/strided_slice_2/stack_1:output:0(lstm_70/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_70/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp3lstm_70_lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0­
lstm_70/lstm_cell_70/MatMulMatMul lstm_70/strided_slice_2:output:02lstm_70/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¢
,lstm_70/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp5lstm_70_lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0§
lstm_70/lstm_cell_70/MatMul_1MatMullstm_70/zeros:output:04lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx£
lstm_70/lstm_cell_70/addAddV2%lstm_70/lstm_cell_70/MatMul:product:0'lstm_70/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
+lstm_70/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp4lstm_70_lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0¬
lstm_70/lstm_cell_70/BiasAddBiasAddlstm_70/lstm_cell_70/add:z:03lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxf
$lstm_70/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstm_70/lstm_cell_70/splitSplit-lstm_70/lstm_cell_70/split/split_dim:output:0%lstm_70/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split~
lstm_70/lstm_cell_70/SigmoidSigmoid#lstm_70/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/Sigmoid_1Sigmoid#lstm_70/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/mulMul"lstm_70/lstm_cell_70/Sigmoid_1:y:0lstm_70/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_70/lstm_cell_70/ReluRelu#lstm_70/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/mul_1Mul lstm_70/lstm_cell_70/Sigmoid:y:0'lstm_70/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/add_1AddV2lstm_70/lstm_cell_70/mul:z:0lstm_70/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/Sigmoid_2Sigmoid#lstm_70/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_70/lstm_cell_70/Relu_1Relulstm_70/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
lstm_70/lstm_cell_70/mul_2Mul"lstm_70/lstm_cell_70/Sigmoid_2:y:0)lstm_70/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_70/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_70/TensorArrayV2_1TensorListReserve.lstm_70/TensorArrayV2_1/element_shape:output:0 lstm_70/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_70/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_70/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_70/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
lstm_70/whileWhile#lstm_70/while/loop_counter:output:0)lstm_70/while/maximum_iterations:output:0lstm_70/time:output:0 lstm_70/TensorArrayV2_1:handle:0lstm_70/zeros:output:0lstm_70/zeros_1:output:0 lstm_70/strided_slice_1:output:0?lstm_70/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_70_lstm_cell_70_matmul_readvariableop_resource5lstm_70_lstm_cell_70_matmul_1_readvariableop_resource4lstm_70_lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_70_while_body_1298734*&
condR
lstm_70_while_cond_1298733*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_70/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ú
*lstm_70/TensorArrayV2Stack/TensorListStackTensorListStacklstm_70/while:output:3Alstm_70/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_70/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_70/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_70/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_70/strided_slice_3StridedSlice3lstm_70/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_70/strided_slice_3/stack:output:0(lstm_70/strided_slice_3/stack_1:output:0(lstm_70/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_70/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_70/transpose_1	Transpose3lstm_70/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_70/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_70/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_68/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_68/dropout/MulMullstm_70/transpose_1:y:0!dropout_68/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
dropout_68/dropout/ShapeShapelstm_70/transpose_1:y:0*
T0*
_output_shapes
:¦
/dropout_68/dropout/random_uniform/RandomUniformRandomUniform!dropout_68/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_68/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ë
dropout_68/dropout/GreaterEqualGreaterEqual8dropout_68/dropout/random_uniform/RandomUniform:output:0*dropout_68/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_68/dropout/CastCast#dropout_68/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_68/dropout/Mul_1Muldropout_68/dropout/Mul:z:0dropout_68/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm_71/ShapeShapedropout_68/dropout/Mul_1:z:0*
T0*
_output_shapes
:e
lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_71/strided_sliceStridedSlicelstm_71/Shape:output:0$lstm_71/strided_slice/stack:output:0&lstm_71/strided_slice/stack_1:output:0&lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_71/zeros/packedPacklstm_71/strided_slice:output:0lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_71/zerosFilllstm_71/zeros/packed:output:0lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_71/zeros_1/packedPacklstm_71/strided_slice:output:0!lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_71/zeros_1Filllstm_71/zeros_1/packed:output:0lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_71/transpose	Transposedropout_68/dropout/Mul_1:z:0lstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_71/Shape_1Shapelstm_71/transpose:y:0*
T0*
_output_shapes
:g
lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_71/strided_slice_1StridedSlicelstm_71/Shape_1:output:0&lstm_71/strided_slice_1/stack:output:0(lstm_71/strided_slice_1/stack_1:output:0(lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_71/TensorArrayV2TensorListReserve,lstm_71/TensorArrayV2/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_71/transpose:y:0Flstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_71/strided_slice_2StridedSlicelstm_71/transpose:y:0&lstm_71/strided_slice_2/stack:output:0(lstm_71/strided_slice_2/stack_1:output:0(lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_71/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3lstm_71_lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0­
lstm_71/lstm_cell_71/MatMulMatMul lstm_71/strided_slice_2:output:02lstm_71/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¢
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0§
lstm_71/lstm_cell_71/MatMul_1MatMullstm_71/zeros:output:04lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx£
lstm_71/lstm_cell_71/addAddV2%lstm_71/lstm_cell_71/MatMul:product:0'lstm_71/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0¬
lstm_71/lstm_cell_71/BiasAddBiasAddlstm_71/lstm_cell_71/add:z:03lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxf
$lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstm_71/lstm_cell_71/splitSplit-lstm_71/lstm_cell_71/split/split_dim:output:0%lstm_71/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split~
lstm_71/lstm_cell_71/SigmoidSigmoid#lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/Sigmoid_1Sigmoid#lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/mulMul"lstm_71/lstm_cell_71/Sigmoid_1:y:0lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_71/lstm_cell_71/ReluRelu#lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/mul_1Mul lstm_71/lstm_cell_71/Sigmoid:y:0'lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/add_1AddV2lstm_71/lstm_cell_71/mul:z:0lstm_71/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/Sigmoid_2Sigmoid#lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_71/lstm_cell_71/Relu_1Relulstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
lstm_71/lstm_cell_71/mul_2Mul"lstm_71/lstm_cell_71/Sigmoid_2:y:0)lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
$lstm_71/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_71/TensorArrayV2_1TensorListReserve.lstm_71/TensorArrayV2_1/element_shape:output:0-lstm_71/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
lstm_71/whileWhile#lstm_71/while/loop_counter:output:0)lstm_71/while/maximum_iterations:output:0lstm_71/time:output:0 lstm_71/TensorArrayV2_1:handle:0lstm_71/zeros:output:0lstm_71/zeros_1:output:0 lstm_71/strided_slice_1:output:0?lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_71_lstm_cell_71_matmul_readvariableop_resource5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource4lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_71_while_body_1298882*&
condR
lstm_71_while_cond_1298881*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   î
*lstm_71/TensorArrayV2Stack/TensorListStackTensorListStacklstm_71/while:output:3Alstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsp
lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_71/strided_slice_3StridedSlice3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_71/strided_slice_3/stack:output:0(lstm_71/strided_slice_3/stack_1:output:0(lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_71/transpose_1	Transpose3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_69/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_69/dropout/MulMul lstm_71/strided_slice_3:output:0!dropout_69/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dropout_69/dropout/ShapeShape lstm_71/strided_slice_3:output:0*
T0*
_output_shapes
:¢
/dropout_69/dropout/random_uniform/RandomUniformRandomUniform!dropout_69/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_69/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ç
dropout_69/dropout/GreaterEqualGreaterEqual8dropout_69/dropout/random_uniform/RandomUniform:output:0*dropout_69/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_69/dropout/CastCast#dropout_69/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_69/dropout/Mul_1Muldropout_69/dropout/Mul:z:0dropout_69/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_122/MatMulMatMuldropout_69/dropout/Mul_1:z:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
dense_122/ReluReludense_122/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_123/MatMulMatMuldense_122/Relu:activations:0'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dense_123/SoftmaxSoftmaxdense_123/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydense_123/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp,^lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp+^lstm_70/lstm_cell_70/MatMul/ReadVariableOp-^lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp^lstm_70/while,^lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp+^lstm_71/lstm_cell_71/MatMul/ReadVariableOp-^lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp^lstm_71/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp2Z
+lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp+lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp2X
*lstm_70/lstm_cell_70/MatMul/ReadVariableOp*lstm_70/lstm_cell_70/MatMul/ReadVariableOp2\
,lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp,lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp2
lstm_70/whilelstm_70/while2Z
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp2X
*lstm_71/lstm_cell_71/MatMul/ReadVariableOp*lstm_71/lstm_cell_71/MatMul/ReadVariableOp2\
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp2
lstm_71/whilelstm_71/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8
Ë
while_body_1299378
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_70_matmul_readvariableop_resource_0:xG
5while_lstm_cell_70_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_70_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_70_matmul_readvariableop_resource:xE
3while_lstm_cell_70_matmul_1_readvariableop_resource:x@
2while_lstm_cell_70_biasadd_readvariableop_resource:x¢)while/lstm_cell_70/BiasAdd/ReadVariableOp¢(while/lstm_cell_70/MatMul/ReadVariableOp¢*while/lstm_cell_70/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_70/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_70/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_70/addAddV2#while/lstm_cell_70/MatMul:product:0%while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_70/BiasAddBiasAddwhile/lstm_cell_70/add:z:01while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_70/splitSplit+while/lstm_cell_70/split/split_dim:output:0#while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_70/SigmoidSigmoid!while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_1Sigmoid!while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mulMul while/lstm_cell_70/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_70/ReluRelu!while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_1Mulwhile/lstm_cell_70/Sigmoid:y:0%while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/add_1AddV2while/lstm_cell_70/mul:z:0while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_2Sigmoid!while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_70/Relu_1Reluwhile/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_2Mul while/lstm_cell_70/Sigmoid_2:y:0'while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_70/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_70/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_70/BiasAdd/ReadVariableOp)^while/lstm_cell_70/MatMul/ReadVariableOp+^while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_70_biasadd_readvariableop_resource4while_lstm_cell_70_biasadd_readvariableop_resource_0"l
3while_lstm_cell_70_matmul_1_readvariableop_resource5while_lstm_cell_70_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_70_matmul_readvariableop_resource3while_lstm_cell_70_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_70/BiasAdd/ReadVariableOp)while/lstm_cell_70/BiasAdd/ReadVariableOp2T
(while/lstm_cell_70/MatMul/ReadVariableOp(while/lstm_cell_70/MatMul/ReadVariableOp2X
*while/lstm_cell_70/MatMul_1/ReadVariableOp*while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
þ
³
)__inference_lstm_70_layer_call_fn_1299022

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_70_layer_call_and_return_conditional_losses_1297497s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ	
f
G__inference_dropout_69_layer_call_and_return_conditional_losses_1300283

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1299377
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1299377___redundant_placeholder05
1while_while_cond_1299377___redundant_placeholder15
1while_while_cond_1299377___redundant_placeholder25
1while_while_cond_1299377___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
$
ä
while_body_1297076
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_71_1297100_0:x.
while_lstm_cell_71_1297102_0:x*
while_lstm_cell_71_1297104_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_71_1297100:x,
while_lstm_cell_71_1297102:x(
while_lstm_cell_71_1297104:x¢*while/lstm_cell_71/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0·
*while/lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_71_1297100_0while_lstm_cell_71_1297102_0while_lstm_cell_71_1297104_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1297061r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_71/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_71/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_71/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_71_1297100while_lstm_cell_71_1297100_0":
while_lstm_cell_71_1297102while_lstm_cell_71_1297102_0":
while_lstm_cell_71_1297104while_lstm_cell_71_1297104_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_71/StatefulPartitionedCall*while/lstm_cell_71/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ùQ

(sequential_64_lstm_70_while_body_1296403H
Dsequential_64_lstm_70_while_sequential_64_lstm_70_while_loop_counterN
Jsequential_64_lstm_70_while_sequential_64_lstm_70_while_maximum_iterations+
'sequential_64_lstm_70_while_placeholder-
)sequential_64_lstm_70_while_placeholder_1-
)sequential_64_lstm_70_while_placeholder_2-
)sequential_64_lstm_70_while_placeholder_3G
Csequential_64_lstm_70_while_sequential_64_lstm_70_strided_slice_1_0
sequential_64_lstm_70_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_70_tensorarrayunstack_tensorlistfromtensor_0[
Isequential_64_lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0:x]
Ksequential_64_lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0:xX
Jsequential_64_lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0:x(
$sequential_64_lstm_70_while_identity*
&sequential_64_lstm_70_while_identity_1*
&sequential_64_lstm_70_while_identity_2*
&sequential_64_lstm_70_while_identity_3*
&sequential_64_lstm_70_while_identity_4*
&sequential_64_lstm_70_while_identity_5E
Asequential_64_lstm_70_while_sequential_64_lstm_70_strided_slice_1
}sequential_64_lstm_70_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_70_tensorarrayunstack_tensorlistfromtensorY
Gsequential_64_lstm_70_while_lstm_cell_70_matmul_readvariableop_resource:x[
Isequential_64_lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource:xV
Hsequential_64_lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource:x¢?sequential_64/lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp¢>sequential_64/lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp¢@sequential_64/lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp
Msequential_64/lstm_70/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
?sequential_64/lstm_70/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_64_lstm_70_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_70_tensorarrayunstack_tensorlistfromtensor_0'sequential_64_lstm_70_while_placeholderVsequential_64/lstm_70/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0È
>sequential_64/lstm_70/while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOpIsequential_64_lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0û
/sequential_64/lstm_70/while/lstm_cell_70/MatMulMatMulFsequential_64/lstm_70/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_64/lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÌ
@sequential_64/lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOpKsequential_64_lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0â
1sequential_64/lstm_70/while/lstm_cell_70/MatMul_1MatMul)sequential_64_lstm_70_while_placeholder_2Hsequential_64/lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxß
,sequential_64/lstm_70/while/lstm_cell_70/addAddV29sequential_64/lstm_70/while/lstm_cell_70/MatMul:product:0;sequential_64/lstm_70/while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÆ
?sequential_64/lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOpJsequential_64_lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0è
0sequential_64/lstm_70/while/lstm_cell_70/BiasAddBiasAdd0sequential_64/lstm_70/while/lstm_cell_70/add:z:0Gsequential_64/lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxz
8sequential_64/lstm_70/while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :±
.sequential_64/lstm_70/while/lstm_cell_70/splitSplitAsequential_64/lstm_70/while/lstm_cell_70/split/split_dim:output:09sequential_64/lstm_70/while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¦
0sequential_64/lstm_70/while/lstm_cell_70/SigmoidSigmoid7sequential_64/lstm_70/while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
2sequential_64/lstm_70/while/lstm_cell_70/Sigmoid_1Sigmoid7sequential_64/lstm_70/while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
,sequential_64/lstm_70/while/lstm_cell_70/mulMul6sequential_64/lstm_70/while/lstm_cell_70/Sigmoid_1:y:0)sequential_64_lstm_70_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_64/lstm_70/while/lstm_cell_70/ReluRelu7sequential_64/lstm_70/while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
.sequential_64/lstm_70/while/lstm_cell_70/mul_1Mul4sequential_64/lstm_70/while/lstm_cell_70/Sigmoid:y:0;sequential_64/lstm_70/while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
.sequential_64/lstm_70/while/lstm_cell_70/add_1AddV20sequential_64/lstm_70/while/lstm_cell_70/mul:z:02sequential_64/lstm_70/while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
2sequential_64/lstm_70/while/lstm_cell_70/Sigmoid_2Sigmoid7sequential_64/lstm_70/while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/sequential_64/lstm_70/while/lstm_cell_70/Relu_1Relu2sequential_64/lstm_70/while/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
.sequential_64/lstm_70/while/lstm_cell_70/mul_2Mul6sequential_64/lstm_70/while/lstm_cell_70/Sigmoid_2:y:0=sequential_64/lstm_70/while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential_64/lstm_70/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_64_lstm_70_while_placeholder_1'sequential_64_lstm_70_while_placeholder2sequential_64/lstm_70/while/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒc
!sequential_64/lstm_70/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_64/lstm_70/while/addAddV2'sequential_64_lstm_70_while_placeholder*sequential_64/lstm_70/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_64/lstm_70/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!sequential_64/lstm_70/while/add_1AddV2Dsequential_64_lstm_70_while_sequential_64_lstm_70_while_loop_counter,sequential_64/lstm_70/while/add_1/y:output:0*
T0*
_output_shapes
: 
$sequential_64/lstm_70/while/IdentityIdentity%sequential_64/lstm_70/while/add_1:z:0!^sequential_64/lstm_70/while/NoOp*
T0*
_output_shapes
: Â
&sequential_64/lstm_70/while/Identity_1IdentityJsequential_64_lstm_70_while_sequential_64_lstm_70_while_maximum_iterations!^sequential_64/lstm_70/while/NoOp*
T0*
_output_shapes
: 
&sequential_64/lstm_70/while/Identity_2Identity#sequential_64/lstm_70/while/add:z:0!^sequential_64/lstm_70/while/NoOp*
T0*
_output_shapes
: È
&sequential_64/lstm_70/while/Identity_3IdentityPsequential_64/lstm_70/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_64/lstm_70/while/NoOp*
T0*
_output_shapes
: »
&sequential_64/lstm_70/while/Identity_4Identity2sequential_64/lstm_70/while/lstm_cell_70/mul_2:z:0!^sequential_64/lstm_70/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
&sequential_64/lstm_70/while/Identity_5Identity2sequential_64/lstm_70/while/lstm_cell_70/add_1:z:0!^sequential_64/lstm_70/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 sequential_64/lstm_70/while/NoOpNoOp@^sequential_64/lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp?^sequential_64/lstm_70/while/lstm_cell_70/MatMul/ReadVariableOpA^sequential_64/lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_64_lstm_70_while_identity-sequential_64/lstm_70/while/Identity:output:0"Y
&sequential_64_lstm_70_while_identity_1/sequential_64/lstm_70/while/Identity_1:output:0"Y
&sequential_64_lstm_70_while_identity_2/sequential_64/lstm_70/while/Identity_2:output:0"Y
&sequential_64_lstm_70_while_identity_3/sequential_64/lstm_70/while/Identity_3:output:0"Y
&sequential_64_lstm_70_while_identity_4/sequential_64/lstm_70/while/Identity_4:output:0"Y
&sequential_64_lstm_70_while_identity_5/sequential_64/lstm_70/while/Identity_5:output:0"
Hsequential_64_lstm_70_while_lstm_cell_70_biasadd_readvariableop_resourceJsequential_64_lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0"
Isequential_64_lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resourceKsequential_64_lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0"
Gsequential_64_lstm_70_while_lstm_cell_70_matmul_readvariableop_resourceIsequential_64_lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0"
Asequential_64_lstm_70_while_sequential_64_lstm_70_strided_slice_1Csequential_64_lstm_70_while_sequential_64_lstm_70_strided_slice_1_0"
}sequential_64_lstm_70_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_70_tensorarrayunstack_tensorlistfromtensorsequential_64_lstm_70_while_tensorarrayv2read_tensorlistgetitem_sequential_64_lstm_70_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
?sequential_64/lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp?sequential_64/lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp2
>sequential_64/lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp>sequential_64/lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp2
@sequential_64/lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp@sequential_64/lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


÷
F__inference_dense_122_layer_call_and_return_conditional_losses_1300303

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²¸
¹	
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298675

inputsE
3lstm_70_lstm_cell_70_matmul_readvariableop_resource:xG
5lstm_70_lstm_cell_70_matmul_1_readvariableop_resource:xB
4lstm_70_lstm_cell_70_biasadd_readvariableop_resource:xE
3lstm_71_lstm_cell_71_matmul_readvariableop_resource:xG
5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource:xB
4lstm_71_lstm_cell_71_biasadd_readvariableop_resource:x:
(dense_122_matmul_readvariableop_resource:d7
)dense_122_biasadd_readvariableop_resource:d:
(dense_123_matmul_readvariableop_resource:d7
)dense_123_biasadd_readvariableop_resource:
identity¢ dense_122/BiasAdd/ReadVariableOp¢dense_122/MatMul/ReadVariableOp¢ dense_123/BiasAdd/ReadVariableOp¢dense_123/MatMul/ReadVariableOp¢+lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp¢*lstm_70/lstm_cell_70/MatMul/ReadVariableOp¢,lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp¢lstm_70/while¢+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp¢*lstm_71/lstm_cell_71/MatMul/ReadVariableOp¢,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp¢lstm_71/whileC
lstm_70/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_70/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_70/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_70/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_70/strided_sliceStridedSlicelstm_70/Shape:output:0$lstm_70/strided_slice/stack:output:0&lstm_70/strided_slice/stack_1:output:0&lstm_70/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_70/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_70/zeros/packedPacklstm_70/strided_slice:output:0lstm_70/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_70/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_70/zerosFilllstm_70/zeros/packed:output:0lstm_70/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_70/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_70/zeros_1/packedPacklstm_70/strided_slice:output:0!lstm_70/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_70/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_70/zeros_1Filllstm_70/zeros_1/packed:output:0lstm_70/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_70/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_70/transpose	Transposeinputslstm_70/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_70/Shape_1Shapelstm_70/transpose:y:0*
T0*
_output_shapes
:g
lstm_70/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_70/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_70/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_70/strided_slice_1StridedSlicelstm_70/Shape_1:output:0&lstm_70/strided_slice_1/stack:output:0(lstm_70/strided_slice_1/stack_1:output:0(lstm_70/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_70/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_70/TensorArrayV2TensorListReserve,lstm_70/TensorArrayV2/element_shape:output:0 lstm_70/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_70/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_70/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_70/transpose:y:0Flstm_70/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_70/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_70/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_70/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_70/strided_slice_2StridedSlicelstm_70/transpose:y:0&lstm_70/strided_slice_2/stack:output:0(lstm_70/strided_slice_2/stack_1:output:0(lstm_70/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_70/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp3lstm_70_lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0­
lstm_70/lstm_cell_70/MatMulMatMul lstm_70/strided_slice_2:output:02lstm_70/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¢
,lstm_70/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp5lstm_70_lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0§
lstm_70/lstm_cell_70/MatMul_1MatMullstm_70/zeros:output:04lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx£
lstm_70/lstm_cell_70/addAddV2%lstm_70/lstm_cell_70/MatMul:product:0'lstm_70/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
+lstm_70/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp4lstm_70_lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0¬
lstm_70/lstm_cell_70/BiasAddBiasAddlstm_70/lstm_cell_70/add:z:03lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxf
$lstm_70/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstm_70/lstm_cell_70/splitSplit-lstm_70/lstm_cell_70/split/split_dim:output:0%lstm_70/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split~
lstm_70/lstm_cell_70/SigmoidSigmoid#lstm_70/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/Sigmoid_1Sigmoid#lstm_70/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/mulMul"lstm_70/lstm_cell_70/Sigmoid_1:y:0lstm_70/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_70/lstm_cell_70/ReluRelu#lstm_70/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/mul_1Mul lstm_70/lstm_cell_70/Sigmoid:y:0'lstm_70/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/add_1AddV2lstm_70/lstm_cell_70/mul:z:0lstm_70/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/lstm_cell_70/Sigmoid_2Sigmoid#lstm_70/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_70/lstm_cell_70/Relu_1Relulstm_70/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
lstm_70/lstm_cell_70/mul_2Mul"lstm_70/lstm_cell_70/Sigmoid_2:y:0)lstm_70/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_70/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_70/TensorArrayV2_1TensorListReserve.lstm_70/TensorArrayV2_1/element_shape:output:0 lstm_70/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_70/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_70/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_70/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
lstm_70/whileWhile#lstm_70/while/loop_counter:output:0)lstm_70/while/maximum_iterations:output:0lstm_70/time:output:0 lstm_70/TensorArrayV2_1:handle:0lstm_70/zeros:output:0lstm_70/zeros_1:output:0 lstm_70/strided_slice_1:output:0?lstm_70/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_70_lstm_cell_70_matmul_readvariableop_resource5lstm_70_lstm_cell_70_matmul_1_readvariableop_resource4lstm_70_lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_70_while_body_1298434*&
condR
lstm_70_while_cond_1298433*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_70/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ú
*lstm_70/TensorArrayV2Stack/TensorListStackTensorListStacklstm_70/while:output:3Alstm_70/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_70/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_70/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_70/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_70/strided_slice_3StridedSlice3lstm_70/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_70/strided_slice_3/stack:output:0(lstm_70/strided_slice_3/stack_1:output:0(lstm_70/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_70/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_70/transpose_1	Transpose3lstm_70/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_70/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_70/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    n
dropout_68/IdentityIdentitylstm_70/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
lstm_71/ShapeShapedropout_68/Identity:output:0*
T0*
_output_shapes
:e
lstm_71/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_71/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_71/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_71/strided_sliceStridedSlicelstm_71/Shape:output:0$lstm_71/strided_slice/stack:output:0&lstm_71/strided_slice/stack_1:output:0&lstm_71/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_71/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_71/zeros/packedPacklstm_71/strided_slice:output:0lstm_71/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_71/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_71/zerosFilllstm_71/zeros/packed:output:0lstm_71/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_71/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_71/zeros_1/packedPacklstm_71/strided_slice:output:0!lstm_71/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_71/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_71/zeros_1Filllstm_71/zeros_1/packed:output:0lstm_71/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_71/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_71/transpose	Transposedropout_68/Identity:output:0lstm_71/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_71/Shape_1Shapelstm_71/transpose:y:0*
T0*
_output_shapes
:g
lstm_71/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_71/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_71/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_71/strided_slice_1StridedSlicelstm_71/Shape_1:output:0&lstm_71/strided_slice_1/stack:output:0(lstm_71/strided_slice_1/stack_1:output:0(lstm_71/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_71/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_71/TensorArrayV2TensorListReserve,lstm_71/TensorArrayV2/element_shape:output:0 lstm_71/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_71/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_71/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_71/transpose:y:0Flstm_71/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_71/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_71/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_71/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_71/strided_slice_2StridedSlicelstm_71/transpose:y:0&lstm_71/strided_slice_2/stack:output:0(lstm_71/strided_slice_2/stack_1:output:0(lstm_71/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_71/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3lstm_71_lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0­
lstm_71/lstm_cell_71/MatMulMatMul lstm_71/strided_slice_2:output:02lstm_71/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¢
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0§
lstm_71/lstm_cell_71/MatMul_1MatMullstm_71/zeros:output:04lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx£
lstm_71/lstm_cell_71/addAddV2%lstm_71/lstm_cell_71/MatMul:product:0'lstm_71/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0¬
lstm_71/lstm_cell_71/BiasAddBiasAddlstm_71/lstm_cell_71/add:z:03lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxf
$lstm_71/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstm_71/lstm_cell_71/splitSplit-lstm_71/lstm_cell_71/split/split_dim:output:0%lstm_71/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split~
lstm_71/lstm_cell_71/SigmoidSigmoid#lstm_71/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/Sigmoid_1Sigmoid#lstm_71/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/mulMul"lstm_71/lstm_cell_71/Sigmoid_1:y:0lstm_71/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_71/lstm_cell_71/ReluRelu#lstm_71/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/mul_1Mul lstm_71/lstm_cell_71/Sigmoid:y:0'lstm_71/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/add_1AddV2lstm_71/lstm_cell_71/mul:z:0lstm_71/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/lstm_cell_71/Sigmoid_2Sigmoid#lstm_71/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_71/lstm_cell_71/Relu_1Relulstm_71/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
lstm_71/lstm_cell_71/mul_2Mul"lstm_71/lstm_cell_71/Sigmoid_2:y:0)lstm_71/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_71/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
$lstm_71/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_71/TensorArrayV2_1TensorListReserve.lstm_71/TensorArrayV2_1/element_shape:output:0-lstm_71/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_71/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_71/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_71/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
lstm_71/whileWhile#lstm_71/while/loop_counter:output:0)lstm_71/while/maximum_iterations:output:0lstm_71/time:output:0 lstm_71/TensorArrayV2_1:handle:0lstm_71/zeros:output:0lstm_71/zeros_1:output:0 lstm_71/strided_slice_1:output:0?lstm_71/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_71_lstm_cell_71_matmul_readvariableop_resource5lstm_71_lstm_cell_71_matmul_1_readvariableop_resource4lstm_71_lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_71_while_body_1298575*&
condR
lstm_71_while_cond_1298574*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_71/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   î
*lstm_71/TensorArrayV2Stack/TensorListStackTensorListStacklstm_71/while:output:3Alstm_71/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsp
lstm_71/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_71/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_71/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_71/strided_slice_3StridedSlice3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_71/strided_slice_3/stack:output:0(lstm_71/strided_slice_3/stack_1:output:0(lstm_71/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_71/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_71/transpose_1	Transpose3lstm_71/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_71/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_71/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    s
dropout_69/IdentityIdentity lstm_71/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_122/MatMulMatMuldropout_69/Identity:output:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
dense_122/ReluReludense_122/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_123/MatMulMatMuldense_122/Relu:activations:0'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dense_123/SoftmaxSoftmaxdense_123/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydense_123/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp,^lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp+^lstm_70/lstm_cell_70/MatMul/ReadVariableOp-^lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp^lstm_70/while,^lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp+^lstm_71/lstm_cell_71/MatMul/ReadVariableOp-^lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp^lstm_71/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp2Z
+lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp+lstm_70/lstm_cell_70/BiasAdd/ReadVariableOp2X
*lstm_70/lstm_cell_70/MatMul/ReadVariableOp*lstm_70/lstm_cell_70/MatMul/ReadVariableOp2\
,lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp,lstm_70/lstm_cell_70/MatMul_1/ReadVariableOp2
lstm_70/whilelstm_70/while2Z
+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp+lstm_71/lstm_cell_71/BiasAdd/ReadVariableOp2X
*lstm_71/lstm_cell_71/MatMul/ReadVariableOp*lstm_71/lstm_cell_71/MatMul/ReadVariableOp2\
,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp,lstm_71/lstm_cell_71/MatMul_1/ReadVariableOp2
lstm_71/whilelstm_71/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
9
Ë
while_body_1299736
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_71_matmul_readvariableop_resource_0:xG
5while_lstm_cell_71_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_71_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_71_matmul_readvariableop_resource:xE
3while_lstm_cell_71_matmul_1_readvariableop_resource:x@
2while_lstm_cell_71_biasadd_readvariableop_resource:x¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¼
Ë
J__inference_sequential_64_layer_call_and_return_conditional_losses_1297706

inputs!
lstm_70_1297498:x!
lstm_70_1297500:x
lstm_70_1297502:x!
lstm_71_1297657:x!
lstm_71_1297659:x
lstm_71_1297661:x#
dense_122_1297683:d
dense_122_1297685:d#
dense_123_1297700:d
dense_123_1297702:
identity¢!dense_122/StatefulPartitionedCall¢!dense_123/StatefulPartitionedCall¢lstm_70/StatefulPartitionedCall¢lstm_71/StatefulPartitionedCall
lstm_70/StatefulPartitionedCallStatefulPartitionedCallinputslstm_70_1297498lstm_70_1297500lstm_70_1297502*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_70_layer_call_and_return_conditional_losses_1297497ã
dropout_68/PartitionedCallPartitionedCall(lstm_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_68_layer_call_and_return_conditional_losses_1297510
lstm_71/StatefulPartitionedCallStatefulPartitionedCall#dropout_68/PartitionedCall:output:0lstm_71_1297657lstm_71_1297659lstm_71_1297661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_1297656ß
dropout_69/PartitionedCallPartitionedCall(lstm_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_69_layer_call_and_return_conditional_losses_1297669
!dense_122/StatefulPartitionedCallStatefulPartitionedCall#dropout_69/PartitionedCall:output:0dense_122_1297683dense_122_1297685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1297682
!dense_123/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0dense_123_1297700dense_123_1297702*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_1297699y
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp"^dense_122/StatefulPartitionedCall"^dense_123/StatefulPartitionedCall ^lstm_70/StatefulPartitionedCall ^lstm_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2B
lstm_70/StatefulPartitionedCalllstm_70/StatefulPartitionedCall2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1299520
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1299520___redundant_placeholder05
1while_while_cond_1299520___redundant_placeholder15
1while_while_cond_1299520___redundant_placeholder25
1while_while_cond_1299520___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¢

÷
F__inference_dense_123_layer_call_and_return_conditional_losses_1300323

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Æ

+__inference_dense_123_layer_call_fn_1300312

inputs
unknown:d
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_1297699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ô

I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1300389

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ª

ø
/__inference_sequential_64_layer_call_fn_1298350

inputs
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:d
	unknown_6:d
	unknown_7:d
	unknown_8:
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_1297706o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
9
Ë
while_body_1297845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_71_matmul_readvariableop_resource_0:xG
5while_lstm_cell_71_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_71_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_71_matmul_readvariableop_resource:xE
3while_lstm_cell_71_matmul_1_readvariableop_resource:x@
2while_lstm_cell_71_biasadd_readvariableop_resource:x¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ú
e
G__inference_dropout_69_layer_call_and_return_conditional_losses_1300271

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

µ
)__inference_lstm_71_layer_call_fn_1299654
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_1297339o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¹A
Ë

lstm_70_while_body_1298434,
(lstm_70_while_lstm_70_while_loop_counter2
.lstm_70_while_lstm_70_while_maximum_iterations
lstm_70_while_placeholder
lstm_70_while_placeholder_1
lstm_70_while_placeholder_2
lstm_70_while_placeholder_3+
'lstm_70_while_lstm_70_strided_slice_1_0g
clstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0:xO
=lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0:xJ
<lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0:x
lstm_70_while_identity
lstm_70_while_identity_1
lstm_70_while_identity_2
lstm_70_while_identity_3
lstm_70_while_identity_4
lstm_70_while_identity_5)
%lstm_70_while_lstm_70_strided_slice_1e
alstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensorK
9lstm_70_while_lstm_cell_70_matmul_readvariableop_resource:xM
;lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource:xH
:lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource:x¢1lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp¢0lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp¢2lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp
?lstm_70/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_70/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensor_0lstm_70_while_placeholderHlstm_70/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¬
0lstm_70/while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp;lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0Ñ
!lstm_70/while/lstm_cell_70/MatMulMatMul8lstm_70/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx°
2lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp=lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0¸
#lstm_70/while/lstm_cell_70/MatMul_1MatMullstm_70_while_placeholder_2:lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxµ
lstm_70/while/lstm_cell_70/addAddV2+lstm_70/while/lstm_cell_70/MatMul:product:0-lstm_70/while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxª
1lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp<lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¾
"lstm_70/while/lstm_cell_70/BiasAddBiasAdd"lstm_70/while/lstm_cell_70/add:z:09lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxl
*lstm_70/while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_70/while/lstm_cell_70/splitSplit3lstm_70/while/lstm_cell_70/split/split_dim:output:0+lstm_70/while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_70/while/lstm_cell_70/SigmoidSigmoid)lstm_70/while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_70/while/lstm_cell_70/Sigmoid_1Sigmoid)lstm_70/while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/while/lstm_cell_70/mulMul(lstm_70/while/lstm_cell_70/Sigmoid_1:y:0lstm_70_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/while/lstm_cell_70/ReluRelu)lstm_70/while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 lstm_70/while/lstm_cell_70/mul_1Mul&lstm_70/while/lstm_cell_70/Sigmoid:y:0-lstm_70/while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 lstm_70/while/lstm_cell_70/add_1AddV2"lstm_70/while/lstm_cell_70/mul:z:0$lstm_70/while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_70/while/lstm_cell_70/Sigmoid_2Sigmoid)lstm_70/while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_70/while/lstm_cell_70/Relu_1Relu$lstm_70/while/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 lstm_70/while/lstm_cell_70/mul_2Mul(lstm_70/while/lstm_cell_70/Sigmoid_2:y:0/lstm_70/while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
2lstm_70/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_70_while_placeholder_1lstm_70_while_placeholder$lstm_70/while/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_70/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_70/while/addAddV2lstm_70_while_placeholderlstm_70/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_70/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_70/while/add_1AddV2(lstm_70_while_lstm_70_while_loop_counterlstm_70/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_70/while/IdentityIdentitylstm_70/while/add_1:z:0^lstm_70/while/NoOp*
T0*
_output_shapes
: 
lstm_70/while/Identity_1Identity.lstm_70_while_lstm_70_while_maximum_iterations^lstm_70/while/NoOp*
T0*
_output_shapes
: q
lstm_70/while/Identity_2Identitylstm_70/while/add:z:0^lstm_70/while/NoOp*
T0*
_output_shapes
: 
lstm_70/while/Identity_3IdentityBlstm_70/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_70/while/NoOp*
T0*
_output_shapes
: 
lstm_70/while/Identity_4Identity$lstm_70/while/lstm_cell_70/mul_2:z:0^lstm_70/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/while/Identity_5Identity$lstm_70/while/lstm_cell_70/add_1:z:0^lstm_70/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_70/while/NoOpNoOp2^lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp1^lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp3^lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_70_while_identitylstm_70/while/Identity:output:0"=
lstm_70_while_identity_1!lstm_70/while/Identity_1:output:0"=
lstm_70_while_identity_2!lstm_70/while/Identity_2:output:0"=
lstm_70_while_identity_3!lstm_70/while/Identity_3:output:0"=
lstm_70_while_identity_4!lstm_70/while/Identity_4:output:0"=
lstm_70_while_identity_5!lstm_70/while/Identity_5:output:0"P
%lstm_70_while_lstm_70_strided_slice_1'lstm_70_while_lstm_70_strided_slice_1_0"z
:lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource<lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0"|
;lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource=lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0"x
9lstm_70_while_lstm_cell_70_matmul_readvariableop_resource;lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0"È
alstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensorclstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp1lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp2d
0lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp0lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp2h
2lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp2lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÅK

D__inference_lstm_71_layer_call_and_return_conditional_losses_1299966
inputs_0=
+lstm_cell_71_matmul_readvariableop_resource:x?
-lstm_cell_71_matmul_1_readvariableop_resource:x:
,lstm_cell_71_biasadd_readvariableop_resource:x
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1299881*
condR
while_cond_1299880*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ê
ô
.__inference_lstm_cell_71_layer_call_fn_1300455

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1297209o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ñ"
ä
while_body_1296916
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_70_1296940_0:x.
while_lstm_cell_70_1296942_0:x*
while_lstm_cell_70_1296944_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_70_1296940:x,
while_lstm_cell_70_1296942:x(
while_lstm_cell_70_1296944:x¢*while/lstm_cell_70/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0·
*while/lstm_cell_70/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_70_1296940_0while_lstm_cell_70_1296942_0while_lstm_cell_70_1296944_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1296857Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_70/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_70/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_70/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_70/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_70_1296940while_lstm_cell_70_1296940_0":
while_lstm_cell_70_1296942while_lstm_cell_70_1296942_0":
while_lstm_cell_70_1296944while_lstm_cell_70_1296944_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_70/StatefulPartitionedCall*while/lstm_cell_70/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
8
Ë
while_body_1299092
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_70_matmul_readvariableop_resource_0:xG
5while_lstm_cell_70_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_70_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_70_matmul_readvariableop_resource:xE
3while_lstm_cell_70_matmul_1_readvariableop_resource:x@
2while_lstm_cell_70_biasadd_readvariableop_resource:x¢)while/lstm_cell_70/BiasAdd/ReadVariableOp¢(while/lstm_cell_70/MatMul/ReadVariableOp¢*while/lstm_cell_70/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_70/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_70/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_70/addAddV2#while/lstm_cell_70/MatMul:product:0%while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_70/BiasAddBiasAddwhile/lstm_cell_70/add:z:01while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_70/splitSplit+while/lstm_cell_70/split/split_dim:output:0#while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_70/SigmoidSigmoid!while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_1Sigmoid!while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mulMul while/lstm_cell_70/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_70/ReluRelu!while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_1Mulwhile/lstm_cell_70/Sigmoid:y:0%while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/add_1AddV2while/lstm_cell_70/mul:z:0while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_2Sigmoid!while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_70/Relu_1Reluwhile/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_2Mul while/lstm_cell_70/Sigmoid_2:y:0'while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_70/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_70/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_70/BiasAdd/ReadVariableOp)^while/lstm_cell_70/MatMul/ReadVariableOp+^while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_70_biasadd_readvariableop_resource4while_lstm_cell_70_biasadd_readvariableop_resource_0"l
3while_lstm_cell_70_matmul_1_readvariableop_resource5while_lstm_cell_70_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_70_matmul_readvariableop_resource3while_lstm_cell_70_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_70/BiasAdd/ReadVariableOp)while/lstm_cell_70/BiasAdd/ReadVariableOp2T
(while/lstm_cell_70/MatMul/ReadVariableOp(while/lstm_cell_70/MatMul/ReadVariableOp2X
*while/lstm_cell_70/MatMul_1/ReadVariableOp*while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
8

D__inference_lstm_70_layer_call_and_return_conditional_losses_1296794

inputs&
lstm_cell_70_1296712:x&
lstm_cell_70_1296714:x"
lstm_cell_70_1296716:x
identity¢$lstm_cell_70/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskù
$lstm_cell_70/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_70_1296712lstm_cell_70_1296714lstm_cell_70_1296716*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1296711n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_70_1296712lstm_cell_70_1296714lstm_cell_70_1296716*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1296725*
condR
while_cond_1296724*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_70/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_70/StatefulPartitionedCall$lstm_cell_70/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1297844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1297844___redundant_placeholder05
1while_while_cond_1297844___redundant_placeholder15
1while_while_cond_1297844___redundant_placeholder25
1while_while_cond_1297844___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
£
H
,__inference_dropout_69_layer_call_fn_1300261

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_69_layer_call_and_return_conditional_losses_1297669`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
e
,__inference_dropout_69_layer_call_fn_1300266

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_69_layer_call_and_return_conditional_losses_1297769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1297570
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1297570___redundant_placeholder05
1while_while_cond_1297570___redundant_placeholder15
1while_while_cond_1297570___redundant_placeholder25
1while_while_cond_1297570___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
º
È
while_cond_1297075
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1297075___redundant_placeholder05
1while_while_cond_1297075___redundant_placeholder15
1while_while_cond_1297075___redundant_placeholder25
1while_while_cond_1297075___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


è
lstm_70_while_cond_1298733,
(lstm_70_while_lstm_70_while_loop_counter2
.lstm_70_while_lstm_70_while_maximum_iterations
lstm_70_while_placeholder
lstm_70_while_placeholder_1
lstm_70_while_placeholder_2
lstm_70_while_placeholder_3.
*lstm_70_while_less_lstm_70_strided_slice_1E
Alstm_70_while_lstm_70_while_cond_1298733___redundant_placeholder0E
Alstm_70_while_lstm_70_while_cond_1298733___redundant_placeholder1E
Alstm_70_while_lstm_70_while_cond_1298733___redundant_placeholder2E
Alstm_70_while_lstm_70_while_cond_1298733___redundant_placeholder3
lstm_70_while_identity

lstm_70/while/LessLesslstm_70_while_placeholder*lstm_70_while_less_lstm_70_strided_slice_1*
T0*
_output_shapes
: [
lstm_70/while/IdentityIdentitylstm_70/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_70_while_identitylstm_70/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¿

ÿ
/__inference_sequential_64_layer_call_fn_1298232
lstm_70_input
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:d
	unknown_6:d
	unknown_7:d
	unknown_8:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCalllstm_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298184o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_70_input
º
È
while_cond_1299091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1299091___redundant_placeholder05
1while_while_cond_1299091___redundant_placeholder15
1while_while_cond_1299091___redundant_placeholder25
1while_while_cond_1299091___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
9
Ë
while_body_1299881
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_71_matmul_readvariableop_resource_0:xG
5while_lstm_cell_71_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_71_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_71_matmul_readvariableop_resource:xE
3while_lstm_cell_71_matmul_1_readvariableop_resource:x@
2while_lstm_cell_71_biasadd_readvariableop_resource:x¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¿

ÿ
/__inference_sequential_64_layer_call_fn_1297729
lstm_70_input
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:d
	unknown_6:d
	unknown_7:d
	unknown_8:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCalllstm_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_1297706o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_70_input
¢

÷
F__inference_dense_123_layer_call_and_return_conditional_losses_1297699

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
8
Ë
while_body_1299235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_70_matmul_readvariableop_resource_0:xG
5while_lstm_cell_70_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_70_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_70_matmul_readvariableop_resource:xE
3while_lstm_cell_70_matmul_1_readvariableop_resource:x@
2while_lstm_cell_70_biasadd_readvariableop_resource:x¢)while/lstm_cell_70/BiasAdd/ReadVariableOp¢(while/lstm_cell_70/MatMul/ReadVariableOp¢*while/lstm_cell_70/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_70/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_70/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_70/addAddV2#while/lstm_cell_70/MatMul:product:0%while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_70/BiasAddBiasAddwhile/lstm_cell_70/add:z:01while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_70/splitSplit+while/lstm_cell_70/split/split_dim:output:0#while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_70/SigmoidSigmoid!while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_1Sigmoid!while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mulMul while/lstm_cell_70/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_70/ReluRelu!while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_1Mulwhile/lstm_cell_70/Sigmoid:y:0%while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/add_1AddV2while/lstm_cell_70/mul:z:0while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_2Sigmoid!while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_70/Relu_1Reluwhile/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_2Mul while/lstm_cell_70/Sigmoid_2:y:0'while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_70/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_70/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_70/BiasAdd/ReadVariableOp)^while/lstm_cell_70/MatMul/ReadVariableOp+^while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_70_biasadd_readvariableop_resource4while_lstm_cell_70_biasadd_readvariableop_resource_0"l
3while_lstm_cell_70_matmul_1_readvariableop_resource5while_lstm_cell_70_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_70_matmul_readvariableop_resource3while_lstm_cell_70_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_70/BiasAdd/ReadVariableOp)while/lstm_cell_70/BiasAdd/ReadVariableOp2T
(while/lstm_cell_70/MatMul/ReadVariableOp(while/lstm_cell_70/MatMul/ReadVariableOp2X
*while/lstm_cell_70/MatMul_1/ReadVariableOp*while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¢K

D__inference_lstm_71_layer_call_and_return_conditional_losses_1297656

inputs=
+lstm_cell_71_matmul_readvariableop_resource:x?
-lstm_cell_71_matmul_1_readvariableop_resource:x:
,lstm_cell_71_biasadd_readvariableop_resource:x
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1297571*
condR
while_cond_1297570*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
ô
.__inference_lstm_cell_70_layer_call_fn_1300340

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1296711o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1

e
,__inference_dropout_68_layer_call_fn_1299615

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_68_layer_call_and_return_conditional_losses_1297959s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8
Ë
while_body_1298034
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_70_matmul_readvariableop_resource_0:xG
5while_lstm_cell_70_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_70_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_70_matmul_readvariableop_resource:xE
3while_lstm_cell_70_matmul_1_readvariableop_resource:x@
2while_lstm_cell_70_biasadd_readvariableop_resource:x¢)while/lstm_cell_70/BiasAdd/ReadVariableOp¢(while/lstm_cell_70/MatMul/ReadVariableOp¢*while/lstm_cell_70/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_70/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_70/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_70/addAddV2#while/lstm_cell_70/MatMul:product:0%while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_70/BiasAddBiasAddwhile/lstm_cell_70/add:z:01while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_70/splitSplit+while/lstm_cell_70/split/split_dim:output:0#while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_70/SigmoidSigmoid!while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_1Sigmoid!while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mulMul while/lstm_cell_70/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_70/ReluRelu!while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_1Mulwhile/lstm_cell_70/Sigmoid:y:0%while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/add_1AddV2while/lstm_cell_70/mul:z:0while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_2Sigmoid!while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_70/Relu_1Reluwhile/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_2Mul while/lstm_cell_70/Sigmoid_2:y:0'while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_70/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_70/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_70/BiasAdd/ReadVariableOp)^while/lstm_cell_70/MatMul/ReadVariableOp+^while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_70_biasadd_readvariableop_resource4while_lstm_cell_70_biasadd_readvariableop_resource_0"l
3while_lstm_cell_70_matmul_1_readvariableop_resource5while_lstm_cell_70_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_70_matmul_readvariableop_resource3while_lstm_cell_70_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_70/BiasAdd/ReadVariableOp)while/lstm_cell_70/BiasAdd/ReadVariableOp2T
(while/lstm_cell_70/MatMul/ReadVariableOp(while/lstm_cell_70/MatMul/ReadVariableOp2X
*while/lstm_cell_70/MatMul_1/ReadVariableOp*while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


÷
F__inference_dense_122_layer_call_and_return_conditional_losses_1297682

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿda
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1297412
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1297412___redundant_placeholder05
1while_while_cond_1297412___redundant_placeholder15
1while_while_cond_1297412___redundant_placeholder25
1while_while_cond_1297412___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÝB
Ë

lstm_71_while_body_1298882,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3+
'lstm_71_while_lstm_71_strided_slice_1_0g
clstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0:xO
=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0:xJ
<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0:x
lstm_71_while_identity
lstm_71_while_identity_1
lstm_71_while_identity_2
lstm_71_while_identity_3
lstm_71_while_identity_4
lstm_71_while_identity_5)
%lstm_71_while_lstm_71_strided_slice_1e
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorK
9lstm_71_while_lstm_cell_71_matmul_readvariableop_resource:xM
;lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource:xH
:lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource:x¢1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp¢0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp¢2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0lstm_71_while_placeholderHlstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¬
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0Ñ
!lstm_71/while/lstm_cell_71/MatMulMatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx°
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0¸
#lstm_71/while/lstm_cell_71/MatMul_1MatMullstm_71_while_placeholder_2:lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxµ
lstm_71/while/lstm_cell_71/addAddV2+lstm_71/while/lstm_cell_71/MatMul:product:0-lstm_71/while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxª
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¾
"lstm_71/while/lstm_cell_71/BiasAddBiasAdd"lstm_71/while/lstm_cell_71/add:z:09lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxl
*lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_71/while/lstm_cell_71/splitSplit3lstm_71/while/lstm_cell_71/split/split_dim:output:0+lstm_71/while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_71/while/lstm_cell_71/SigmoidSigmoid)lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid)lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/while/lstm_cell_71/mulMul(lstm_71/while/lstm_cell_71/Sigmoid_1:y:0lstm_71_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/while/lstm_cell_71/ReluRelu)lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 lstm_71/while/lstm_cell_71/mul_1Mul&lstm_71/while/lstm_cell_71/Sigmoid:y:0-lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 lstm_71/while/lstm_cell_71/add_1AddV2"lstm_71/while/lstm_cell_71/mul:z:0$lstm_71/while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid)lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_71/while/lstm_cell_71/Relu_1Relu$lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 lstm_71/while/lstm_cell_71/mul_2Mul(lstm_71/while/lstm_cell_71/Sigmoid_2:y:0/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8lstm_71/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_71_while_placeholder_1Alstm_71/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_71/while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_71/while/addAddV2lstm_71_while_placeholderlstm_71/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_71/while/add_1AddV2(lstm_71_while_lstm_71_while_loop_counterlstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_71/while/IdentityIdentitylstm_71/while/add_1:z:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 
lstm_71/while/Identity_1Identity.lstm_71_while_lstm_71_while_maximum_iterations^lstm_71/while/NoOp*
T0*
_output_shapes
: q
lstm_71/while/Identity_2Identitylstm_71/while/add:z:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 
lstm_71/while/Identity_3IdentityBlstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 
lstm_71/while/Identity_4Identity$lstm_71/while/lstm_cell_71/mul_2:z:0^lstm_71/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/while/Identity_5Identity$lstm_71/while/lstm_cell_71/add_1:z:0^lstm_71/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_71/while/NoOpNoOp2^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_71_while_identitylstm_71/while/Identity:output:0"=
lstm_71_while_identity_1!lstm_71/while/Identity_1:output:0"=
lstm_71_while_identity_2!lstm_71/while/Identity_2:output:0"=
lstm_71_while_identity_3!lstm_71/while/Identity_3:output:0"=
lstm_71_while_identity_4!lstm_71/while/Identity_4:output:0"=
lstm_71_while_identity_5!lstm_71/while/Identity_5:output:0"P
%lstm_71_while_lstm_71_strided_slice_1'lstm_71_while_lstm_71_strided_slice_1_0"z
:lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0"|
;lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0"x
9lstm_71_while_lstm_cell_71_matmul_readvariableop_resource;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0"È
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp2d
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp2h
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹A
Ë

lstm_70_while_body_1298734,
(lstm_70_while_lstm_70_while_loop_counter2
.lstm_70_while_lstm_70_while_maximum_iterations
lstm_70_while_placeholder
lstm_70_while_placeholder_1
lstm_70_while_placeholder_2
lstm_70_while_placeholder_3+
'lstm_70_while_lstm_70_strided_slice_1_0g
clstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0:xO
=lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0:xJ
<lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0:x
lstm_70_while_identity
lstm_70_while_identity_1
lstm_70_while_identity_2
lstm_70_while_identity_3
lstm_70_while_identity_4
lstm_70_while_identity_5)
%lstm_70_while_lstm_70_strided_slice_1e
alstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensorK
9lstm_70_while_lstm_cell_70_matmul_readvariableop_resource:xM
;lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource:xH
:lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource:x¢1lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp¢0lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp¢2lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp
?lstm_70/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_70/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensor_0lstm_70_while_placeholderHlstm_70/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¬
0lstm_70/while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp;lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0Ñ
!lstm_70/while/lstm_cell_70/MatMulMatMul8lstm_70/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx°
2lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp=lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0¸
#lstm_70/while/lstm_cell_70/MatMul_1MatMullstm_70_while_placeholder_2:lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxµ
lstm_70/while/lstm_cell_70/addAddV2+lstm_70/while/lstm_cell_70/MatMul:product:0-lstm_70/while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxª
1lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp<lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¾
"lstm_70/while/lstm_cell_70/BiasAddBiasAdd"lstm_70/while/lstm_cell_70/add:z:09lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxl
*lstm_70/while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_70/while/lstm_cell_70/splitSplit3lstm_70/while/lstm_cell_70/split/split_dim:output:0+lstm_70/while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_70/while/lstm_cell_70/SigmoidSigmoid)lstm_70/while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_70/while/lstm_cell_70/Sigmoid_1Sigmoid)lstm_70/while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/while/lstm_cell_70/mulMul(lstm_70/while/lstm_cell_70/Sigmoid_1:y:0lstm_70_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/while/lstm_cell_70/ReluRelu)lstm_70/while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 lstm_70/while/lstm_cell_70/mul_1Mul&lstm_70/while/lstm_cell_70/Sigmoid:y:0-lstm_70/while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 lstm_70/while/lstm_cell_70/add_1AddV2"lstm_70/while/lstm_cell_70/mul:z:0$lstm_70/while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_70/while/lstm_cell_70/Sigmoid_2Sigmoid)lstm_70/while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_70/while/lstm_cell_70/Relu_1Relu$lstm_70/while/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 lstm_70/while/lstm_cell_70/mul_2Mul(lstm_70/while/lstm_cell_70/Sigmoid_2:y:0/lstm_70/while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
2lstm_70/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_70_while_placeholder_1lstm_70_while_placeholder$lstm_70/while/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_70/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_70/while/addAddV2lstm_70_while_placeholderlstm_70/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_70/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_70/while/add_1AddV2(lstm_70_while_lstm_70_while_loop_counterlstm_70/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_70/while/IdentityIdentitylstm_70/while/add_1:z:0^lstm_70/while/NoOp*
T0*
_output_shapes
: 
lstm_70/while/Identity_1Identity.lstm_70_while_lstm_70_while_maximum_iterations^lstm_70/while/NoOp*
T0*
_output_shapes
: q
lstm_70/while/Identity_2Identitylstm_70/while/add:z:0^lstm_70/while/NoOp*
T0*
_output_shapes
: 
lstm_70/while/Identity_3IdentityBlstm_70/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_70/while/NoOp*
T0*
_output_shapes
: 
lstm_70/while/Identity_4Identity$lstm_70/while/lstm_cell_70/mul_2:z:0^lstm_70/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_70/while/Identity_5Identity$lstm_70/while/lstm_cell_70/add_1:z:0^lstm_70/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_70/while/NoOpNoOp2^lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp1^lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp3^lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_70_while_identitylstm_70/while/Identity:output:0"=
lstm_70_while_identity_1!lstm_70/while/Identity_1:output:0"=
lstm_70_while_identity_2!lstm_70/while/Identity_2:output:0"=
lstm_70_while_identity_3!lstm_70/while/Identity_3:output:0"=
lstm_70_while_identity_4!lstm_70/while/Identity_4:output:0"=
lstm_70_while_identity_5!lstm_70/while/Identity_5:output:0"P
%lstm_70_while_lstm_70_strided_slice_1'lstm_70_while_lstm_70_strided_slice_1_0"z
:lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource<lstm_70_while_lstm_cell_70_biasadd_readvariableop_resource_0"|
;lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource=lstm_70_while_lstm_cell_70_matmul_1_readvariableop_resource_0"x
9lstm_70_while_lstm_cell_70_matmul_readvariableop_resource;lstm_70_while_lstm_cell_70_matmul_readvariableop_resource_0"È
alstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensorclstm_70_while_tensorarrayv2read_tensorlistgetitem_lstm_70_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp1lstm_70/while/lstm_cell_70/BiasAdd/ReadVariableOp2d
0lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp0lstm_70/while/lstm_cell_70/MatMul/ReadVariableOp2h
2lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp2lstm_70/while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÚJ

D__inference_lstm_70_layer_call_and_return_conditional_losses_1299319
inputs_0=
+lstm_cell_70_matmul_readvariableop_resource:x?
-lstm_cell_70_matmul_1_readvariableop_resource:x:
,lstm_cell_70_biasadd_readvariableop_resource:x
identity¢#lstm_cell_70/BiasAdd/ReadVariableOp¢"lstm_cell_70/MatMul/ReadVariableOp¢$lstm_cell_70/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_70/MatMul/ReadVariableOpReadVariableOp+lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMulMatMulstrided_slice_2:output:0*lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMul_1MatMulzeros:output:0,lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_70/addAddV2lstm_cell_70/MatMul:product:0lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_70/BiasAddBiasAddlstm_cell_70/add:z:0+lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_70/splitSplit%lstm_cell_70/split/split_dim:output:0lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_70/SigmoidSigmoidlstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_1Sigmoidlstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_70/mulMullstm_cell_70/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_70/ReluRelulstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_1Mullstm_cell_70/Sigmoid:y:0lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_70/add_1AddV2lstm_cell_70/mul:z:0lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_2Sigmoidlstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_70/Relu_1Relulstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_2Mullstm_cell_70/Sigmoid_2:y:0!lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_70_matmul_readvariableop_resource-lstm_cell_70_matmul_1_readvariableop_resource,lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1299235*
condR
while_cond_1299234*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_70/BiasAdd/ReadVariableOp#^lstm_cell_70/MatMul/ReadVariableOp%^lstm_cell_70/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_70/BiasAdd/ReadVariableOp#lstm_cell_70/BiasAdd/ReadVariableOp2H
"lstm_cell_70/MatMul/ReadVariableOp"lstm_cell_70/MatMul/ReadVariableOp2L
$lstm_cell_70/MatMul_1/ReadVariableOp$lstm_cell_70/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

Õ
#__inference__traced_restore_1300786
file_prefix3
!assignvariableop_dense_122_kernel:d/
!assignvariableop_1_dense_122_bias:d5
#assignvariableop_2_dense_123_kernel:d/
!assignvariableop_3_dense_123_bias:@
.assignvariableop_4_lstm_70_lstm_cell_70_kernel:xJ
8assignvariableop_5_lstm_70_lstm_cell_70_recurrent_kernel:x:
,assignvariableop_6_lstm_70_lstm_cell_70_bias:x@
.assignvariableop_7_lstm_71_lstm_cell_71_kernel:xJ
8assignvariableop_8_lstm_71_lstm_cell_71_recurrent_kernel:x:
,assignvariableop_9_lstm_71_lstm_cell_71_bias:x'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: =
+assignvariableop_19_adam_dense_122_kernel_m:d7
)assignvariableop_20_adam_dense_122_bias_m:d=
+assignvariableop_21_adam_dense_123_kernel_m:d7
)assignvariableop_22_adam_dense_123_bias_m:H
6assignvariableop_23_adam_lstm_70_lstm_cell_70_kernel_m:xR
@assignvariableop_24_adam_lstm_70_lstm_cell_70_recurrent_kernel_m:xB
4assignvariableop_25_adam_lstm_70_lstm_cell_70_bias_m:xH
6assignvariableop_26_adam_lstm_71_lstm_cell_71_kernel_m:xR
@assignvariableop_27_adam_lstm_71_lstm_cell_71_recurrent_kernel_m:xB
4assignvariableop_28_adam_lstm_71_lstm_cell_71_bias_m:x=
+assignvariableop_29_adam_dense_122_kernel_v:d7
)assignvariableop_30_adam_dense_122_bias_v:d=
+assignvariableop_31_adam_dense_123_kernel_v:d7
)assignvariableop_32_adam_dense_123_bias_v:H
6assignvariableop_33_adam_lstm_70_lstm_cell_70_kernel_v:xR
@assignvariableop_34_adam_lstm_70_lstm_cell_70_recurrent_kernel_v:xB
4assignvariableop_35_adam_lstm_70_lstm_cell_70_bias_v:xH
6assignvariableop_36_adam_lstm_71_lstm_cell_71_kernel_v:xR
@assignvariableop_37_adam_lstm_71_lstm_cell_71_recurrent_kernel_v:xB
4assignvariableop_38_adam_lstm_71_lstm_cell_71_bias_v:x
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Þ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*
valueúB÷(B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B é
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_dense_122_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_122_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_123_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_123_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp.assignvariableop_4_lstm_70_lstm_cell_70_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_5AssignVariableOp8assignvariableop_5_lstm_70_lstm_cell_70_recurrent_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_lstm_70_lstm_cell_70_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_71_lstm_cell_71_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_71_lstm_cell_71_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_71_lstm_cell_71_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_122_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_122_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_123_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_123_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_lstm_70_lstm_cell_70_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_lstm_70_lstm_cell_70_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_lstm_70_lstm_cell_70_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_lstm_71_lstm_cell_71_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_lstm_71_lstm_cell_71_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_71_lstm_cell_71_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_122_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_122_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_123_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_123_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_lstm_70_lstm_cell_70_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_lstm_70_lstm_cell_70_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_70_lstm_cell_70_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_lstm_71_lstm_cell_71_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_37AssignVariableOp@assignvariableop_37_adam_lstm_71_lstm_cell_71_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_38AssignVariableOp4assignvariableop_38_adam_lstm_71_lstm_cell_71_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ©
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Î 

J__inference_sequential_64_layer_call_and_return_conditional_losses_1298292
lstm_70_input!
lstm_70_1298265:x!
lstm_70_1298267:x
lstm_70_1298269:x!
lstm_71_1298273:x!
lstm_71_1298275:x
lstm_71_1298277:x#
dense_122_1298281:d
dense_122_1298283:d#
dense_123_1298286:d
dense_123_1298288:
identity¢!dense_122/StatefulPartitionedCall¢!dense_123/StatefulPartitionedCall¢"dropout_68/StatefulPartitionedCall¢"dropout_69/StatefulPartitionedCall¢lstm_70/StatefulPartitionedCall¢lstm_71/StatefulPartitionedCall
lstm_70/StatefulPartitionedCallStatefulPartitionedCalllstm_70_inputlstm_70_1298265lstm_70_1298267lstm_70_1298269*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_70_layer_call_and_return_conditional_losses_1298118ó
"dropout_68/StatefulPartitionedCallStatefulPartitionedCall(lstm_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_68_layer_call_and_return_conditional_losses_1297959§
lstm_71/StatefulPartitionedCallStatefulPartitionedCall+dropout_68/StatefulPartitionedCall:output:0lstm_71_1298273lstm_71_1298275lstm_71_1298277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_1297930
"dropout_69/StatefulPartitionedCallStatefulPartitionedCall(lstm_71/StatefulPartitionedCall:output:0#^dropout_68/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_69_layer_call_and_return_conditional_losses_1297769
!dense_122/StatefulPartitionedCallStatefulPartitionedCall+dropout_69/StatefulPartitionedCall:output:0dense_122_1298281dense_122_1298283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_122_layer_call_and_return_conditional_losses_1297682
!dense_123/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0dense_123_1298286dense_123_1298288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_123_layer_call_and_return_conditional_losses_1297699y
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^dense_122/StatefulPartitionedCall"^dense_123/StatefulPartitionedCall#^dropout_68/StatefulPartitionedCall#^dropout_69/StatefulPartitionedCall ^lstm_70/StatefulPartitionedCall ^lstm_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2H
"dropout_68/StatefulPartitionedCall"dropout_68/StatefulPartitionedCall2H
"dropout_69/StatefulPartitionedCall"dropout_69/StatefulPartitionedCall2B
lstm_70/StatefulPartitionedCalllstm_70/StatefulPartitionedCall2B
lstm_71/StatefulPartitionedCalllstm_71/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_70_input
º
È
while_cond_1299234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1299234___redundant_placeholder05
1while_while_cond_1299234___redundant_placeholder15
1while_while_cond_1299234___redundant_placeholder25
1while_while_cond_1299234___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÝB
Ë

lstm_71_while_body_1298575,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3+
'lstm_71_while_lstm_71_strided_slice_1_0g
clstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0:xO
=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0:xJ
<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0:x
lstm_71_while_identity
lstm_71_while_identity_1
lstm_71_while_identity_2
lstm_71_while_identity_3
lstm_71_while_identity_4
lstm_71_while_identity_5)
%lstm_71_while_lstm_71_strided_slice_1e
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorK
9lstm_71_while_lstm_cell_71_matmul_readvariableop_resource:xM
;lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource:xH
:lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource:x¢1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp¢0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp¢2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp
?lstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_71/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0lstm_71_while_placeholderHlstm_71/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0¬
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0Ñ
!lstm_71/while/lstm_cell_71/MatMulMatMul8lstm_71/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx°
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0¸
#lstm_71/while/lstm_cell_71/MatMul_1MatMullstm_71_while_placeholder_2:lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxµ
lstm_71/while/lstm_cell_71/addAddV2+lstm_71/while/lstm_cell_71/MatMul:product:0-lstm_71/while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxª
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¾
"lstm_71/while/lstm_cell_71/BiasAddBiasAdd"lstm_71/while/lstm_cell_71/add:z:09lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxl
*lstm_71/while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_71/while/lstm_cell_71/splitSplit3lstm_71/while/lstm_cell_71/split/split_dim:output:0+lstm_71/while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_71/while/lstm_cell_71/SigmoidSigmoid)lstm_71/while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_71/while/lstm_cell_71/Sigmoid_1Sigmoid)lstm_71/while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/while/lstm_cell_71/mulMul(lstm_71/while/lstm_cell_71/Sigmoid_1:y:0lstm_71_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/while/lstm_cell_71/ReluRelu)lstm_71/while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 lstm_71/while/lstm_cell_71/mul_1Mul&lstm_71/while/lstm_cell_71/Sigmoid:y:0-lstm_71/while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 lstm_71/while/lstm_cell_71/add_1AddV2"lstm_71/while/lstm_cell_71/mul:z:0$lstm_71/while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_71/while/lstm_cell_71/Sigmoid_2Sigmoid)lstm_71/while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_71/while/lstm_cell_71/Relu_1Relu$lstm_71/while/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 lstm_71/while/lstm_cell_71/mul_2Mul(lstm_71/while/lstm_cell_71/Sigmoid_2:y:0/lstm_71/while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8lstm_71/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_71/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_71_while_placeholder_1Alstm_71/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_71/while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_71/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_71/while/addAddV2lstm_71_while_placeholderlstm_71/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_71/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_71/while/add_1AddV2(lstm_71_while_lstm_71_while_loop_counterlstm_71/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_71/while/IdentityIdentitylstm_71/while/add_1:z:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 
lstm_71/while/Identity_1Identity.lstm_71_while_lstm_71_while_maximum_iterations^lstm_71/while/NoOp*
T0*
_output_shapes
: q
lstm_71/while/Identity_2Identitylstm_71/while/add:z:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 
lstm_71/while/Identity_3IdentityBlstm_71/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_71/while/NoOp*
T0*
_output_shapes
: 
lstm_71/while/Identity_4Identity$lstm_71/while/lstm_cell_71/mul_2:z:0^lstm_71/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_71/while/Identity_5Identity$lstm_71/while/lstm_cell_71/add_1:z:0^lstm_71/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_71/while/NoOpNoOp2^lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1^lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp3^lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_71_while_identitylstm_71/while/Identity:output:0"=
lstm_71_while_identity_1!lstm_71/while/Identity_1:output:0"=
lstm_71_while_identity_2!lstm_71/while/Identity_2:output:0"=
lstm_71_while_identity_3!lstm_71/while/Identity_3:output:0"=
lstm_71_while_identity_4!lstm_71/while/Identity_4:output:0"=
lstm_71_while_identity_5!lstm_71/while/Identity_5:output:0"P
%lstm_71_while_lstm_71_strided_slice_1'lstm_71_while_lstm_71_strided_slice_1_0"z
:lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource<lstm_71_while_lstm_cell_71_biasadd_readvariableop_resource_0"|
;lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource=lstm_71_while_lstm_cell_71_matmul_1_readvariableop_resource_0"x
9lstm_71_while_lstm_cell_71_matmul_readvariableop_resource;lstm_71_while_lstm_cell_71_matmul_readvariableop_resource_0"È
alstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensorclstm_71_while_tensorarrayv2read_tensorlistgetitem_lstm_71_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp1lstm_71/while/lstm_cell_71/BiasAdd/ReadVariableOp2d
0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp0lstm_71/while/lstm_cell_71/MatMul/ReadVariableOp2h
2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp2lstm_71/while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
$
ä
while_body_1297269
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_71_1297293_0:x.
while_lstm_cell_71_1297295_0:x*
while_lstm_cell_71_1297297_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_71_1297293:x,
while_lstm_cell_71_1297295:x(
while_lstm_cell_71_1297297:x¢*while/lstm_cell_71/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0·
*while/lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_71_1297293_0while_lstm_cell_71_1297295_0while_lstm_cell_71_1297297_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1297209r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_71/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity3while/lstm_cell_71/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_71/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_71/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_71_1297293while_lstm_cell_71_1297293_0":
while_lstm_cell_71_1297295while_lstm_cell_71_1297295_0":
while_lstm_cell_71_1297297while_lstm_cell_71_1297297_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_71/StatefulPartitionedCall*while/lstm_cell_71/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ö
³
)__inference_lstm_71_layer_call_fn_1299676

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_1297930o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
9

D__inference_lstm_71_layer_call_and_return_conditional_losses_1297146

inputs&
lstm_cell_71_1297062:x&
lstm_cell_71_1297064:x"
lstm_cell_71_1297066:x
identity¢$lstm_cell_71/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskù
$lstm_cell_71/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_71_1297062lstm_cell_71_1297064lstm_cell_71_1297066*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1297061n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_71_1297062lstm_cell_71_1297064lstm_cell_71_1297066*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1297076*
condR
while_cond_1297075*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_71/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_71/StatefulPartitionedCall$lstm_cell_71/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

µ
)__inference_lstm_71_layer_call_fn_1299643
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_71_layer_call_and_return_conditional_losses_1297146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ú
e
G__inference_dropout_69_layer_call_and_return_conditional_losses_1297669

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÅK

D__inference_lstm_71_layer_call_and_return_conditional_losses_1299821
inputs_0=
+lstm_cell_71_matmul_readvariableop_resource:x?
-lstm_cell_71_matmul_1_readvariableop_resource:x:
,lstm_cell_71_biasadd_readvariableop_resource:x
identity¢#lstm_cell_71/BiasAdd/ReadVariableOp¢"lstm_cell_71/MatMul/ReadVariableOp¢$lstm_cell_71/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_71/MatMul/ReadVariableOpReadVariableOp+lstm_cell_71_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMulMatMulstrided_slice_2:output:0*lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_71_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_71/MatMul_1MatMulzeros:output:0,lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_71/addAddV2lstm_cell_71/MatMul:product:0lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_71_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_71/BiasAddBiasAddlstm_cell_71/add:z:0+lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_71/splitSplit%lstm_cell_71/split/split_dim:output:0lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_71/SigmoidSigmoidlstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_1Sigmoidlstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_71/mulMullstm_cell_71/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_71/ReluRelulstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_1Mullstm_cell_71/Sigmoid:y:0lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_71/add_1AddV2lstm_cell_71/mul:z:0lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_71/Sigmoid_2Sigmoidlstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_71/Relu_1Relulstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_71/mul_2Mullstm_cell_71/Sigmoid_2:y:0!lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_71_matmul_readvariableop_resource-lstm_cell_71_matmul_1_readvariableop_resource,lstm_cell_71_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1299736*
condR
while_cond_1299735*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_71/BiasAdd/ReadVariableOp#^lstm_cell_71/MatMul/ReadVariableOp%^lstm_cell_71/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_71/BiasAdd/ReadVariableOp#lstm_cell_71/BiasAdd/ReadVariableOp2H
"lstm_cell_71/MatMul/ReadVariableOp"lstm_cell_71/MatMul/ReadVariableOp2L
$lstm_cell_71/MatMul_1/ReadVariableOp$lstm_cell_71/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ì

I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1297209

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
þ
³
)__inference_lstm_70_layer_call_fn_1299033

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_70_layer_call_and_return_conditional_losses_1298118s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
8
Ë
while_body_1297413
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_70_matmul_readvariableop_resource_0:xG
5while_lstm_cell_70_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_70_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_70_matmul_readvariableop_resource:xE
3while_lstm_cell_70_matmul_1_readvariableop_resource:x@
2while_lstm_cell_70_biasadd_readvariableop_resource:x¢)while/lstm_cell_70/BiasAdd/ReadVariableOp¢(while/lstm_cell_70/MatMul/ReadVariableOp¢*while/lstm_cell_70/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_70/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_70_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_70/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_70_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_70/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_70/addAddV2#while/lstm_cell_70/MatMul:product:0%while/lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_70_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_70/BiasAddBiasAddwhile/lstm_cell_70/add:z:01while/lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_70/splitSplit+while/lstm_cell_70/split/split_dim:output:0#while/lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_70/SigmoidSigmoid!while/lstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_1Sigmoid!while/lstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mulMul while/lstm_cell_70/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_70/ReluRelu!while/lstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_1Mulwhile/lstm_cell_70/Sigmoid:y:0%while/lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/add_1AddV2while/lstm_cell_70/mul:z:0while/lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_70/Sigmoid_2Sigmoid!while/lstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_70/Relu_1Reluwhile/lstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_70/mul_2Mul while/lstm_cell_70/Sigmoid_2:y:0'while/lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_70/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_70/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_70/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_70/BiasAdd/ReadVariableOp)^while/lstm_cell_70/MatMul/ReadVariableOp+^while/lstm_cell_70/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_70_biasadd_readvariableop_resource4while_lstm_cell_70_biasadd_readvariableop_resource_0"l
3while_lstm_cell_70_matmul_1_readvariableop_resource5while_lstm_cell_70_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_70_matmul_readvariableop_resource3while_lstm_cell_70_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_70/BiasAdd/ReadVariableOp)while/lstm_cell_70/BiasAdd/ReadVariableOp2T
(while/lstm_cell_70/MatMul/ReadVariableOp(while/lstm_cell_70/MatMul/ReadVariableOp2X
*while/lstm_cell_70/MatMul_1/ReadVariableOp*while/lstm_cell_70/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
9
Ë
while_body_1297571
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_71_matmul_readvariableop_resource_0:xG
5while_lstm_cell_71_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_71_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_71_matmul_readvariableop_resource:xE
3while_lstm_cell_71_matmul_1_readvariableop_resource:x@
2while_lstm_cell_71_biasadd_readvariableop_resource:x¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
9
Ë
while_body_1300171
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_71_matmul_readvariableop_resource_0:xG
5while_lstm_cell_71_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_71_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_71_matmul_readvariableop_resource:xE
3while_lstm_cell_71_matmul_1_readvariableop_resource:x@
2while_lstm_cell_71_biasadd_readvariableop_resource:x¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
J

D__inference_lstm_70_layer_call_and_return_conditional_losses_1299605

inputs=
+lstm_cell_70_matmul_readvariableop_resource:x?
-lstm_cell_70_matmul_1_readvariableop_resource:x:
,lstm_cell_70_biasadd_readvariableop_resource:x
identity¢#lstm_cell_70/BiasAdd/ReadVariableOp¢"lstm_cell_70/MatMul/ReadVariableOp¢$lstm_cell_70/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_70/MatMul/ReadVariableOpReadVariableOp+lstm_cell_70_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMulMatMulstrided_slice_2:output:0*lstm_cell_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
$lstm_cell_70/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_70_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0
lstm_cell_70/MatMul_1MatMulzeros:output:0,lstm_cell_70/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell_70/addAddV2lstm_cell_70/MatMul:product:0lstm_cell_70/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#lstm_cell_70/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_70_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0
lstm_cell_70/BiasAddBiasAddlstm_cell_70/add:z:0+lstm_cell_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx^
lstm_cell_70/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_70/splitSplit%lstm_cell_70/split/split_dim:output:0lstm_cell_70/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_70/SigmoidSigmoidlstm_cell_70/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_1Sigmoidlstm_cell_70/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_70/mulMullstm_cell_70/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_70/ReluRelulstm_cell_70/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_1Mullstm_cell_70/Sigmoid:y:0lstm_cell_70/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_70/add_1AddV2lstm_cell_70/mul:z:0lstm_cell_70/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_70/Sigmoid_2Sigmoidlstm_cell_70/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_70/Relu_1Relulstm_cell_70/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_70/mul_2Mullstm_cell_70/Sigmoid_2:y:0!lstm_cell_70/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_70_matmul_readvariableop_resource-lstm_cell_70_matmul_1_readvariableop_resource,lstm_cell_70_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1299521*
condR
while_cond_1299520*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_70/BiasAdd/ReadVariableOp#^lstm_cell_70/MatMul/ReadVariableOp%^lstm_cell_70/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_70/BiasAdd/ReadVariableOp#lstm_cell_70/BiasAdd/ReadVariableOp2H
"lstm_cell_70/MatMul/ReadVariableOp"lstm_cell_70/MatMul/ReadVariableOp2L
$lstm_cell_70/MatMul_1/ReadVariableOp$lstm_cell_70/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
ô
.__inference_lstm_cell_70_layer_call_fn_1300357

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1296857o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
9
Ë
while_body_1300026
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_71_matmul_readvariableop_resource_0:xG
5while_lstm_cell_71_matmul_1_readvariableop_resource_0:xB
4while_lstm_cell_71_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_71_matmul_readvariableop_resource:xE
3while_lstm_cell_71_matmul_1_readvariableop_resource:x@
2while_lstm_cell_71_biasadd_readvariableop_resource:x¢)while/lstm_cell_71/BiasAdd/ReadVariableOp¢(while/lstm_cell_71/MatMul/ReadVariableOp¢*while/lstm_cell_71/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_71/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_71_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0¹
while/lstm_cell_71/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
*while/lstm_cell_71/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_71_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0 
while/lstm_cell_71/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_71/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
while/lstm_cell_71/addAddV2#while/lstm_cell_71/MatMul:product:0%while/lstm_cell_71/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
)while/lstm_cell_71/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_71_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0¦
while/lstm_cell_71/BiasAddBiasAddwhile/lstm_cell_71/add:z:01while/lstm_cell_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
"while/lstm_cell_71/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_71/splitSplit+while/lstm_cell_71/split/split_dim:output:0#while/lstm_cell_71/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_71/SigmoidSigmoid!while/lstm_cell_71/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_1Sigmoid!while/lstm_cell_71/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mulMul while/lstm_cell_71/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_71/ReluRelu!while/lstm_cell_71/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_1Mulwhile/lstm_cell_71/Sigmoid:y:0%while/lstm_cell_71/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/add_1AddV2while/lstm_cell_71/mul:z:0while/lstm_cell_71/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_71/Sigmoid_2Sigmoid!while/lstm_cell_71/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_71/Relu_1Reluwhile/lstm_cell_71/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_71/mul_2Mul while/lstm_cell_71/Sigmoid_2:y:0'while/lstm_cell_71/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_71/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_71/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_71/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_71/BiasAdd/ReadVariableOp)^while/lstm_cell_71/MatMul/ReadVariableOp+^while/lstm_cell_71/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_71_biasadd_readvariableop_resource4while_lstm_cell_71_biasadd_readvariableop_resource_0"l
3while_lstm_cell_71_matmul_1_readvariableop_resource5while_lstm_cell_71_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_71_matmul_readvariableop_resource3while_lstm_cell_71_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_71/BiasAdd/ReadVariableOp)while/lstm_cell_71/BiasAdd/ReadVariableOp2T
(while/lstm_cell_71/MatMul/ReadVariableOp(while/lstm_cell_71/MatMul/ReadVariableOp2X
*while/lstm_cell_71/MatMul_1/ReadVariableOp*while/lstm_cell_71/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


è
lstm_71_while_cond_1298574,
(lstm_71_while_lstm_71_while_loop_counter2
.lstm_71_while_lstm_71_while_maximum_iterations
lstm_71_while_placeholder
lstm_71_while_placeholder_1
lstm_71_while_placeholder_2
lstm_71_while_placeholder_3.
*lstm_71_while_less_lstm_71_strided_slice_1E
Alstm_71_while_lstm_71_while_cond_1298574___redundant_placeholder0E
Alstm_71_while_lstm_71_while_cond_1298574___redundant_placeholder1E
Alstm_71_while_lstm_71_while_cond_1298574___redundant_placeholder2E
Alstm_71_while_lstm_71_while_cond_1298574___redundant_placeholder3
lstm_71_while_identity

lstm_71/while/LessLesslstm_71_while_placeholder*lstm_71_while_less_lstm_71_strided_slice_1*
T0*
_output_shapes
: [
lstm_71/while/IdentityIdentitylstm_71/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_71_while_identitylstm_71/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ª

ø
/__inference_sequential_64_layer_call_fn_1298375

inputs
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:d
	unknown_6:d
	unknown_7:d
	unknown_8:
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298184o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¼
serving_default¨
K
lstm_70_input:
serving_default_lstm_70_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_1230
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¯³

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
¼
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
Ú
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator
'cell
(
state_spec"
_tf_keras_rnn_layer
¼
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/_random_generator"
_tf_keras_layer
»
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
»
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias"
_tf_keras_layer
f
@0
A1
B2
C3
D4
E5
66
77
>8
?9"
trackable_list_wrapper
f
@0
A1
B2
C3
D4
E5
66
77
>8
?9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ñ
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32
/__inference_sequential_64_layer_call_fn_1297729
/__inference_sequential_64_layer_call_fn_1298350
/__inference_sequential_64_layer_call_fn_1298375
/__inference_sequential_64_layer_call_fn_1298232¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
Ý
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32ò
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298675
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298989
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298262
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298292¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
ÓBÐ
"__inference__wrapped_model_1296644lstm_70_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_rate6mÂ7mÃ>mÄ?mÅ@mÆAmÇBmÈCmÉDmÊEmË6vÌ7vÍ>vÎ?vÏ@vÐAvÑBvÒCvÓDvÔEvÕ"
	optimizer
,
Xserving_default"
signature_map
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Ystates
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
_trace_0
`trace_1
atrace_2
btrace_32
)__inference_lstm_70_layer_call_fn_1299000
)__inference_lstm_70_layer_call_fn_1299011
)__inference_lstm_70_layer_call_fn_1299022
)__inference_lstm_70_layer_call_fn_1299033Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z_trace_0z`trace_1zatrace_2zbtrace_3
Ú
ctrace_0
dtrace_1
etrace_2
ftrace_32ï
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299176
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299319
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299462
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299605Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zctrace_0zdtrace_1zetrace_2zftrace_3
"
_generic_user_object
ø
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
m_random_generator
n
state_size

@kernel
Arecurrent_kernel
Bbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
É
ttrace_0
utrace_12
,__inference_dropout_68_layer_call_fn_1299610
,__inference_dropout_68_layer_call_fn_1299615³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zttrace_0zutrace_1
ÿ
vtrace_0
wtrace_12È
G__inference_dropout_68_layer_call_and_return_conditional_losses_1299620
G__inference_dropout_68_layer_call_and_return_conditional_losses_1299632³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zvtrace_0zwtrace_1
"
_generic_user_object
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

xstates
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ò
~trace_0
trace_1
trace_2
trace_32
)__inference_lstm_71_layer_call_fn_1299643
)__inference_lstm_71_layer_call_fn_1299654
)__inference_lstm_71_layer_call_fn_1299665
)__inference_lstm_71_layer_call_fn_1299676Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z~trace_0ztrace_1ztrace_2ztrace_3
â
trace_0
trace_1
trace_2
trace_32ï
D__inference_lstm_71_layer_call_and_return_conditional_losses_1299821
D__inference_lstm_71_layer_call_and_return_conditional_losses_1299966
D__inference_lstm_71_layer_call_and_return_conditional_losses_1300111
D__inference_lstm_71_layer_call_and_return_conditional_losses_1300256Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
"
_generic_user_object

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

state_size

Ckernel
Drecurrent_kernel
Ebias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Í
trace_0
trace_12
,__inference_dropout_69_layer_call_fn_1300261
,__inference_dropout_69_layer_call_fn_1300266³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12È
G__inference_dropout_69_layer_call_and_return_conditional_losses_1300271
G__inference_dropout_69_layer_call_and_return_conditional_losses_1300283³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_dense_122_layer_call_fn_1300292¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_dense_122_layer_call_and_return_conditional_losses_1300303¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
": d2dense_122/kernel
:d2dense_122/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ñ
£trace_02Ò
+__inference_dense_123_layer_call_fn_1300312¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z£trace_0

¤trace_02í
F__inference_dense_123_layer_call_and_return_conditional_losses_1300323¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¤trace_0
": d2dense_123/kernel
:2dense_123/bias
-:+x2lstm_70/lstm_cell_70/kernel
7:5x2%lstm_70/lstm_cell_70/recurrent_kernel
':%x2lstm_70/lstm_cell_70/bias
-:+x2lstm_71/lstm_cell_71/kernel
7:5x2%lstm_71/lstm_cell_71/recurrent_kernel
':%x2lstm_71/lstm_cell_71/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
0
¥0
¦1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_64_layer_call_fn_1297729lstm_70_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bý
/__inference_sequential_64_layer_call_fn_1298350inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bý
/__inference_sequential_64_layer_call_fn_1298375inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
/__inference_sequential_64_layer_call_fn_1298232lstm_70_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298675inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298989inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¢B
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298262lstm_70_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¢B
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298292lstm_70_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÒBÏ
%__inference_signature_wrapper_1298325lstm_70_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_lstm_70_layer_call_fn_1299000inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_70_layer_call_fn_1299011inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_70_layer_call_fn_1299022inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_70_layer_call_fn_1299033inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299176inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299319inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299462inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299605inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
²
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
Û
¬trace_0
­trace_12 
.__inference_lstm_cell_70_layer_call_fn_1300340
.__inference_lstm_cell_70_layer_call_fn_1300357½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¬trace_0z­trace_1

®trace_0
¯trace_12Ö
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1300389
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1300421½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0z¯trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ñBî
,__inference_dropout_68_layer_call_fn_1299610inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñBî
,__inference_dropout_68_layer_call_fn_1299615inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_dropout_68_layer_call_and_return_conditional_losses_1299620inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_dropout_68_layer_call_and_return_conditional_losses_1299632inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_lstm_71_layer_call_fn_1299643inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_71_layer_call_fn_1299654inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_71_layer_call_fn_1299665inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_lstm_71_layer_call_fn_1299676inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_71_layer_call_and_return_conditional_losses_1299821inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
D__inference_lstm_71_layer_call_and_return_conditional_losses_1299966inputs/0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_71_layer_call_and_return_conditional_losses_1300111inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
D__inference_lstm_71_layer_call_and_return_conditional_losses_1300256inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Û
µtrace_0
¶trace_12 
.__inference_lstm_cell_71_layer_call_fn_1300438
.__inference_lstm_cell_71_layer_call_fn_1300455½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zµtrace_0z¶trace_1

·trace_0
¸trace_12Ö
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1300487
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1300519½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0z¸trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ñBî
,__inference_dropout_69_layer_call_fn_1300261inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñBî
,__inference_dropout_69_layer_call_fn_1300266inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_dropout_69_layer_call_and_return_conditional_losses_1300271inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_dropout_69_layer_call_and_return_conditional_losses_1300283inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_122_layer_call_fn_1300292inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_122_layer_call_and_return_conditional_losses_1300303inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_123_layer_call_fn_1300312inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_123_layer_call_and_return_conditional_losses_1300323inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
¹	variables
º	keras_api

»total

¼count"
_tf_keras_metric
c
½	variables
¾	keras_api

¿total

Àcount
Á
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_lstm_cell_70_layer_call_fn_1300340inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
.__inference_lstm_cell_70_layer_call_fn_1300357inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1300389inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1300421inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_lstm_cell_71_layer_call_fn_1300438inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
.__inference_lstm_cell_71_layer_call_fn_1300455inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1300487inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬B©
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1300519inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
»0
¼1"
trackable_list_wrapper
.
¹	variables"
_generic_user_object
:  (2total
:  (2count
0
¿0
À1"
trackable_list_wrapper
.
½	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
':%d2Adam/dense_122/kernel/m
!:d2Adam/dense_122/bias/m
':%d2Adam/dense_123/kernel/m
!:2Adam/dense_123/bias/m
2:0x2"Adam/lstm_70/lstm_cell_70/kernel/m
<::x2,Adam/lstm_70/lstm_cell_70/recurrent_kernel/m
,:*x2 Adam/lstm_70/lstm_cell_70/bias/m
2:0x2"Adam/lstm_71/lstm_cell_71/kernel/m
<::x2,Adam/lstm_71/lstm_cell_71/recurrent_kernel/m
,:*x2 Adam/lstm_71/lstm_cell_71/bias/m
':%d2Adam/dense_122/kernel/v
!:d2Adam/dense_122/bias/v
':%d2Adam/dense_123/kernel/v
!:2Adam/dense_123/bias/v
2:0x2"Adam/lstm_70/lstm_cell_70/kernel/v
<::x2,Adam/lstm_70/lstm_cell_70/recurrent_kernel/v
,:*x2 Adam/lstm_70/lstm_cell_70/bias/v
2:0x2"Adam/lstm_71/lstm_cell_71/kernel/v
<::x2,Adam/lstm_71/lstm_cell_71/recurrent_kernel/v
,:*x2 Adam/lstm_71/lstm_cell_71/bias/v¥
"__inference__wrapped_model_1296644
@ABCDE67>?:¢7
0¢-
+(
lstm_70_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_123# 
	dense_123ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_122_layer_call_and_return_conditional_losses_1300303\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ~
+__inference_dense_122_layer_call_fn_1300292O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿd¦
F__inference_dense_123_layer_call_and_return_conditional_losses_1300323\>?/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_123_layer_call_fn_1300312O>?/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ¯
G__inference_dropout_68_layer_call_and_return_conditional_losses_1299620d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¯
G__inference_dropout_68_layer_call_and_return_conditional_losses_1299632d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dropout_68_layer_call_fn_1299610W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_dropout_68_layer_call_fn_1299615W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dropout_69_layer_call_and_return_conditional_losses_1300271\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 §
G__inference_dropout_69_layer_call_and_return_conditional_losses_1300283\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dropout_69_layer_call_fn_1300261O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_dropout_69_layer_call_fn_1300266O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÓ
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299176@ABO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ó
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299319@ABO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¹
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299462q@AB?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¹
D__inference_lstm_70_layer_call_and_return_conditional_losses_1299605q@AB?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ª
)__inference_lstm_70_layer_call_fn_1299000}@ABO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
)__inference_lstm_70_layer_call_fn_1299011}@ABO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)__inference_lstm_70_layer_call_fn_1299022d@AB?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_lstm_70_layer_call_fn_1299033d@AB?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÅ
D__inference_lstm_71_layer_call_and_return_conditional_losses_1299821}CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
D__inference_lstm_71_layer_call_and_return_conditional_losses_1299966}CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
D__inference_lstm_71_layer_call_and_return_conditional_losses_1300111mCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
D__inference_lstm_71_layer_call_and_return_conditional_losses_1300256mCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_lstm_71_layer_call_fn_1299643pCDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_lstm_71_layer_call_fn_1299654pCDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_lstm_71_layer_call_fn_1299665`CDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_lstm_71_layer_call_fn_1299676`CDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿË
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1300389ý@AB¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 Ë
I__inference_lstm_cell_70_layer_call_and_return_conditional_losses_1300421ý@AB¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
  
.__inference_lstm_cell_70_layer_call_fn_1300340í@AB¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ 
.__inference_lstm_cell_70_layer_call_fn_1300357í@AB¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿË
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1300487ýCDE¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 Ë
I__inference_lstm_cell_71_layer_call_and_return_conditional_losses_1300519ýCDE¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
  
.__inference_lstm_cell_71_layer_call_fn_1300438íCDE¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ 
.__inference_lstm_cell_71_layer_call_fn_1300455íCDE¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÅ
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298262w
@ABCDE67>?B¢?
8¢5
+(
lstm_70_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298292w
@ABCDE67>?B¢?
8¢5
+(
lstm_70_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298675p
@ABCDE67>?;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
J__inference_sequential_64_layer_call_and_return_conditional_losses_1298989p
@ABCDE67>?;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_sequential_64_layer_call_fn_1297729j
@ABCDE67>?B¢?
8¢5
+(
lstm_70_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_64_layer_call_fn_1298232j
@ABCDE67>?B¢?
8¢5
+(
lstm_70_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_64_layer_call_fn_1298350c
@ABCDE67>?;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_64_layer_call_fn_1298375c
@ABCDE67>?;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿº
%__inference_signature_wrapper_1298325
@ABCDE67>?K¢H
¢ 
Aª>
<
lstm_70_input+(
lstm_70_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_123# 
	dense_123ÿÿÿÿÿÿÿÿÿ