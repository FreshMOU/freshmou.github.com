---
layout: post
title: ARM内嵌汇编
date: 2018-11-07
tags: ARM
---

### 前言

GCC提供了一个允许程序员将内嵌汇编语言嵌入C/C++代码的接口。不同于汇编中编写完整的函数，内嵌汇编允许程序员只写需要用汇编代码编写的代码组件。

### 内嵌汇编

内嵌汇编允许程序员将C/C++变量和指针指定为通常取寄存器或基于寄存器的地址指令的目标和操作数。在这种情况下，编译器将变量替换为寄存器，并自动添加任何附加的加载立即数，或加载和存储临时分配所需寄存器所需的指令。编程器也可以直接使用寄存器。在这种情况下，程序员必须在clobber列表中指定受影响的寄存器，以便编译器可以确保寄存器状态不会收到内嵌指令的不利影响。

内嵌程序语法如下：

```
asm [volatile](code : output operand list : input operand list : clobber list);
```

#### 向量模式和标量模式

操作可以指定对多个数据源进行相同的处理。若控制寄存器中LEN长度为4，则单个向量加法指令在4个地址上执行。在ARM术语中，这被称为向量浮点运算。在ARMv5架构中引入了向量浮点(VFP)扩展执行短向量指令以加速浮点运算。源和目标寄存器可以是标量操作的单个寄存器，也可以是两个序列(8个寄存器)的矢量操作。因为SIMD操作比VFP操作更有效地执行向量计算，矢量模式操作在ARMv7上被弃用，取而代之的是NEON技术在宽寄存器上执行多种操作。浮点和NEON的操作使用共通的通用寄存器。

```
  VADD.F32 S24, S8, S16
  // four operations occur
  // S24 = S8 +S16
  // S25 = S9 +S17
  // S26 = S10 +S18
  // S27 = S11 +S20
```


### ARM限定符


| Constraint	| Usage in ARM state	| Usage in Thumb state
| : ------- : | : -------- : | :----:|
|f	|Floating point registers f0 .. f7	| 
|h	|	|Registers r8..r15|
|G	|Immediate floating point constant	|
|H	|Same a G, but negated|
|I	|Immediate value in data processing instructionse.g. ORR R0, R0, #operand|Constant in the range 0 .. 255 e.g. SWI operand
|J	|Indexing constants -4095 .. 4095 e.g. LDR R1, [PC, #operand] | Constant in the range -255 .. -1 e.g. SUB R0, R0, #operand
|K	|Same as I, but inverted |Same as I, but shifted
|L	|Same as I, but negated |Constant in the range -7 .. 7 e.g. SUB R0, R1, #operand
|l	|Same as r	|Registers r0..r7 e.g. PUSH operand
|M |Constant in the range of 0 .. 32 or a power of 2 e.g. MOV R2, R1, ROR #operand	|Constant that is a multiple of 4 in the range of 0 .. 1020 e.g. ADD R0, SP, #operand
|m	|Any valid memory address
|N ||Constant in the range of 0 .. 31 e.g. LSL R0, R1, #operand
|O|	|Constant that is a multiple of 4 in the range of -508 .. 508 e.g. ADD SP, #operand
|r|	General register r0 .. r15 e.g. SUB operand1, operand2, operand3
|w|	Vector floating point registers s0 .. s31
|X|	Any operand



|Modifier|	Specifies
|:---:|:---:|
|=|	Write-only operand, usually used for all output operands
|+|	Read-write operand, must be listed as an output operand
|&|	A register that should be used for output only

### 典型代码

下列代码为一个循环代码，其中%w0表示输出部分的第一个数，"=r","0"都是限定符，此处的"0"即表示%1指向的是%0。%+数字(%0,%1...)即在代码后的表示第几个，此处%0即nn，%1也是nn。

```c
int main(void)
{
    int nn = 10;
    asm volatile ( 
        "0:                         \n"
        "subs   %w0, %w0, #1        \n"
        "bne    0b                  \n"
        : "=r"(nn)
        : "0"(nn)
        : 
    );
 
    prinft("nn=%d\n", nn);
    return 0;
}
```