# 编程规范

总结c/c++编程规范，不断精进coding手艺。  

[google开源项目风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/)  
[c++参考手册](https://zh.cppreference.com/w/cpp)
[cplusplus](https://cplusplus.com/)

## 头文件
* 头文件应该是自给自足的（self-contained），也就是说一个头文件应该是可以独立编译的  
比如my_class.h，其中依赖了std::string，但没有包含<string\>，单独编译该头文件无法编译

```my_class.h
#ifndef MY_CLASS_H_
#define MY_CLASS_H_
class MyClass {
 public:
  MyClass();
  const std::string& value();
 private:
  std::string value_;
};
#endif
```

* 所有头文件都应该有#define保护来防止被多重包含，命名格式：
> <PROJECT\>\_<PATH\>\_<FILE\>\_H\_
也可以用`#pragma once`

* 前置声明
> 定义：类、函数和模板的纯粹声明，没有伴随其定义

```foo.h
#pragma once
class bar;
class foo
{
public:
	Bar getBar();
private
	Bar* _bar;
};
```
> 前置声明要求：其声明的类是文件所声明的类的数据成员时，是指针成员或引用成员，而不能是对象成员。
尽量避免使用前置声明；当两个类相互包含头文件时无法通过编译是，必须使用。

* \#include的路径及顺序
头文件包含次序如下：
> 1. 关联头文件（即该cpp对应的头文件）  
> 2. C系统文件
> 3. C++系统文件
> 4. 其他库的`.h`文件
> 5. 本项目内`.h`文件  

举例：
```
#include "foo/public/fooserver.h" // 优先位置

#include <sys/types.h>
#include <unistd.h>

#include <hash_map>
#include <vector>

#include "base/basictypes.h"
#include "base/commandlineflags.h"
#include "foo/public/bar.h"
```

## 作用域

* 命名空间的使用方式  
```
具名命名空间
namespace mynamespace {

} // namespace mynamespace  --注意注释

匿名命名空间
namespace {

} // namespace  --注意注释
```

* 不应该使用using指示引入整个命名空间的标识符
* 不要在头文件中使用命名空间别名，命名空间别名一般应只在一个局部范围中使用
* 在`.cc`或`.cpp`中定义一个不需要外部引用的函数或变量时，可以将它们放在匿名命名空间或声明为`static`，尽量不要用裸的全局函数；但是不要在`.h`文件中这么做
* 禁止定义静态储存周期为非POD变量，静态生存周期的对象，即包括了全局变量，静态变量，静态类成员变量和函数静态变量，都必须是原生数据类型 (POD : Plain Old Data): 即 `int`, `char` 和 `float`, 以及 POD 类型的指针、数组和结构体。

## 类

* 不要在构造函数中调用虚函数，不要有过多的逻辑，如果需要non-trivial的初始化，可以考虑使用明确的init()
* 在类型定义中，类型转换运算符和单参数构造函数都应当用`explicit`进行标记；其中类型转换运算函数原型为：`operate Type()`，普通数据类型到类类型之间的转换调用转换构造函数，而类类型到普通数据类型的转换调用类型转换函数。
* 如果让类型可拷贝, 一定要同时给出拷贝构造函数和赋值操作的定义, 反之亦然. 如果让类型可拷贝, 同时移动操作的效率高于拷贝操作, 那么就把移动的两个操作 (移动构造函数和赋值操作)也给出定义
* 如果不需要拷贝/移动操作，显式地通过在`public`域使用`= delete`禁用之
```
// MyClass is neither copyable nor movable.
MyClass(const MyClass&) = delete;
MyClass& operator=(const MyClass&) = delete;
```
* 析构函数声明为 virtual. 如果你的类有虚函数, 则析构函数也应该为虚函数
* 对于重载的虚函数或虚析构函数, 使用`override`, 或 (较不常用的)`final`关键字显式地进行标记
* 声明次序: `public` -> `protected` -> `private`，要缩进一个空格

## 函数
* 函数参数顺序：输入参数在先，后跟输出参数
* 函数应尽量短小，单一职责
* 需要加上const的地方尽量减少const，如输入参数，不会修改成员变量的const函数
* 函数重载和缺省参数，虚函数不允许使用缺省参数;如果函数和声明分开, 缺省参数要写在函数的声明位置, 而函数定义部分不要重复指定；靠右原则, 如果函数的某个参数具有缺省值, 那么该参数右侧的所有参数都必须带有缺省
```
void func(int i = 0, int j){...} //erro  
void func(int i, int j = 0){...} //true
```

## 智能指针
* `std::unique_ptr`对动态分配出的对象有独一无二的的所有权，当其离开作用域时，对象就会被销毁；`std::unique_ptr`不能被复制，但可以把它移动给新的`std::unique_ptr`
* 如何没有好的理由，则不要使用共享所有权；如果需要使用共享所有权，建议使用`std::shared_ptr<>`

## Cpplint
用于检查风格错误

## 其他C++特性

### 右值引用和移动语义-[参考](https://blog.csdn.net/baidu_41388533/article/details/106468153?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.opensearchhbase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.opensearchhbase)
* 目的是为了使用临时变量，减少大数据类的拷贝开销，而直接采用移动的方式；因此一般当类包含复杂数据成员时，需要定义移动构造和赋值函数
* 右值引用是一种只能绑定到临时对象的引用的一种，只在定义移动构造函数和移动赋值函数操作时使用，右值引用可以延长右值的生命周期
* `std::move`用于指示对象t可以被移动，即允许从t到另一个对象的有效率资源传递，[参考](https://en.cppreference.com/w/cpp/utility/move)
* 常见创建临时对象的三种情况：
> 1. 以值或const引用的方式给函数传参
> 2. 让函数调用成功而发生的类型转换
> 3. 函数需要返回对象时
* 左值和右值的概念
> 1. 左值是可以放在赋值号左边可以被赋值的值；左值必须要在内存中有实体；
> 2. 右值当在赋值号右边取出值赋给其他变量的值；右值可以在内存也可以在CPU寄存器。
> 3. 一个对象被用作右值时，使用的是它的内容(值)，被当作左值时，使用的是它的地址。
* 返回函数内部变量的最佳实践
> 1. 应该按照正常写法（Best Practice）返回local变量
> 2. 编译器会决定要么使用NRVO，要么使用move语义来优化返回语句
> 3. 如果使用move语义，需要返回的类型有move constructor，否则只会进行复制

### 类型转换
不要使用 C 风格类型转换. 而应该使用 C++ 风格  

* 用`static_cast`替代 C 风格的值转换, 或某个类指针需要明确的向上转换为父类指针时.
* 用`const_cast`去掉`const`限定符.
* 用`reinterpret_cast`指针类型和整型或其它指针之间进行不安全的相互转换. 仅在你对所做一切了然于心时使用.

### 自增和自减
对简单数值 (非对象), 两种都无所谓. 对迭代器和模板类型, 使用前置自增 (自减).  

### 其他
* 尽量使用`const`和`constexpr`
* 整数用 `0`, 实数用 `0.0`, 指针用 `nullptr` 或 `NULL`, 字符 (串) 用 `'\0'`.
* `auto` 只能用在局部变量里用。别用在文件作用域变量，命名空间作用域变量和类数据成员里。永远别列表初始化 `auto` 变量