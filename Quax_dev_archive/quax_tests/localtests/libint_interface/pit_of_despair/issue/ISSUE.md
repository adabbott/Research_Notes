I ran into this exact same issue after successfully installing libint2 on Ubuntu 18.04 with all the tests passing.
Considering this issue came up for me, @mdav2, and in issue #130, I will lay out what I found for the sake of posterity.  
Admittedly, I am not a C++ expert, so forgive me if any part of my explanation is off.

If one compiles libint from source without generating shared libraries e.g. 
`cmake . -DLIBINT2_BUILD_SHARED_AND_STATIC_LIBS=OFF`, which currently by default is OFF, we observe that for the following file, 
`test.cc`, which only imports libint2 and prints something, does not trivially compile.

```c++
#include <iostream>
#include <libint2.hpp>

int main() {
  std::cout << "Libint2 successfully imported\n";
  return 0;
}
```

The following compilation fails:
`g++ -I /path/to/eigen/ -I /path/to/libint2/include -I /path/to/libint2/include/libint2 -lint2 -L/path/to/where/libint2.a/is/located/ test.cc -o test`
and the error from above is reproduced:
```
/tmp/ccTbzoLO.o: In function `libint2::detail::__initializer::~__initializer()':
comon.cc:(.text._ZN7libint26detail13__initializerD2Ev[_ZN7libint26detail13__initializerD5Ev]+0xd): undefined reference to `libint2_static_cleanup'
collect2: error: ld returned 1 exit status
```

Note this compilation works fine if one omits `#include <libint2.hpp>` from `test.cc`, with or without the unnecessary include and library paths.

To fix, the compiling and linking needs to be done separately, this can be done by passing -c, and then linking with a second command:
`g++ -c test.cc -o test.o -I /path/to/eigen/ -I /path/to/libint2/include -I /path/to/libint2/include/libint2 -lint2 -L/path/to/where/libint2.a/is/located/`
`g++ test.o -o test -I /path/to/eigen/ -I /path/to/libint2/include -I /path/to/libint2/include/libint2 -lint2 -L/path/to/where/libint2.a/is/located/`

If one compiles libint with shared libraries,
`cmake . -DLIBINT2_BUILD_SHARED_AND_STATIC_LIBS=ON` 

things change a little, you can compile with a one-liner if you link with the flags `Wl,-rpath` to where the shared library libint2.so is. 
`g++ test.cc -o test -I /path/to/eigen/ -I /path/to/libint2/include -I /path/to/libint2/include/libint2 -lint2 -L/path/to/where/libint2.a/is/located/ -Wl,-rpath /path/to/where/libint2.so/is/located `


