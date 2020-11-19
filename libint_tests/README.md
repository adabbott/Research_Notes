### Libint tests
This repository is for testing and developing Libint code which can be called by Python, load arrays into HDF5 files, etc.

Of key importance is that Libint must be compiled with position independent code to play nice with Pybind11, since Pybind11 can only create a dynamic (shared) library
which represents the importable python module.

Here is the procedure for building libint (assuming you are familiar to installing libint from source and the various dependencies that are required):


Clone the libint repository and cd into it.

Run autogen:  
```
./autogen.sh 
```

Now make a build directory, cd into it, make a directory called PREFIX

```
mkdir BUILD
cd BUILD
mkdir PREFIX
```

Here we decide what maximum angular momentum (s,p,d,f,g..) in our basis to support in libint library.
Setting `--with-opt-am=0` makes the configuring very fast (less than an hour in most cases) whereas the default setting, 3, which
optimizes all integral classes with up to f functions, makes this particular configuration take a long time (about 6 days).
The `--enable` commands control the maximum derivative order to compile for.
So this configuration creates Libint code for up to g functions, with fourth order derivatives of couloumb, overlap, kinetic, and potential integrals (and a few others).
We are turning 3 and 2 center ERI's off, as well as explicitly correlated and relativistic intetgrals.

```
../configure --prefix=/path/to/libint/BUILD/PREFIX/  --with-max-am=4 --with-opt-am=4 --enable-1body=4 --enable-eri=4 --enable-eri3=no --enable-eri2=no --enable-g12=no --enable-g12dkh=no
```

This command makes the tarball (takes a long time depending on above options)
```
make export
```

Unpack the library tarball `tar -xvf libint....tgz`, cd into it.
Next we make a new PREFIX directory which is where the headers, library, and shared files will go. 
Then I use Ninja to build, with specific PREFIX path and  and prepare Ninja build with Position independent code ON.
Then the build with threading `j4` is very fast, you can run the tests, and then install which dumps the headers and libraries into PREFIX.

```
mkdir PREFIX

cmake . -G Ninja -DCMAKE_INSTALL_PREFIX=/custom/path/to/PREFIX/where/headers/and/library/will/go/ -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build . -- -j4

cmake --build . --target check

cmake --build . --target install
```



