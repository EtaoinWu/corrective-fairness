class Curried:
    def __init__(self, fn):
        self.fn = fn
        self.args = ()
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        args = self.args + args
        kwargs = {**self.kwargs, **kwargs}
        if len(args) + len(kwargs) >= self.fn.__code__.co_argcount:
            return self.fn(*args, **kwargs)
        new_curried = Curried(self.fn)
        new_curried.args = args
        new_curried.kwargs = kwargs
        return new_curried

def curry(fn):
    return Curried(fn)

def c2i(f):
    return f(lambda x: x + 1)(0)
def i2c(n):
    @curry
    def inner(f, x):
        for _ in range(n):
            x = f(x)
        return x
    return inner

@curry
def cons(x, y, f):
    return f(x)(y)

@curry
def true(x, y):
    return x

@curry
def false(x, y):
    return y

def car(p):
    return p(true)

def cdr(p):
    return p(false)

@curry
def and_op(x, y):
    return x(y, false)

@curry
def or_op(x, y):
    return x(true, y)

@curry
def not_op(f, x, y):
    return f(y, x)

@curry
def zero(f, x):
    return x

@curry
def succ(n, f, x):
    return f(n(f)(x))

@curry
def add(m, n, f, x):
    return m(f)(n(f)(x))

@curry
def mult(m, n, f, x):
    return m(n(f))(x)

def prec(n):
    init = cons(false, zero)
    def iter(pair):
        l, r = car(pair), cdr(pair)
        return cons(true, l(succ(r))(r))
    end = n(iter)(init)
    return cdr(end)

def Z(f):
    def z1(x):
        def z2(y):
            return x(x)(y)
        return f(z2)
    return z1(z1)

@curry
def frac(frac_, n):
    if n > 0:
        return n * frac_(n - 1)
    else:
        return 1
    
@curry
def cfrac(cfrac_, n):
    return n(true(true))(false)(                # if n > 0:
        lambda z: mult(cfrac_(prec(n)))(n)(z)   #   return (n-1)! * n
    )(                                          # else:
        succ(zero)                              #   return 1
    )

def main():
    res = c2i(Z(cfrac)(i2c(4)))
    print(res)

if __name__ == "__main__":
    main()
