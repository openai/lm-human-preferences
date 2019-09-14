def combos(*xs):
    if xs:
        return [x + combo for x in xs[0] for combo in combos(*xs[1:])]
    else:
        return [()]

def each(*xs):
    return [y for x in xs for y in x]

def bind(var, val, descriptor=''):
    extra = {}
    if descriptor:
        extra['descriptor'] = descriptor
    return [((var, val, extra),)]

def label(descriptor):
    return bind(None, None, descriptor)

def labels(*descriptors):
    return each(*[label(d) for d in descriptors])

def options(var, opts_with_descs):
    return each(*[bind(var, val, descriptor) for val, descriptor in opts_with_descs])

def _shortstr(v):
    if isinstance(v, float):
        s = f"{v:.03}"
        if '.' in s:
            s = s.lstrip('0').replace('.','x')
    else:
        s = str(v)
    return s

def options_shortdesc(var, desc, opts):
    return each(*[bind(var, val, desc + _shortstr(val)) for val in opts])

def options_vardesc(var, opts):
    return options_shortdesc(var, var, opts)

def repeat(n):
    return each(*[label(i) for i in range(n)])

# list monad bind; passes descriptors to body
def foreach(inputs, body):
    return [inp + y for inp in inputs for y in body(*[extra['descriptor'] for var, val, extra in inp])]

def bind_nested(prefix, binds):
    return [
        tuple([ (var if var is None else prefix + '.' + var, val, extra) for (var, val, extra) in x ])
        for x in binds
    ]
