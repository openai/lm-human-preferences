import json
import sys
import typing
from dataclasses import fields, is_dataclass
from functools import lru_cache

from typeguard import check_type

from lm_human_preferences.utils import gcs


class HParams:
    """Used as a base class for hyperparameter structs. They also need to be annotated with @dataclass."""

    def override_from_json_file(self, filename):
        if filename.startswith('gs://'):
            hparams_str = gcs.download_contents(filename)
        else:
            hparams_str = open(filename).read()
        self.parse_json(hparams_str)

    def override_from_str(self, hparam_str):
        """Overrides values from a string like 'x.y=1,name=foobar'.

        Like tensorflow.contrib.training.HParams, this method does not allow specifying string values containing commas.
        """
        kvp_strs = hparam_str.split(',')
        flat_dict = {}
        for kvp_str in kvp_strs:
            k, sep, v = kvp_str.partition('=')
            if not sep:
                raise ValueError(f"Malformed hyperparameter value: '{kvp_str}'")
            flat_dict[k] = v

        self.override_from_str_dict(flat_dict)

    def override_from_str_dict(self, flat_dict, separator='.'):
        """Overrides values from a dict like {'x.y': "1", 'name': "foobar"}.

        Treats keys with dots as paths into nested HParams.
        Parses values according to the types in the HParams classes.
        """
        typemap = _type_map(type(self), separator=separator)

        parsed = {}
        for flat_k, s in flat_dict.items():
            if flat_k not in typemap:
                raise AttributeError(f"no field {flat_k} in {typemap}")
            parsed[flat_k] = _parse_typed_value(typemap[flat_k], s)

        self.override_from_dict(parsed, separator=separator)

    def parse_json(self, s: str):
        self.override_from_nested_dict(json.loads(s))

    def override_from_dict(self, flat_dict, separator='.'):
        """Overrides values from a dict like {'x.y': 1, 'name': "foobar"}.

        Treats keys with dots as paths into nested HParams.
        Values should be parsed already.
        """
        # Parse 'on' and 'off' values.
        typemap = _type_map(type(self), separator=separator)

        flat_dict_parsed = {}
        for flat_k, v in flat_dict.items():
            cls = _type_to_class(typemap[flat_k])
            if is_hparam_type(cls) and v == 'on':
                parsed_v = cls()
            elif is_hparam_type(cls) and v == 'off':
                parsed_v = None
            else:
                parsed_v = v
            flat_dict_parsed[flat_k] = parsed_v

        # Expand implicit nested 'on' values. For instance, {'x.y': 'on'} should mean {'x': 'on', 'x.y': 'on'}.
        flat_dict_expanded = {}
        for flat_k, v in flat_dict_parsed.items():
            flat_dict_expanded[flat_k] = v
            cls = _type_to_class(typemap[flat_k])
            if is_hparam_type(cls) and v is not None:
                parts = flat_k.split(separator)
                prefix = parts[0]
                for i in range(1, len(parts)):
                    if prefix not in flat_dict_expanded:
                        flat_dict_expanded[prefix] = _type_to_class(typemap[prefix])()
                    prefix += separator + parts[i]

        # Set all the values. The sort ensures that outer classes get initialized before their fields.
        for flat_k in sorted(flat_dict_expanded.keys()):
            v = flat_dict_expanded[flat_k]
            *ks, f = flat_k.split(separator)
            hp = self
            for i, k in enumerate(ks):
                try:
                    hp = getattr(hp, k)
                except AttributeError:
                    raise AttributeError(f"{hp} {'(' + separator.join(ks[:i]) + ') ' if i else ''}has no field '{k}'")
            try:
                setattr(hp, f, v)
            except AttributeError:
                raise AttributeError(f"{hp} ({separator.join(ks)}) has no field '{f}'")

    def override_from_nested_dict(self, nested_dict):
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                if getattr(self, k) is None:
                    cls = _type_to_class(_get_field(self, k).type)
                    setattr(self, k, cls())
                getattr(self, k).override_from_nested_dict(v)
            else:
                setattr(self, k, v)

    def to_nested_dict(self):
        d = {}
        for f in fields(self):
            fieldval = getattr(self, f.name)
            if isinstance(fieldval, HParams):
                fieldval = fieldval.to_nested_dict()
            d[f.name] = fieldval
        return d

    def validate(self, *, prefix=''):
        assert is_dataclass(self), f"You forgot to annotate {type(self)} with @dataclass"
        for f in fields(self):
            fieldval = getattr(self, f.name)
            check_type(prefix + f.name, fieldval, f.type)
            if isinstance(fieldval, HParams):
                fieldval.validate(prefix=prefix + f.name + '.')


def is_hparam_type(ty):
    if isinstance(ty, type) and issubclass(ty, HParams):
        assert is_dataclass(ty)
        return True
    else:
        return False


def _is_union_type(ty):
    return getattr(ty, '__origin__', None) is typing.Union


def dump(hparams, *, name='hparams', out=sys.stdout):
    out.write('%s:\n' % name)
    def dump_nested(hp, indent):
        for f in sorted(fields(hp), key=lambda f: f.name):
            v = getattr(hp, f.name)
            if isinstance(v, HParams):
                out.write('%s%s:\n' % (indent, f.name))
                dump_nested(v, indent=indent+'  ')
            else:
                out.write('%s%s: %s\n' % (indent, f.name, v))
    dump_nested(hparams, indent='  ')


def _can_distinguish_unambiguously(type_set):
    """Whether it's always possible to tell which type in type_set a certain value is supposed to be"""
    if len(type_set) == 1:
        return True
    if type(None) in type_set:
        return True
    if str in type_set:
        return False
    if int in type_set and float in type_set:
        return False
    if any(_is_union_type(ty) for ty in type_set):
        # Nested unions *might* be unambiguous, but don't support for now
        return False
    return True


def _parse_typed_value(ty, s):
    if ty is str:
        return s
    elif ty in (int, float):
        return ty(s)
    elif ty is bool:
        if s in ('t', 'true', 'True'):
            return True
        elif s in ('f', 'false', 'False'):
            return False
        else:
            raise ValueError(f"Invalid bool '{s}'")
    elif ty is type(None):
        if s in ('None', 'none', ''):
            return None
        else:
            raise ValueError(f"Invalid None value '{s}'")
    elif is_hparam_type(ty):
        if s in ('on', 'off'):
            # The class will be constructed later
            return s
        else:
            raise ValueError(f"Invalid hparam class value '{s}'")
    elif _is_union_type(ty):
        if not _can_distinguish_unambiguously(ty.__args__):
            raise TypeError(f"Can't always unambiguously parse a value of union '{ty}'")
        for ty_option in ty.__args__:
            try:
                return _parse_typed_value(ty_option, s)
            except ValueError:
                continue
        raise ValueError(f"Couldn't parse '{s}' as any of the types in '{ty}'")
    else:
        raise ValueError(f"Unsupported hparam type '{ty}'")


def _get_field(data, fieldname):
    matching_fields = [f for f in fields(data) if f.name == fieldname]
    if len(matching_fields) != 1:
        raise AttributeError(f"couldn't find field '{fieldname}' in {data}")
    return matching_fields[0]


def _update_disjoint(dst: dict, src: dict):
    for k, v in src.items():
        assert k not in dst
        dst[k] = v


@lru_cache()
def _type_map(ty, separator):
    typemap = {}
    for f in fields(ty):
        typemap[f.name] = f.type
        if is_hparam_type(f.type):
            nested = _type_map(f.type, separator=separator)
        elif _is_union_type(f.type):
            nested = {}
            for ty_option in f.type.__args__:
                if is_hparam_type(ty_option):
                    _update_disjoint(nested, _type_map(ty_option, separator=separator))
        else:
            nested = {}
        _update_disjoint(typemap, {f'{f.name}{separator}{k}': t for k, t in nested.items()})
    return typemap


def _type_to_class(ty):
    """Extract a constructible class from a type. For instance, `typing.Optional[int]` gives `int`"""
    if _is_union_type(ty):
        # Only typing.Optional supported: must be of form typing.Union[ty, None]
        assert len(ty.__args__) == 2
        assert ty.__args__[1] is type(None)
        return ty.__args__[0]
    else:
        return ty

