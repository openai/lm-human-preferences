from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import pytest

from lm_human_preferences.utils import hyperparams


@dataclass
class Simple(hyperparams.HParams):
    mandatory_nodefault: int = None
    mandatory_withdefault: str = "foo"
    optional_nodefault: Optional[int] = None
    fun: bool = True

def test_simple_works():
    hp = Simple()
    hp.override_from_str("mandatory_nodefault=3,optional_nodefault=None,fun=false")
    hp.validate()
    assert hp.mandatory_nodefault == 3
    assert hp.mandatory_withdefault == "foo"
    assert hp.optional_nodefault is None
    assert not hp.fun

def test_simple_failures():
    hp = Simple()
    with pytest.raises(TypeError):
        hp.validate()  # mandatory_nodefault unset
    with pytest.raises(ValueError):
        hp.override_from_str("mandatory_nodefault=abc")
    with pytest.raises(AttributeError):
        hp.override_from_str("nonexistent_field=7.0")
    with pytest.raises(ValueError):
        hp.override_from_str("fun=?")

@dataclass
class Nested(hyperparams.HParams):
    first: bool = False
    simple_1: Simple = field(default_factory=Simple)
    simple_2: Optional[Simple] = None

def test_nested():
    hp = Nested()
    hp.override_from_str("simple_1.mandatory_nodefault=8,simple_2=on,simple_2.mandatory_withdefault=HELLO")
    with pytest.raises(TypeError):
        hp.validate()  # simple_2.mandatory_nodefault unset
    hp.override_from_dict({'simple_2/mandatory_nodefault': 7, 'simple_1/optional_nodefault': 55}, separator='/')
    hp.validate()
    assert hp.simple_1.mandatory_nodefault == 8
    assert hp.simple_1.mandatory_withdefault == "foo"
    assert hp.simple_1.optional_nodefault == 55
    assert hp.simple_2.mandatory_nodefault == 7
    assert hp.simple_2.mandatory_withdefault == "HELLO"
    assert hp.simple_2.optional_nodefault is None

    hp.override_from_str("simple_2=off")
    hp.validate()
    assert hp.simple_2 is None

    with pytest.raises((TypeError, AttributeError)):
        hp.override_from_str("simple_2.fun=True")
    with pytest.raises(ValueError):
        hp.override_from_str("simple_2=BADVAL")

def test_nested_dict():
    hp = Nested()
    hp.override_from_nested_dict(
        {'simple_1': {'mandatory_nodefault': 8}, 'simple_2': {'mandatory_withdefault': "HELLO"}})
    with pytest.raises(TypeError):
        hp.validate()  # simple_2.mandatory_nodefault unset
    hp.override_from_nested_dict(
        {'simple_2': {'mandatory_nodefault': 7}, 'simple_1': {'optional_nodefault': 55}, 'first': True})
    hp.validate()
    assert hp.to_nested_dict() == {
        'first': True,
        'simple_1': {
            'mandatory_nodefault': 8,
            'mandatory_withdefault': "foo",
            'optional_nodefault': 55,
            'fun': True,
        },
        'simple_2': {
            'mandatory_nodefault': 7,
            'mandatory_withdefault': "HELLO",
            'optional_nodefault': None,
            'fun': True,
        },
    }

def test_nested_order():
    hp = Nested()
    # Either order should work
    hp.override_from_str_dict(OrderedDict([('simple_2.fun', 'True'), ('simple_2', 'on')]))
    hp.override_from_str_dict(OrderedDict([('simple_2', 'on'), ('simple_2.fun', 'True')]))

@dataclass
class Deeply(hyperparams.HParams):
    nested: Nested = None

def test_deeply_nested():
    hp = Deeply()
    hp.override_from_str("nested.simple_2=on")
    assert hp.nested is not None
    assert hp.nested.simple_2 is not None

    hp = Deeply()
    hp.override_from_dict({'nested.simple_2': 'on'})
    assert hp.nested is not None
    assert hp.nested.simple_2 is not None

def test_set_order():
    hp = Deeply()
    hp.override_from_dict(OrderedDict([('nested.first', True), ('nested.simple_1', 'on')]))
    assert hp.nested.first is True

    hp = Deeply()
    hp.override_from_dict(OrderedDict([('nested.simple_1', 'on'), ('nested.first', True)]))
    assert hp.nested.first is True
