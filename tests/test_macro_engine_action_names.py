"""The authorized PAUSE action spelling changes no CORE qualification."""
from copy import deepcopy

import pytest

import macro_engine.engine as engine
from test_macro_engine_contract import ASOF, POLICY, base, macro


@pytest.mark.parametrize('pressure', [False, True])
def test_pause_names_an_executable_core_add_without_margin(macro, pressure):
    book, facts = base()
    macro['pressure'] = pressure
    hiking = engine.decide(book, facts, POLICY, ASOF).to_dict()
    statements = deepcopy(POLICY) + [
        {**POLICY[0], 'date': '2026-09-30', 'action': 'HOLD'},
    ]
    pause = engine.decide(book, facts, statements, ASOF).to_dict()

    assert hiking['regime'] == 'HIKING' and pause['regime'] == 'PAUSE'
    assert hiking['action'] == ('CORE_ADD_NO_MARGIN' if pressure else 'CORE_ADD')
    assert pause['action'] == 'CORE_ADD_NO_MARGIN'
    assert pause['constraints']['allow_margin'] is False
    assert hiking['size_unit'] == pause['size_unit'] == .25
    assert hiking['cores'] == pause['cores']
    assert hiking['satellites'] == pause['satellites'] == []
    assert hiking['rejects'] == pause['rejects'] == []
    assert pause['cores'][0]['status'] == '可执行'
    assert hiking['trace']['closest'] == pause['trace']['closest']
    assert pause['trace']['closest']['passed'] is True
    for field in pause['constraints']:
        if field != 'allow_margin':
            assert hiking['constraints'][field] == pause['constraints'][field]
