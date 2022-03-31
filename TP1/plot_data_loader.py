

from dataclasses import dataclass
import json
from typing import Generic, TypeVar

from dataclasses_serialization.serializer_base.dictionary import dict_serialization
from dataclasses_serialization.serializer_base.noop import (
    noop_deserialization,
    noop_serialization,
)
from dataclasses_serialization.serializer_base.serializer import Serializer

from cube import Cube
from methods import ExecutionData, FullExecutionData, Output
from tree import Node


JSONSerializer = Serializer(
    serialization_functions={
        dict: lambda dct: dict_serialization(dct, key_serialization_func=JSONSerializer.serialize, value_serialization_func=JSONSerializer.serialize),
        list: lambda lst: list(map(JSONSerializer.serialize, lst)),
        Node: lambda node: {"depth": node.get_depth()},
        type(TimeoutError): lambda _: "timeout",
        Cube: lambda cube: repr(cube),
        (str, int, float, bool, type(None)): noop_serialization
    },
    deserialization_functions={
        # # dict: lambda cls, dct: None,
        # dict: lambda cls, dct: dict_deserialization(cls, dct, key_deserialization_func=JSONSerializer.deserialize, value_deserialization_func=JSONSerializer.deserialize),
        # # list: lambda cls, lst: list_deserialization(cls, lst, deserialization_func=JSONSerializer.deserialize(list if type(lst[0]) is list else ExecutionData)),
        # list: lambda cls, lst: deserialize_list(lst),
        # Cube: lambda cls, cube: Cube.parse(cube),
        # # Node: lambda cls, node: Node(JSONSerializer.deserialize(Cube, node['state']), node['action'], JSONSerializer.deserialize(Node.__init__, node['parent'])),
        Output: lambda cls, output: Output(output['solution'], output['expanded_count'], output['border_count']) if output != "timeout" else TimeoutError,
        ExecutionData: lambda cls, execution_data: ExecutionData(execution_data['time'], JSONSerializer.deserialize(Output, execution_data['output'])),
        (str, int, float, bool, type(None)): noop_deserialization
    }
)


def save(executions_by_method: FullExecutionData, filename: str = 'plot_data.json'):
    data = JSONSerializer.serialize(executions_by_method)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)


def load(filename: str = 'plot_data.json') -> FullExecutionData:
    with open(filename) as file:
        data = json.load(file)
    return {
        str(method_name): {
            int(depth): 
            [
                JSONSerializer.deserialize(ExecutionData, execution_data) for execution_data in execution_data_group #type: ignore
            ]
            for depth, execution_data_group in execution_data_groups.items()
        } for method_name, execution_data_groups in data.items()
    }
