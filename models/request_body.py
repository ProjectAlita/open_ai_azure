from enum import Enum
from typing import Optional, List, Union, Any, Mapping

try:
    from pydantic.v1 import (
        BaseModel,
        StrictStr,
        ConstrainedFloat,
        ConstrainedInt,
        ConstrainedList,
        PositiveInt,
    )
except:  # pylint: disable=W0702
    from pydantic import (
        BaseModel,
        StrictStr,
        ConstrainedFloat,
        ConstrainedInt,
        ConstrainedList,
        PositiveInt,
    )

from pylon.core.tools import log


class ExtraForbidModel(BaseModel):
    class Config:
        extra = "forbid"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class FunctionCall(ExtraForbidModel):
    name: str
    arguments: str


class Message(ExtraForbidModel):
    role: Role
    content: Optional[StrictStr] = None
    name: Optional[StrictStr] = None
    function_call: Optional[FunctionCall] = None


class Function(ExtraForbidModel):
    name: StrictStr
    description: StrictStr
    parameters: StrictStr


class Temperature(ConstrainedFloat):
    ge = 0
    le = 2


class TopP(ConstrainedFloat):
    ge = 0
    le = 1


class N(ConstrainedInt):
    ge = 1
    le = 128


class Stop(ConstrainedList):
    max_items: int = 4
    __args__ = tuple([StrictStr])


class Penalty(ConstrainedFloat):
    ge = -2
    le = 2


class ChatCompletionRequestBody(BaseModel):
    deployment_id: StrictStr
    model: Optional[StrictStr] = None
    messages: List[Message]
    functions: Optional[List[Function]] = None
    function_call: Optional[Union[StrictStr, Mapping[StrictStr, StrictStr]]] = None
    stream: bool = False
    temperature: Optional[Temperature] = None
    top_p: Optional[TopP] = None
    n: Optional[N] = None
    stop: Optional[Union[StrictStr, Stop]] = None
    max_tokens: Optional[PositiveInt] = None
    presence_penalty: Optional[Penalty] = None
    frequency_penalty: Optional[Penalty] = None
    logit_bias: Optional[Mapping[int, float]] = None
    user: Optional[StrictStr] = None


class CompletionRequestBody(BaseModel):
    deployment_id: StrictStr
    prompt: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[PositiveInt] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[TopP] = None
    logit_bias: Optional[Mapping[int, float]] = None
    user: Optional[StrictStr] = None
    n: Optional[N] = None
    stream: bool = False
    logprobs: Optional[int] = None
    suffix: Optional[str] = None
    echo: Optional[bool] = None
    stop: Optional[Union[StrictStr, Stop]] = None
    presence_penalty: Optional[Penalty] = None
    frequency_penalty: Optional[Penalty] = None
    best_of: Optional[int] = None
