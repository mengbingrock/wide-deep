# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: allreduce.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='allreduce.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0f\x61llreduce.proto\x1a\x0c\x63ommon.proto\"\x8e\x01\n\x10RingAllReduceReq\x12&\n\x05stage\x18\x01 \x01(\x0e\x32\x17.RingAllReduceReq.Stage\x12&\n\x0enode_gradients\x18\x02 \x01(\x0b\x32\x0e.NodeGradients\"*\n\x05Stage\x12\x08\n\x04INIT\x10\x00\x12\x0b\n\x07SCATTER\x10\x01\x12\n\n\x06GATHER\x10\x02\"\x13\n\x11RingAllReduceResp2\x95\x01\n\x14RingAllReduceService\x12I\n\x13VariableWeightsInit\x12\x17.VariableWeightsReqResp\x1a\x17.VariableWeightsReqResp\"\x00\x12\x32\n\x07Recieve\x12\x11.RingAllReduceReq\x1a\x12.RingAllReduceResp\"\x00\x62\x06proto3')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])



_RINGALLREDUCEREQ_STAGE = _descriptor.EnumDescriptor(
  name='Stage',
  full_name='RingAllReduceReq.Stage',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='INIT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SCATTER', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GATHER', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=134,
  serialized_end=176,
)
_sym_db.RegisterEnumDescriptor(_RINGALLREDUCEREQ_STAGE)


_RINGALLREDUCEREQ = _descriptor.Descriptor(
  name='RingAllReduceReq',
  full_name='RingAllReduceReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='stage', full_name='RingAllReduceReq.stage', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='node_gradients', full_name='RingAllReduceReq.node_gradients', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RINGALLREDUCEREQ_STAGE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=176,
)


_RINGALLREDUCERESP = _descriptor.Descriptor(
  name='RingAllReduceResp',
  full_name='RingAllReduceResp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=178,
  serialized_end=197,
)

_RINGALLREDUCEREQ.fields_by_name['stage'].enum_type = _RINGALLREDUCEREQ_STAGE
_RINGALLREDUCEREQ.fields_by_name['node_gradients'].message_type = common__pb2._NODEGRADIENTS
_RINGALLREDUCEREQ_STAGE.containing_type = _RINGALLREDUCEREQ
DESCRIPTOR.message_types_by_name['RingAllReduceReq'] = _RINGALLREDUCEREQ
DESCRIPTOR.message_types_by_name['RingAllReduceResp'] = _RINGALLREDUCERESP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RingAllReduceReq = _reflection.GeneratedProtocolMessageType('RingAllReduceReq', (_message.Message,), dict(
  DESCRIPTOR = _RINGALLREDUCEREQ,
  __module__ = 'allreduce_pb2'
  # @@protoc_insertion_point(class_scope:RingAllReduceReq)
  ))
_sym_db.RegisterMessage(RingAllReduceReq)

RingAllReduceResp = _reflection.GeneratedProtocolMessageType('RingAllReduceResp', (_message.Message,), dict(
  DESCRIPTOR = _RINGALLREDUCERESP,
  __module__ = 'allreduce_pb2'
  # @@protoc_insertion_point(class_scope:RingAllReduceResp)
  ))
_sym_db.RegisterMessage(RingAllReduceResp)



_RINGALLREDUCESERVICE = _descriptor.ServiceDescriptor(
  name='RingAllReduceService',
  full_name='RingAllReduceService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=200,
  serialized_end=349,
  methods=[
  _descriptor.MethodDescriptor(
    name='VariableWeightsInit',
    full_name='RingAllReduceService.VariableWeightsInit',
    index=0,
    containing_service=None,
    input_type=common__pb2._VARIABLEWEIGHTSREQRESP,
    output_type=common__pb2._VARIABLEWEIGHTSREQRESP,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Recieve',
    full_name='RingAllReduceService.Recieve',
    index=1,
    containing_service=None,
    input_type=_RINGALLREDUCEREQ,
    output_type=_RINGALLREDUCERESP,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_RINGALLREDUCESERVICE)

DESCRIPTOR.services_by_name['RingAllReduceService'] = _RINGALLREDUCESERVICE

# @@protoc_insertion_point(module_scope)
