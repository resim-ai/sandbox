from typing import Any, Type, Dict, Optional, Callable

import mcap.decoder
import mcap.records as mcap_types
import mcap.well_known

import resim.actor.state.proto.observable_state_pb2 as ObservableState


class DecoderFactory(mcap.decoder.DecoderFactory):
    """An mcap DecoderFactory which can decode actor states."""

    def __init__(self):
        self._types: Dict[str, Type[Any]] = {
            "resim.actor.state.proto.ObservableStates": ObservableState.ObservableStates,
        }

    def decoder_for(
        self, message_encoding: str, schema: Optional[mcap_types.Schema]
    ) -> Optional[Callable[[bytes], Any]]:
        if (
            message_encoding != mcap.well_known.MessageEncoding.Protobuf
            or schema is None
            or schema.encoding != mcap.well_known.SchemaEncoding.Protobuf
        ):
            return None
        message_type = self._types.get(schema.name)
        if message_type is None:
            return None

        def decoder(data: bytes) -> Any:
            msg = message_type()
            msg.ParseFromString(data)
            return msg

        return decoder
