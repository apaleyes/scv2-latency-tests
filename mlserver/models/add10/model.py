from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.codecs import NumpyCodec


class Add10(MLModel):

    async def load(self) -> bool:
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        input = next(x for x in payload.inputs if x.name == "INPUT")
        input_data = NumpyCodec.decode_input(input)

        output_data = input_data + 10.0
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                NumpyCodec.encode_output("sum", output_data)
            ]
        )
