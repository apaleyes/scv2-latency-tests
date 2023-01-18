from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.errors import MLServerError
from mlserver.codecs import NumpyCodec
import numpy as np
from fastapi import status

import requests
import logging


TIMES_KEY = "times"
URL_KEY = "url"


class ModelParametersMissing(MLServerError):
  def __init__(self, model_name: str, reason: str):
    super().__init__(
      f"Parameters missing for model {model_name} {reason}", status.HTTP_400_BAD_REQUEST
    )

class Caller(MLModel):

    async def load(self) -> bool:
        if self.settings.parameters is None or \
            self.settings.parameters.extra is None:
            raise ModelParametersMissing(self.name, "no settings.parameters.extra found")

        self.times = self.settings.parameters.extra[TIMES_KEY]
        if self.times is None:
            raise ModelParametersMissing(self.name, "no settings.parameters.extra.query found")
        else:
            self.times = int(self.times)
        self.url = self.settings.parameters.extra[URL_KEY]
        if self.url is None:
            raise ModelParametersMissing(self.name, "no settings.parameters.extra.url found")

        self.ready = True
        return self.ready


    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        input = next(x for x in payload.inputs if x.name == "INPUT0")
        data = NumpyCodec.decode_input(input)
        i = 0
        while i < self.times:
            inference_request = {
                "inputs": [
                    {
                        "name": "INPUT",
                        "shape": list(data.shape),
                        "datatype": "FP32",
                        "data": data.tolist()
                    }
                ]
            }

            import asyncio
            from functools import partial

            loop = asyncio.get_running_loop()
            send_request = partial(requests.post, url=self.url, json=inference_request, timeout=5)
            response = await loop.run_in_executor(None, send_request)

            # response = requests.post(self.url, json=inference_request, timeout=5).json()

            response_json = response.json()
            if "outputs" in response_json:
                outputs = response_json["outputs"]
                output = next(x for x in outputs if x["name"] == "sum")
                data = np.array(output["data"])
            else:
                data = np.array([response_json["error"]]) #np.array([response_json["text"], response_json["status_code"]])
                break

            i += 1

        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                NumpyCodec.encode_output("OUTPUT0", data)
            ])
