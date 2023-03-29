import json
import triton_python_backend_utils as pb_utils

import logging

class TritonPythonModel:
    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        logging.debug(f"Incoming requests: {requests}")

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            params = inference_request.parameters()
            if "times" not in params:
                raise pb_utils.TritonModelException("Could not find parameter named 'times'")
            times = int(params["times"])

            # Get INPUT
            inp = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()

            for i in range(times):
                inference_request = pb_utils.InferenceRequest(
                        model_name='add10',
                        requested_output_names=['sum'],
                        inputs=[pb_utils.Tensor("INPUT", inp)]
                )

                inference_response = inference_request.exec()

                if inference_response.has_error():
                    raise pb_utils.TritonModelException(inference_response.error().message())
                else:
                    inp = pb_utils.get_output_tensor_by_name(inference_response, "sum").as_numpy()


            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor = pb_utils.Tensor("OUTPUT", inp)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

