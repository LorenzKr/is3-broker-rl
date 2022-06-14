import logging
import ray
from ray import serve
from starlette.requests import Request
import time

ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
@serve.deployment(route_prefix="/observe")

class ObservationController:
    def __init__(self):
        self._log = logging.getLogger(__name__)

    def __call__(self, request: Request):
        self._log.debug(request)
        
        return {"test": "test2"}


# Uncomment to disable API endpoint
#serve.start(detached=True)
ObservationController.deploy()

#while True:
#    time.sleep(5)
    
