import logging

import dotenv
import ray
from ray import serve
import time
import requests
from conf import setup_logging
#from is3_rl_wholesale.api.observation_controller import ObservationController
from is3_rl_wholesale.api.wholesale_controller import WholesaleController
from is3_rl_wholesale.api.policy_server import start_server
import json

def main():
    setup_logging()
    dotenv.load_dotenv()
    log = logging.getLogger(__name__)
    log.info("Starting Ray server ...")
    #time.sleep(5)
    
    ray.init(address="auto", namespace="serve", ignore_reinit_error=True)
    serve.start()
    
    response = requests.get("http://127.0.0.1:8000/wholesale")
    print(response)
    start_server()
    #while True:
    #   time.sleep(1)



if __name__ == "__main__":
    
    
    main()
