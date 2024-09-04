import requests
from enum import Enum
from typing import Any
from multiaddr import Multiaddr
from hivemind.optim.optimizer import logger
import json
import base64

class PrimeIntellectProgressLogger:
    """
    Logs the status of nodes, and training progress to Prime Intellect's API.
    """

    def __init__(self, peer_id, project, config, maddrs, *args, **kwargs):
        self.peer_id = str(peer_id)
        self.project = project
        self.config = self._serialize_payload(config)
        self.data = []
        self.batch_size = 10
        self.base_url = "https://protocol-api.primeintellect.ai/training_runs"

        self.maddrs = [str(maddr) for maddr in maddrs]
        self.run_id = self._initialize_run()

    def _serialize_payload(self, data):
        def serialize_custom(obj):
            if isinstance(obj, Enum):
                return obj.name
            elif isinstance(obj, Multiaddr):
                return str(obj)
            elif isinstance(obj, bytes):
                return base64.b64encode(obj).decode('utf-8')
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        return json.loads(json.dumps(data, default=serialize_custom))

    def _initialize_run(self):
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "project": self.project,
            "config": self.config,
            "peer_maddrs": self.maddrs,
            "peer_id": self.peer_id
        }
        api = f"{self.base_url}/init"
        try:
            response = requests.post(api, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            run_id = response_data.get('run_id')
            if run_id:
                logger.info(f"Successfully initialized run on Prime Intellect API. Run ID: {run_id}")
                return run_id
            else:
                raise ValueError("No run ID returned from Prime Intellect API")
        except requests.RequestException as e:
            logger.error(f"Failed to initialize run on Prime Intellect API: {e}")
            return None

    def _remove_duplicates(self):
        seen = set()
        unique_logs = []
        for log in self.data:
            log_tuple = tuple(sorted(log.items()))
            if log_tuple not in seen:
                unique_logs.append(log)
                seen.add(log_tuple)
        self.data = unique_logs

    def log(self, data: dict[str, Any]):
        serialized_data = self._serialize_payload(data)
        # Add peer_id to log data, so that logs can be associated with the correct node
        serialized_data['peer_id'] = self.peer_id
        self.data.append(serialized_data)
        if len(self.data) >= self.batch_size:
            self._remove_duplicates()  # Remove duplicates before sending
            self._send_batch()

    def _send_batch(self):
        # Remove duplicates before sending
        self._remove_duplicates()
        
        # Send batch of logs to Prime Intellect's API endpoint
        batch = self.data[:self.batch_size]
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "run_id": self.run_id,
            "logs": batch
        }
        api = f"{self.base_url}/logs"
        try:
            response = requests.post(api, json=payload, headers=headers)
            response.raise_for_status()
            logger.debug(f"Successfully sent batch of {len(batch)} logs to Prime Intellect API")
        except requests.RequestException as e:
            logger.warning(f"Failed to send logs to Prime Intellect API: {e}")
        
        self.data = self.data[self.batch_size:]

    def _finish(self):
        headers = {
            "Content-Type": "application/json"
        }
        api = f"{self.base_url}/{self.run_id}/finish"
        try:
            response = requests.post(api, headers=headers)
            response.raise_for_status()
            logger.debug(f"Successfully called finish endpoint for run ID: {self.run_id}")
        except requests.RequestException as e:
            logger.warning(f"Failed to call finish endpoint: {e}")

    def finish(self):
        # Remove duplicates before sending any remaining logs
        self._remove_duplicates()
        
        # Send any remaining logs
        while self.data:
            self._send_batch()

        self._finish()

_progress_logger = None

def init_pi_progress_logger(peer_id, project, config, *args, **kwargs):
    global _progress_logger
    _progress_logger = PrimeIntellectProgressLogger(peer_id, project, config, *args, **kwargs)

def get_pi_progress_logger():
    global _progress_logger
    if _progress_logger is None:
        raise ValueError("Status logger has not been initialized. Please call init_status_logger first.")
    return _progress_logger

def log_progress_to_pi(data: dict[str, Any]):
    logger = get_pi_progress_logger()
    logger.log(data)

def finish_pi_progress_logger():
    logger = get_pi_progress_logger()
    logger.finish()
