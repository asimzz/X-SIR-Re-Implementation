import os
import re
import sys
import json
import time
import aiohttp
import asyncio
import logging
import tiktoken
from tqdm import tqdm
from typing import Callable
from dataclasses import dataclass, field

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def api_endpoint_from_url(request_url):
    match = re.search(r"^https://[^/]+/v\d+/(.+)$", request_url)
    if match is None:
        match = re.search(r"^https://[^/]+/openrouter/[^/]+/(.+?)(\?|$)", request_url)
    if match:
        return match[1]
    return "unknown"

def task_id_generator_function():
    task_id = 0
    while True:
        yield task_id
        task_id += 1

@dataclass
class StatusTracker:
    num_tasks_total: int = 0
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0

@dataclass
class APIRequest:
    task_id: int
    request_json: dict
    attempts_left: int
    metadata: dict
    response_to_output_func: Callable[[dict, str], None]
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
        progress_bar=tqdm
    ):
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1

        except Exception as e:
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left > 0:
                logging.warning(
                    f"Request {self.request_json} failed with errors: {self.result}. Retry attempt {self.attempts_left} left."
                )
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts.")
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = {
                "response": response,
                "metadata": self.metadata if self.metadata else None
            }
            self.response_to_output_func(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            progress_bar.update(n=1)
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")

class CallOpenRouter:
    def __init__(
        self,
        request_url,
        api_key,
        input_file_path,
        output_file_path,
        input_to_requests_func,
        response_to_output_func,
        is_all_done_func=None,
        post_run_func=None,
        max_attempts=5,
        logging_level=logging.INFO
    ):
        self.request_url = request_url
        self.api_key = api_key
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.input_to_requests_func = input_to_requests_func
        self.response_to_output_func = response_to_output_func
        self.is_all_done_func = is_all_done_func
        self.post_run_func = post_run_func
        self.api_endpoint = api_endpoint_from_url(request_url)
        self.max_attempts = max_attempts
        self.logging_level = logging_level
        self.progress_bar = tqdm()
        self.request_header = {"Authorization": f"Bearer {self.api_key}"}
        logging.basicConfig(level=logging_level)
        logging.debug(f"Logging initialized at level {logging_level}")
        self.queue_of_requests_to_retry = asyncio.Queue()
        self.task_id_generator = task_id_generator_function()
        self.status_tracker = StatusTracker()
        self.next_request = None
        assert os.path.isfile(self.input_file_path), f"Input file {self.input_file_path} does NOT exist."
        output_directory = os.path.dirname(self.output_file_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        logging.debug(f"Initialization complete.")

    async def run(self):
        if self.is_all_done_func and self.is_all_done_func(self.input_file_path, self.output_file_path):
            logging.info("All done!")
            return

        requests = self.input_to_requests_func(self.input_file_path, self.output_file_path)
        total = len(requests)
        self.progress_bar.reset(total=total)
        self.status_tracker.num_tasks_total = total

        if total == 0:
            logging.info("No requests to run.")
            return

        async with aiohttp.ClientSession() as session:
            while True:
                if self.next_request is None:
                    if not self.queue_of_requests_to_retry.empty():
                        self.next_request = self.queue_of_requests_to_retry.get_nowait()
                    elif requests:
                        request_json = requests.pop(0)
                        self.next_request = APIRequest(
                            task_id=next(self.task_id_generator),
                            request_json=request_json,
                            attempts_left=self.max_attempts,
                            metadata=request_json.pop("metadata", None),
                            response_to_output_func=self.response_to_output_func
                        )
                        self.status_tracker.num_tasks_started += 1
                        self.status_tracker.num_tasks_in_progress += 1

                if self.next_request:
                    asyncio.create_task(
                        self.next_request.call_api(
                            session=session,
                            request_url=self.request_url,
                            request_header=self.request_header,
                            retry_queue=self.queue_of_requests_to_retry,
                            save_filepath=self.output_file_path,
                            status_tracker=self.status_tracker,
                            progress_bar=self.progress_bar
                        )
                    )
                    self.next_request = None

                if self.status_tracker.num_tasks_in_progress == 0:
                    break

                await asyncio.sleep(0.01)

        if self.post_run_func:
            self.post_run_func(self.output_file_path)
        logging.info("All done!")