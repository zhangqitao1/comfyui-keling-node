from typing import List, Dict, Optional, Union
from .prediction import Prediction, VideoPredictionResponse
from .client import Client

try:
    from pydantic import v1 as pydantic
except ImportError:
    import pydantic


class OmniVideo(Prediction):
    model_name: str = "kling-video-o1"
    prompt: str
    image_list: List[Dict] = []
    element_list: List[Dict] = []
    video_list: List[Dict] = []
    mode: str = "pro"
    aspect_ratio: Optional[str] = None
    duration: str = "5"
    callback_url: Optional[str] = None
    external_task_id: Optional[str] = None

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/videos/omni-video"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/videos/omni-video"
        self._response_cls = VideoPredictionResponse

