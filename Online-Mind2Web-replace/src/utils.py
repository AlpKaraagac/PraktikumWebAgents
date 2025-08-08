import base64
import io
import os
import backoff
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    OpenAI
)
from PIL import Image

MAX_SIDE   = 512        # <-- was 16384
MAX_PIXELS = 350_000    # 512×682 ~ 0.35 MP

def _resize_if_needed(img: Image.Image,
                      max_side: int = MAX_SIDE,
                      max_pixels: int = MAX_PIXELS) -> Image.Image:
    w, h = img.size
    if w > max_side or h > max_side or w * h > max_pixels:
        scale = min(max_side / max(w, h), (max_pixels / (w * h)) ** 0.5)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def encode_image(img: Image.Image) -> str:
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70, optimize=True)   # <- quality 70
    return base64.b64encode(buf.getvalue()).decode()

def is_blank(img: Image.Image, var_thresh: int = 3) -> bool:
    from PIL import ImageStat
    return ImageStat.Stat(img.convert("L")).var[0] < var_thresh

def safe_open(path: str) -> Image.Image:
    """Open an image and down-scale it so Pillow never explodes."""
    img = Image.open(path)
    img.load()                          # force the file to read immediately
    return _resize_if_needed(img)


def extract_predication(response, mode):
    """Extract the prediction from the response."""
    if mode == "Autonomous_eval":
        try:
            return 1 if "success" in response.lower().split('status:')[1] else 0
        except:
            return 0
    elif mode == "AgentTrek_eval":
        try:
            return 1 if "success" in response.lower().split('status:')[1] else 0
        except:
            return 0
    elif mode == "WebVoyager_eval":
        return 0 if "FAILURE" in response else 1
    elif mode in ("WebJudge_Online_Mind2Web_eval", "WebJudge_general_eval"):
        try:
            return 1 if "success" in response.lower().split('status:')[1] else 0
        except:
            return 0
    else:
        raise ValueError(f"Unknown mode: {mode}")


class OpenaiEngine:
    def __init__(
        self,
        api_key=None,
        stop=None,
        rate_limit=-1,
        model=None,
        tokenizer=None,
        temperature=0,
        port=-1,
        endpoint_target_uri="",
        base_url=None,
        **kwargs,
    ) -> None:
        """
        A thin wrapper around the OpenAI Python client.

        If `base_url` is provided, directs requests to that endpoint
        (e.g. for Gemini compatibility).
        """
        assert os.getenv("OPENAI_API_KEY", api_key), "must set OPENAI_API_KEY or pass api_key"
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.stop = stop or []
        self.temperature = temperature
        self.model = model
        # rate_limit -> minimum interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_available = [0]
        # initialize client, optionally with custom base_url
        client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip('/')
        self.client = OpenAI(**client_kwargs)

    def log_error(details):
        print(f"Retrying in {details['wait']:0.1f}s due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=log_error
    )
    def generate(self, messages, max_new_tokens=512, temperature=None, model=None, **kwargs):
        import time
        m    = model or self.model
        temp = temperature if temperature is not None else self.temperature

        # Always respect our per‐engine request_interval
        now = time.time()
        wait = self.next_available[0] - now
        if wait > 0:
            time.sleep(wait)

        # Loop until we get a successful response
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=m,
                    messages=messages,
                    max_tokens=4096,
                    temperature=temp,
                    tool_choice="none",
                    **kwargs,
                )
                # schedule next slot
                self.next_available[0] = time.time() + self.request_interval
                return [c.message.content for c in resp.choices]
            except RateLimitError as e:
                # Try to read a Retry-After header
                delay = None
                if hasattr(e, 'response') and e.response:
                    hdr = e.response.headers.get("Retry-After")
                    if hdr:
                        try:
                            delay = float(hdr)
                        except ValueError:
                            pass
                # Fallback to 60s if we couldn’t parse anything
                if delay is None:
                    delay = max(self.request_interval, 60.0)
                print(f"[RATE LIMIT] hit, sleeping for {delay:.1f}s before retry")
                time.sleep(delay)


class GeminiEngine(OpenaiEngine):
    def __init__(self, api_key: str, model: str, **kwargs):
        """
        A Gemini-compatible engine via the OpenAI client’s OpenAI class,
        pointing at Google’s Generative Language endpoints.
        """
        # Vertex AI's OpenAI-compatible endpoint
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs
        )


def get_engine(provider: str, api_key: str, model: str, **kwargs):
    """
    Return an engine for 'openai' or 'gemini'.

    provider: "openai" or "gemini"
    api_key: the API key
    model: e.g. "gpt-4o" or "gemini-2.0-flash"
    Additional kwargs are forwarded to the engine constructor.
    """
    if provider.lower() == "gemini":
        return GeminiEngine(api_key=api_key, model=model, **kwargs)
    return OpenaiEngine(api_key=api_key, model=model, **kwargs)