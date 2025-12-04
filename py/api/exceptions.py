import httpx


class KLingAPIException(Exception):
    """A base class for all exceptions."""


class KLingAPIError(KLingAPIException):
    code: str

    message: str

    status_code: int

    request_id: str

    def __init__(
            self,
            code: str = None,
            message: str = None,
            request_id: str = None,
            status_code: int = None
    ) -> None:
        self.code = code
        self.message = message
        self.request_id = request_id
        self.status_code = status_code

    @classmethod
    def from_response(cls, response: httpx.Response) -> "KLingAPIError":
        try:
            data = response.json()
        except ValueError:
            data = {}

        return cls(
            code=data.get("code"),
            message=data.get("message"),
            request_id=data.get("request_id"),
            status_code=response.status_code,
        )

    def __str__(self) -> str:
        return "KLingAPIError Details:\n" + "\n".join(
            [f"{key}: {value}" for key, value in self.__dict__.items()]
        )
