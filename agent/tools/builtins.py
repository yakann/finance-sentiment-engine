from datetime import datetime, timezone
from pydantic import BaseModel
from agent.tools.base import Tool


class _GetCurrentTimeInput(BaseModel):
    timezone_name: str = "UTC"


def _get_current_time(inputs: _GetCurrentTimeInput) -> str:
    now = datetime.now(timezone.utc)
    return now.strftime(f"%Y-%m-%d %H:%M:%S UTC  (requested tz: {inputs.timezone_name})")


get_current_time = Tool(
    name="get_current_time",
    description="Returns the current UTC date and time. Optionally accepts a timezone_name hint (display only).",
    input_schema=_GetCurrentTimeInput,
    handler=_get_current_time,
)
