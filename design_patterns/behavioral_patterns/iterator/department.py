from dataclasses import dataclass
from datetime import datetime


@dataclass
class Department(object):
    number: int
    name: str
    date: datetime
