from dataclasses import dataclass
from handlers import handler_chain


@dataclass
class Request:
    request_type: str
    request_details: str


REQUESTS = [
    Request('dogs', 'walk twice a day'),
    Request('cats', 'see how pretty they are'),
    Request('fish', 'keep the tank clean'),
    Request('parrot', 'check for signs of life'),
    Request('tarantula', 'who keeps a tarantula as a pet?')
]


if __name__ == '__main__':
    for r in REQUESTS:
        handler_chain.handle(r)
