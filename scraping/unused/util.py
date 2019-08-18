from json import JSONDecoder, JSONDecodeError
import re

# https://stackoverflow.com/a/50384432/2639190
NOT_WHITESPACE = re.compile(r"[^\s]")


def parse_jsonlines(document, pos=0, decoder=JSONDecoder()):
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj
