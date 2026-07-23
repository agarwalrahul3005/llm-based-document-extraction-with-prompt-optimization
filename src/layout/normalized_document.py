from dataclasses import dataclass


@dataclass
class NormalizedWord:
    text: str
    bbox: list

@dataclass
class NormalizedLine:

    id: int
    words: list
    bbox: list

    @property
    def text(self):
        return " ".join(
            w.text for w in self.words
        )

    @property
    def left(self):
        return self.bbox[0]

    @property
    def top(self):
        return self.bbox[1]

    @property
    def right(self):
        return self.bbox[2]

    @property
    def bottom(self):
        return self.bbox[3]


@dataclass
class NormalizedDocument:

    lines: list
    width: int
    height: int


    def to_prompt(self):

        output = []

        output.append(
            f"""
            DOCUMENT SIZE
            width: {self.width}
            height: {self.height}

            Coordinates are normalized between 0 and 1.

            """
        )


        for line in self.lines:

            output.append(
                f"""
                ==============================
                Line ID: {line.id}

                Text:
                {line.text}

                Bounding Box:
                left={line.left}
                top={line.top}
                right={line.right}
                bottom={line.bottom}

                Words:
                """
            )


            for word in line.words:

                output.append(
                    f"""
                    - text: {word.text}
                    bbox: {word.bbox}
                    """
                )
                
        return "\n".join(output)
