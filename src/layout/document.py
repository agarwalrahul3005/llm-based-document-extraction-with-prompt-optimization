from dataclasses import dataclass


@dataclass
class Word:
    text: str
    bbox: list

@dataclass
class Line:
    words: list
    bbox: list

    @property
    def text(self):
        return " ".join(word.text for word in self.words)

    @property
    def left(self):
        return self.bbox[0]

    @property
    def top(self):
        return self.bbox[1]

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

@dataclass
class Document:
    lines: list

    def to_prompt(self, include_words=False):
        output = []
        output.append( "The following OCR was extracted from a scanned form.")
        # output.append("Each line contains its text followed by its bounding box.")
        output.append("")
        for index, line in enumerate(self.lines):
            output.append(line.text)
            # output.append(f"Location: {line.bbox}")
            if include_words:
                output.append("Words:")
                for word in line.words:
                    output.append(
                        f'   "{word.text}" -> {word.bbox}'
                    )

        output.append("") 

        return "\n".join(output)