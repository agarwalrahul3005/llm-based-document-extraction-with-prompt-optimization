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

    # def to_prompt(self):

    #     sections = []

    #     for index, line in enumerate(self.lines):

    #         section = []
    #         section.append(f"========== Line {index+1} ==========")
    #         section.append(f"Line {index+1}")
    #         section.append(f"Text: {line.text}")
    #         section.append(f"Line BBox: {line.bbox}")
    #         section.append("Words:")

    #         for word in line.words:
    #             section.append(
    #                 f'  - "{word.text}" {word.bbox}'
    #             )

    #         sections.append("\n".join(section))

    #     return "\n\n".join(sections)

    def to_prompt(self, include_words=False):
        output = []
        for index, line in enumerate(self.lines):
            output.append(f"=== Line {index+1} ===")
            output.append(f"Text : {line.text}")
            output.append(f"Line BBox : {line.bbox}")

            if include_words:
                output.append("Words:")
                for word in line.words:
                    output.append(
                        f'   "{word.text}" -> {word.bbox}'
                    )

            output.append("")

        return "\n".join(output)