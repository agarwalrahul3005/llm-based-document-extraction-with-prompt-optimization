from layout.document import Word, Line, Document


class RawDocumentBuilder:

    def __init__(self, line_threshold=12):
        self.line_threshold = line_threshold


    def build(self, ocr_words):
        words = sorted(ocr_words,key=lambda w: (w["bbox"][1], w["bbox"][0]))
        lines = []
        current_words = []
        prev_y = None

        for item in words:
            word = Word(text=item["text"], bbox=item["bbox"])

            y = word.bbox[1]

            if prev_y is None:
                current_words.append(word)
            elif abs(y-prev_y) <= self.line_threshold:
                current_words.append(word)
            else:
                lines.append(self.merge(current_words))
                current_words=[word]

            prev_y=y

        if current_words:
            lines.append(self.merge(current_words))

        return Document(lines=lines)


    def merge(self, words):
        words=sorted(words,key=lambda w:w.bbox[0])
        x1=min(w.bbox[0] for w in words)
        y1=min(w.bbox[1] for w in words)
        x2=max(w.bbox[2] for w in words)
        y2=max(w.bbox[3] for w in words)
        return Line(words=words, bbox=[x1,y1,x2,y2])