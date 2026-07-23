from layout.normalized_document import (
    NormalizedDocument,
    NormalizedLine,
    NormalizedWord
)



class NormalizedDocumentBuilder:

    def __init__(self, line_threshold=12):
        self.line_threshold = line_threshold

    def build(self, ocr_words):
        max_x = 0
        max_y = 0

        for item in ocr_words:
            bbox = item["bbox"]
            max_x = max(max_x, bbox[2])
            max_y = max(max_y,bbox[3])


        words = []

        for item in ocr_words:
            bbox=item["bbox"]
            normalized=[
                round(bbox[0]/max_x, 4),
                round(bbox[1]/max_y, 4),
                round(bbox[2]/max_x,4),
                round(bbox[3]/max_y, 4)
            ]

            words.append(
                {
                    "text":item["text"],
                    "bbox":normalized,
                    "raw":bbox
                }
            )

        # sort by y then x
        words.sort( key=lambda x: (x["raw"][1], x["raw"][0]))

        lines=[]
        current=[]
        prev_y=None

        for item in words:
            y=item["raw"][1]

            word=NormalizedWord(text=item["text"],bbox=item["bbox"])

            if prev_y is None:
                current.append(word)
            elif abs(y-prev_y) <= self.line_threshold:
                current.append(word)
            else:
                lines.append(self.make_line(current, len(lines)+1))
                current=[word]

            prev_y=y

        if current:
            lines.append(self.make_line(current,len(lines)+1))

        return NormalizedDocument(lines=lines, width=max_x, height=max_y)

    def make_line(self, words, index):

        words.sort(key=lambda w:w.bbox[0])

        x1 = min(w.bbox[0] for w in words )
        y1 = min(w.bbox[1] for w in words)
        x2 = max(w.bbox[2] for w in words)
        y2 = max(w.bbox[3] for w in words)

        return NormalizedLine(id=index, words=words, bbox=[round(x1,4), round(y1,4), round(x2,4), round(y2,4)])