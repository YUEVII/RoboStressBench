from .dataset import QuestionSample

_MCQ_FOOTER = (
    "\n\nAnswer with exactly one uppercase letter only. "
    "Do not output anything else."
)


def get_model_prompt(sample: QuestionSample) -> str:
    stem = sample.instruction.strip()
    if sample.gt_type != "mcq" or not sample.options:
        return stem
    lines = []
    for i, option in enumerate(sample.options):
        if i >= 26:
            break
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {option}")
    option_block = "\n".join(lines)
    return f"{stem}\n\n{option_block}{_MCQ_FOOTER}"
