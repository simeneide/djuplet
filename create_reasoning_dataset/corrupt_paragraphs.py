#!/usr/bin/env python3
"""
This script reads a JSONL file containing Wikipedia paragraphs,
applies a series of random "corruptions" (modifications) to the text,
and writes the results to a new JSONL file.

Each record in the input file should be a JSON object with at least a 'text' field.
The script adds two new fields:
  - "corrupt": the corrupted version of the original text.
  - "corrupt_level": an integer (0-9) representing the level/type of corruption applied.

Usage:
  python corrupt_paragraphs.py --input_file input.jsonl --output_file output.jsonl

Dependencies:
  - tqdm (for progress bars)
    pip install tqdm
"""

import json
import random
import argparse
import re
from tqdm import tqdm

# The following import seems redundant because the function is defined below.
# If there is an external corrupter module intended, remove the local definition.
# from corrupter import corrupt_paragraph

def find_valid_punctuation_positions(text: str, punctuation: set = {',', '.', '?', ':', '!'}) -> list:
    """
    Finds positions in the text where a punctuation character can be inserted safely,
    avoiding consecutive punctuation marks.
    
    Parameters:
        text (str): The text to search.
        punctuation (set): Set of punctuation characters to consider.
    
    Returns:
        list: A list of integer positions where punctuation can be inserted.
    """
    positions = []
    for i in range(1, len(text)):
        if text[i] not in punctuation and text[i - 1] not in punctuation:
            positions.append(i)
    return positions

def total_punctuation(text: str, punctuation: set = {',', '.', '?', ':', '!'}) -> int:
    """
    Counts the total number of punctuation characters in the text.
    
    Parameters:
        text (str): The text in which to count punctuation.
        punctuation (set): Set of punctuation characters to count.
    
    Returns:
        int: The total count of punctuation characters in the text.
    """
    return sum(1 for char in text if char in punctuation)

def corrupt_paragraph(paragraph: str, level: int) -> str:
    """
    Applies a corruption transformation to the paragraph based on the specified level.
    
    The corruption level determines which transformation to apply:
      0: No change.
      1: Move one punctuation mark.
      2: Move multiple punctuation marks.
      3: Remove one punctuation mark.
      4: Remove multiple punctuation marks.
      5: Add punctuation marks at random positions.
      6: Randomize character casing.
      7: Remove all punctuation.
      8: Remove all punctuation and convert to lowercase.
      9: Apply a random combination of two transformations.
    
    Parameters:
        paragraph (str): The original text to be corrupted.
        level (int): The corruption level (0-9).
    
    Returns:
        str: The corrupted text.
    """
    PUNCTUATION = {',', '.', '?', ':', '!'}

    if level == 0:
        return paragraph

    def move_one_punctuation(text: str) -> str:
        """
        Removes one punctuation character from a random position and reinserts it at a random valid position.
        """
        punctuation_positions = [m.start() for m in re.finditer(r'[,.?:!]', text)]
        if not punctuation_positions:
            return text

        pos = random.choice(punctuation_positions)
        char = text[pos]
        # Remove the punctuation character
        text = text[:pos] + text[pos + 1:]
        valid_positions = find_valid_punctuation_positions(text, PUNCTUATION)
        if valid_positions:
            new_pos = random.choice(valid_positions)
            text = text[:new_pos] + char + text[new_pos:]
        return text

    def move_multiple_punctuation(text: str) -> str:
        """
        Applies the move_one_punctuation transformation multiple times.
        """
        total_punct = total_punctuation(text, PUNCTUATION)
        if total_punct <= 1:
            return text
        num = random.randint(1, min(3, total_punct - 1))
        for _ in range(num):
            text = move_one_punctuation(text)
        return text

    def remove_one_punctuation(text: str) -> str:
        """
        Removes a single random punctuation character from the text.
        """
        punctuation_positions = [m.start() for m in re.finditer(r'[,.?:!]', text)]
        if not punctuation_positions:
            return text
        pos = random.choice(punctuation_positions)
        return text[:pos] + text[pos + 1:]

    def remove_multiple_punctuation(text: str) -> str:
        """
        Removes multiple punctuation characters from the text.
        """
        total_punct = total_punctuation(text, PUNCTUATION)
        if total_punct == 0:
            return text
        num = random.randint(1, min(3, total_punct))
        for _ in range(num):
            text = remove_one_punctuation(text)
        return text

    def add_punctuation(text: str) -> str:
        """
        Adds random punctuation characters at valid positions in the text.
        """
        num = random.randint(1, 5)
        valid_positions = find_valid_punctuation_positions(text, PUNCTUATION)

        if not valid_positions:
            return text

        for _ in range(num):
            if not valid_positions:
                break
            pos = random.choice(valid_positions)
            char = random.choice(list(PUNCTUATION))
            text = text[:pos] + char + text[pos:]
            # Adjust valid positions dynamically: shift positions that come after the inserted punctuation.
            valid_positions = [p + 1 if p >= pos else p for p in valid_positions]

        return text

    def randomize_casing(text: str) -> str:
        """
        Randomly changes the casing of characters in the text.
        Each character has a 30% chance to be converted to uppercase; otherwise, it is lowercase.
        """
        return ''.join(c.upper() if random.random() < 0.3 else c.lower() for c in text)

    def remove_all_punctuation(text: str) -> str:
        """
        Removes all punctuation characters from the text.
        """
        return re.sub(r'[,.?:!]', '', text)

    def lowercase_no_punctuation(text: str) -> str:
        """
        Removes all punctuation characters from the text and converts it to lowercase.
        """
        return remove_all_punctuation(text).lower()

    def random_combo(text: str) -> str:
        """
        Applies a random combination of two different transformations to the text.
        This function randomly selects two transformation functions from the list and applies them sequentially.
        """
        functions = [
            move_one_punctuation,
            move_multiple_punctuation,
            remove_one_punctuation,
            remove_multiple_punctuation,
            add_punctuation,
            randomize_casing,
            remove_all_punctuation,
            lowercase_no_punctuation
        ]
        for _ in range(5):
            f1, f2 = random.sample(functions, 2)
            modified_text = f2(f1(text))
            if modified_text != text:
                return modified_text
        return text

    transformations = {
        1: move_one_punctuation,
        2: move_multiple_punctuation,
        3: remove_one_punctuation,
        4: remove_multiple_punctuation,
        5: add_punctuation,
        6: randomize_casing,
        7: remove_all_punctuation,
        8: lowercase_no_punctuation,
        9: random_combo
    }

    # Apply the transformation based on the provided corruption level.
 

